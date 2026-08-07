from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from smartpet import __version__
from smartpet.data.dataset import PairedMNIPatchDataset
from smartpet.metrics import image_quality_metrics
from smartpet.models import (
    OUTPUT_MODES,
    PatchDiscriminator3D,
    SmartPETGenerator,
    initialize_gan_weights,
    initialize_identity_residual_head,
)

from .checkpoint import load_checkpoint, save_checkpoint
from .distributed import (
    DistributedEvalSampler,
    Runtime,
    barrier,
    cleanup,
    setup,
    unwrap,
    wrap,
)
from .precision import (
    autocast_context,
    optimizer_max_step,
    require_optimizer_advanced,
    resolve_precision,
)
from .preview import PREVIEW_SELECTIONS, run_epoch_preview


@dataclass
class TrainConfig:
    train_csv: str
    val_csv: str
    mni_reference: str
    out_dir: str
    backend: str = "auto"
    patch_size: tuple[int, int, int] = (128, 128, 128)
    base_channels: int = 32
    attention_levels: tuple[int, ...] = (2, 3)
    batch_size: int = 1
    epochs: int = 35
    lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    lambda_gan: float = 1.0
    gan_loss: str = "lsgan"
    grad_clip: float = 1.0
    num_workers: int = 4
    val_batches: int = 0
    amp: bool = True
    amp_dtype: str = "auto"
    seed: int = 2023
    resume: str | None = None
    init_checkpoint: str | None = None
    decay_start_epoch: int = 9
    max_steps: int = 0
    smoke_steps: int = 0
    checkpoint_every_steps: int = 250
    keep_step_checkpoints: int = 2
    log_every_steps: int = 25
    asinh_scale: float = 1.0
    output_mode: str = "positive_softplus_residual"
    initialization: str = "normal_0.02"
    residual_head_initialization: str = "zero_identity"
    train_patch_mode: str = "random_foreground"
    val_patch_mode: str = "center"
    min_foreground_fraction: float = 0.05
    foreground_threshold: float = 0.0
    foreground_attempts: int = 16
    preview_every_epochs: int = 1
    preview_at_start: bool = True
    preview_subject_id: str | None = None
    preview_selection: str = "fixed_random"
    preview_seed: int = 5104
    preview_stride: tuple[int, int, int] = (64, 64, 64)
    preview_save_nifti: bool = True
    preview_vgg19_weights: str | None = None


def seed_all(seed: int, rank: int) -> None:
    value = int(seed) + 100_003 * int(rank)
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def _write_run_provenance(config: TrainConfig, runtime: Runtime, out_dir: Path) -> None:
    if not runtime.is_main:
        return
    global_batch = int(config.batch_size) * int(runtime.world_size)
    payload = {
        "format_version": 1,
        "smartpet_version": __version__,
        "config": asdict(config),
        "runtime": {
            "backend": config.backend,
            "world_size": runtime.world_size,
            "per_rank_batch_size": config.batch_size,
            "global_batch_size": global_batch,
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "initialization": {
            "resume_checkpoint": (str(Path(config.resume).resolve()) if config.resume else None),
            "resume_checkpoint_sha256": (_sha256(config.resume) if config.resume else None),
            "init_checkpoint": (
                str(Path(config.init_checkpoint).resolve()) if config.init_checkpoint else None
            ),
            "init_checkpoint_sha256": (
                _sha256(config.init_checkpoint) if config.init_checkpoint else None
            ),
        },
        "inputs": {
            "train_csv": str(Path(config.train_csv).resolve()),
            "train_csv_sha256": _sha256(config.train_csv),
            "val_csv": str(Path(config.val_csv).resolve()),
            "val_csv_sha256": _sha256(config.val_csv),
            "mni_reference": str(Path(config.mni_reference).resolve()),
            "mni_reference_sha256": _sha256(config.mni_reference),
        },
        "optimization_semantics": {
            "optimizer_step_unit": "one synchronized DDP global batch",
            "subject_exposure_unit": "one source-target pair consumed on one rank",
            "ddp_gradient_reduction": "mean across ranks",
        },
    }
    _write_json(out_dir / "run_manifest.json", payload)
    _write_json(out_dir / "config.json", asdict(config))


def _reduce_mean(sum_value: float, count: float, runtime: Runtime) -> float:
    tensor = torch.tensor([sum_value, count], device=runtime.device, dtype=torch.float64)
    if runtime.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item() / max(1.0, tensor[1].item()))


def _reduce_metric_sums(
    sums: dict[str, float],
    count: float,
    runtime: Runtime,
) -> dict[str, float]:
    keys = sorted(sums)
    tensor = torch.tensor(
        [*[sums[key] for key in keys], count],
        device=runtime.device,
        dtype=torch.float64,
    )
    if runtime.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    denominator = max(1.0, tensor[-1].item())
    return {key: float(tensor[index].item() / denominator) for index, key in enumerate(keys)}


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _backward_and_prepare_step(
    *,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> torch.Tensor:
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        grad_clip,
        error_if_nonfinite=True,
    )
    if scaler.is_enabled():
        scaler.step(optimizer)
    else:
        optimizer.step()
    return grad_norm


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    fieldnames = list(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exists:
        with path.open(newline="") as handle:
            existing = csv.DictReader(handle).fieldnames
        if existing != fieldnames:
            raise RuntimeError(
                f"CSV schema mismatch for {path}: existing={existing}, new={fieldnames}"
            )
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _clone_checkpoint(source: Path, destination: Path) -> None:
    """Create an atomic restart point without serializing the large payload twice."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_old_step_checkpoints(directory: Path, keep: int) -> None:
    if keep <= 0:
        return
    paths = sorted(directory.glob("step_*.pt"))
    for path in paths[:-keep]:
        path.unlink(missing_ok=True)


def _checkpoint_training_state(
    sums: dict[str, float],
    batches: int,
) -> dict[str, Any]:
    return {"epoch_sums": dict(sums), "epoch_batches": int(batches)}


def _save_progress_checkpoint(
    *,
    out_dir: Path,
    generator: nn.Module,
    discriminator: nn.Module,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    g_scheduler: object,
    d_scheduler: object,
    epoch: int,
    batch_in_epoch: int,
    epoch_complete: bool,
    global_step: int,
    samples_seen: int,
    runtime: Runtime,
    config: TrainConfig,
    best: float,
    precision_payload: dict[str, object],
    training_sums: dict[str, float],
    training_batches: int,
    save_named_step: bool,
) -> None:
    kwargs = dict(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_opt,
        d_optimizer=d_opt,
        scaler=scaler,
        g_scheduler=g_scheduler,
        d_scheduler=d_scheduler,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
        epoch_complete=epoch_complete,
        global_step=global_step,
        samples_seen=samples_seen,
        runtime=runtime,
        config=asdict(config),
        best_metric=best,
        precision=precision_payload,
        training_state=_checkpoint_training_state(training_sums, training_batches),
    )
    checkpoint_dir = out_dir / "checkpoints"
    last_path = checkpoint_dir / "last.pt"
    save_checkpoint(last_path, **kwargs)
    if save_named_step and runtime.is_main:
        _clone_checkpoint(last_path, checkpoint_dir / f"step_{global_step:08d}.pt")
        _remove_old_step_checkpoints(checkpoint_dir, config.keep_step_checkpoints)
    barrier(runtime)


def _validate(
    *,
    generator: nn.Module,
    val_loader: DataLoader,
    config: TrainConfig,
    runtime: Runtime,
    precision,
) -> dict[str, float]:
    generator.eval()
    sums: dict[str, float] = {}
    count = 0.0
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            if config.val_batches > 0 and index >= config.val_batches:
                break
            source = batch["source"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            with autocast_context(precision, runtime.device):
                prediction = generator(source)
            prediction = prediction.float()
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Validation prediction contains non-finite values")
            negative = float((prediction < 0).float().mean().item())
            target_suv = torch.sinh(target.float()) * float(config.asinh_scale)
            prediction_suv = torch.sinh(prediction) * float(config.asinh_scale)
            source_suv = torch.sinh(source.float()) * float(config.asinh_scale)
            if not torch.isfinite(prediction_suv).all():
                raise RuntimeError("Validation SUV prediction contains non-finite values")
            pred_metrics = image_quality_metrics(prediction_suv, target_suv)
            input_metrics = image_quality_metrics(source_suv, target_suv)
            batch_size = float(source.shape[0])
            values = {
                "val_l1_normalized": float(nn.functional.l1_loss(prediction, target).item()),
                "val_negative_fraction": negative,
                **{f"val_prediction_{key}_suv": value for key, value in pred_metrics.items()},
                **{f"val_input_{key}_suv": value for key, value in input_metrics.items()},
            }
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + float(value) * batch_size
            count += batch_size
    if count <= 0:
        raise RuntimeError("Validation completed without evaluating a batch")
    return _reduce_metric_sums(sums, count, runtime)


def run(config: TrainConfig) -> Path:
    if config.resume and config.init_checkpoint:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    if config.backend not in {"auto", "single", "ddp"}:
        raise ValueError("backend must be auto, single, or ddp")
    if config.base_channels <= 0:
        raise ValueError("base_channels must be positive")
    levels = tuple(int(level) for level in config.attention_levels)
    if len(set(levels)) != len(levels) or any(level < 0 or level > 6 for level in levels):
        raise ValueError("attention_levels must contain unique encoder levels in [0, 6]")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.decay_start_epoch < 0:
        raise ValueError("decay_start_epoch must be non-negative")
    if config.lr <= 0:
        raise ValueError("lr must be positive")
    if config.lambda_l1 < 0 or config.lambda_gan < 0:
        raise ValueError("loss weights must be non-negative")
    if config.grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    if config.val_batches < 0 or config.max_steps < 0 or config.smoke_steps < 0:
        raise ValueError("val_batches and step limits must be non-negative")
    if config.checkpoint_every_steps < 0 or config.keep_step_checkpoints < 0:
        raise ValueError("checkpoint controls must be non-negative")
    if config.preview_every_epochs < 0:
        raise ValueError("preview_every_epochs must be non-negative")
    if config.asinh_scale <= 0:
        raise ValueError("asinh_scale must be positive")
    if any(int(v) <= 0 for v in (*config.patch_size, *config.preview_stride)):
        raise ValueError("patch_size and preview_stride must contain positive integers")
    if config.output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unsupported output_mode={config.output_mode!r}; expected {OUTPUT_MODES}")
    if config.output_mode == "linear":
        raise ValueError(
            "Long-run training with unconstrained linear output is disabled. "
            "Use positive_softplus_residual for non-negative asinh-SUV targets."
        )
    if config.gan_loss != "lsgan":
        raise ValueError("The modernized SMART-PET baseline requires gan_loss='lsgan'")
    if config.initialization != "normal_0.02":
        raise ValueError("The modernized baseline requires initialization='normal_0.02'")
    if config.residual_head_initialization != "zero_identity":
        raise ValueError(
            "The modernized baseline requires "
            "residual_head_initialization='zero_identity'"
        )
    if config.preview_selection not in PREVIEW_SELECTIONS:
        raise ValueError(f"Unsupported preview_selection={config.preview_selection!r}")

    runtime = setup(config.backend)
    try:
        seed_all(config.seed, runtime.rank)
        out_dir = Path(config.out_dir)
        if runtime.is_main:
            out_dir.mkdir(parents=True, exist_ok=True)
        barrier(runtime)
        _write_run_provenance(config, runtime, out_dir)

        train_set = PairedMNIPatchDataset(
            config.train_csv,
            config.mni_reference,
            config.patch_size,
            config.seed,
            patch_mode=config.train_patch_mode,
            min_foreground_fraction=config.min_foreground_fraction,
            foreground_threshold=config.foreground_threshold,
            foreground_attempts=config.foreground_attempts,
        )
        val_set = PairedMNIPatchDataset(
            config.val_csv,
            config.mni_reference,
            config.patch_size,
            config.seed + 17,
            patch_mode=config.val_patch_mode,
            min_foreground_fraction=0.0,
            foreground_threshold=config.foreground_threshold,
        )
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
        val_sampler = DistributedEvalSampler(val_set, runtime) if runtime.distributed else None
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=runtime.device.type == "cuda",
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=runtime.device.type == "cuda",
            persistent_workers=False,
        )

        raw_generator = SmartPETGenerator(
            config.base_channels,
            attention_levels=config.attention_levels,
            output_mode=config.output_mode,
        ).to(runtime.device)
        raw_discriminator = PatchDiscriminator3D(config.base_channels).to(runtime.device)
        raw_generator.apply(initialize_gan_weights)
        raw_discriminator.apply(initialize_gan_weights)
        initialize_identity_residual_head(raw_generator.output)
        if config.init_checkpoint:
            checkpoint = load_checkpoint(
                config.init_checkpoint,
                generator=raw_generator,
                discriminator=raw_discriminator,
                device=runtime.device,
                require_discriminator=True,
            )
            checkpoint_config = checkpoint.get("config", {})
            architecture_fields = (
                "base_channels",
                "attention_levels",
                "output_mode",
                "asinh_scale",
            )
            current_config = asdict(config)
            mismatches: dict[str, tuple[Any, Any]] = {}
            for field in architecture_fields:
                checkpoint_value = checkpoint_config.get(field)
                current_value = current_config[field]
                if isinstance(checkpoint_value, list):
                    checkpoint_value = tuple(checkpoint_value)
                if isinstance(current_value, list):
                    current_value = tuple(current_value)
                if checkpoint_value != current_value:
                    mismatches[field] = (checkpoint_value, current_value)
            if mismatches:
                raise RuntimeError(
                    "Fine-tuning checkpoint architecture is incompatible with the current "
                    f"configuration: {mismatches}"
                )
        generator = wrap(raw_generator, runtime)
        discriminator = wrap(raw_discriminator, runtime)
        g_opt = torch.optim.Adam(
            generator.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        d_opt = torch.optim.Adam(
            discriminator.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

        def lr_factor(epoch: int) -> float:
            if epoch < config.decay_start_epoch:
                return 1.0
            denominator = max(1, config.epochs - config.decay_start_epoch)
            return max(0.0, 1.0 - (epoch - config.decay_start_epoch) / denominator)

        g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=lr_factor)
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_opt, lr_lambda=lr_factor)
        precision = resolve_precision(
            amp=config.amp,
            amp_dtype=config.amp_dtype,
            device=runtime.device,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=precision.scaler_enabled)
        adversarial = nn.MSELoss()
        l1 = nn.L1Loss()

        start_epoch = 0
        resume_batch = 0
        global_step = 0
        samples_seen = 0
        best = float("inf")
        resumed_sums: dict[str, float] = {}
        resumed_batches = 0
        if config.resume:
            checkpoint = load_checkpoint(
                config.resume,
                generator=generator,
                discriminator=discriminator,
                device=runtime.device,
                g_optimizer=g_opt,
                d_optimizer=d_opt,
                scaler=scaler,
                g_scheduler=g_scheduler,
                d_scheduler=d_scheduler,
                validate_optimizer_progress=True,
                restore_rank_rng=runtime.rank,
                require_discriminator=True,
            )
            checkpoint_config = checkpoint.get("config", {})
            checkpoint_world_size = int(checkpoint.get("world_size", 1))
            if checkpoint_world_size != runtime.world_size:
                raise RuntimeError(
                    "Exact resume requires the same world size: "
                    f"checkpoint={checkpoint_world_size}, current={runtime.world_size}"
                )
            compatibility_fields = (
                "backend",
                "train_csv",
                "val_csv",
                "mni_reference",
                "output_mode",
                "base_channels",
                "attention_levels",
                "batch_size",
                "patch_size",
                "gan_loss",
                "lr",
                "beta1",
                "beta2",
                "lambda_l1",
                "lambda_gan",
                "grad_clip",
                "asinh_scale",
                "decay_start_epoch",
                "initialization",
                "residual_head_initialization",
                "seed",
                "train_patch_mode",
                "min_foreground_fraction",
                "foreground_threshold",
                "foreground_attempts",
            )
            current_config = asdict(config)
            for field in compatibility_fields:
                checkpoint_value = checkpoint_config.get(field)
                current_value = current_config.get(field)
                if field in {"train_csv", "val_csv", "mni_reference"}:
                    checkpoint_value = str(Path(checkpoint_value).resolve())
                    current_value = str(Path(current_value).resolve())
                if isinstance(checkpoint_value, list):
                    checkpoint_value = tuple(checkpoint_value)
                if isinstance(current_value, list):
                    current_value = tuple(current_value)
                if checkpoint_value != current_value:
                    raise RuntimeError(
                        f"Resume checkpoint {field} mismatch: "
                        f"checkpoint={checkpoint_value!r}, current={current_value!r}"
                    )
            checkpoint_precision = str(checkpoint.get("precision", {}).get("resolved", "fp32"))
            if checkpoint_precision != precision.resolved:
                raise RuntimeError(
                    "Exact resume requires the same resolved precision: "
                    f"checkpoint={checkpoint_precision}, current={precision.resolved}"
                )
            checkpoint_epoch = int(checkpoint["epoch"])
            if bool(checkpoint.get("epoch_complete", True)):
                start_epoch = checkpoint_epoch + 1
            else:
                start_epoch = checkpoint_epoch
                resume_batch = int(checkpoint.get("batch_in_epoch", 0))
                states = checkpoint.get("training_states")
                if isinstance(states, list) and runtime.rank < len(states):
                    state = states[runtime.rank]
                else:
                    state = checkpoint.get("training_state", {})
                resumed_sums = {
                    str(k): float(v) for k, v in state.get("epoch_sums", {}).items()
                }
                resumed_batches = int(state.get("epoch_batches", 0))
            global_step = int(checkpoint["global_step"])
            samples_seen = int(checkpoint.get("samples_seen", 0))
            best = float(checkpoint.get("best_metric", best))

        if runtime.is_main:
            print(f"precision_requested={precision.requested}", flush=True)
            print(f"precision_resolved={precision.resolved}", flush=True)
            print(f"grad_scaler_enabled={scaler.is_enabled()}", flush=True)
            print(f"gan_loss={config.gan_loss}", flush=True)
            print(f"initialization={config.initialization}", flush=True)
            print(
                f"residual_head_initialization={config.residual_head_initialization}",
                flush=True,
            )
            print(f"output_mode={config.output_mode}", flush=True)
            print(f"attention_levels={list(config.attention_levels)}", flush=True)
            print(f"init_checkpoint={config.init_checkpoint or ''}", flush=True)
            print(f"per_rank_batch_size={config.batch_size}", flush=True)
            print(f"global_batch_size={config.batch_size * runtime.world_size}", flush=True)

        metrics_path = out_dir / "metrics.csv"
        step_metrics_path = out_dir / "train_steps.csv"

        if (
            config.preview_at_start
            and start_epoch == 0
            and global_step == 0
            and runtime.is_main
        ):
            run_epoch_preview(
                model=unwrap(generator),
                val_manifest=config.val_csv,
                mni_reference=config.mni_reference,
                out_dir=out_dir,
                device=runtime.device,
                precision=precision,
                patch_size=config.patch_size,
                stride=config.preview_stride,
                asinh_scale=config.asinh_scale,
                epoch=-1,
                global_step=0,
                subject_id=config.preview_subject_id,
                selection=config.preview_selection,
                seed=config.preview_seed,
                save_nifti=config.preview_save_nifti,
                vgg19_weights=config.preview_vgg19_weights,
            )
        barrier(runtime)

        hard_limit = int(config.max_steps or config.smoke_steps)
        if hard_limit > 0 and global_step >= hard_limit:
            raise RuntimeError(
                f"Checkpoint is already at global_step={global_step}, which meets or exceeds "
                f"the requested max_steps={hard_limit}. Increase --max-steps or remove the limit."
            )
        stop_training = False
        for epoch in range(start_epoch, config.epochs):
            train_set.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            generator.train()
            discriminator.train()
            train_sums = dict(resumed_sums) if epoch == start_epoch else {}
            train_batches = int(resumed_batches) if epoch == start_epoch else 0
            consumed_batches = resume_batch if epoch == start_epoch else 0
            epoch_start = time.monotonic()

            for batch_index, batch in enumerate(train_loader):
                if batch_index < consumed_batches:
                    continue
                source = batch["source"].to(runtime.device, non_blocking=True)
                target = batch["target"].to(runtime.device, non_blocking=True)
                foreground_fraction = float(batch["foreground_fraction"].float().mean().item())
                d_before = optimizer_max_step(d_opt)
                g_before = optimizer_max_step(g_opt)

                _set_requires_grad(discriminator, True)
                d_opt.zero_grad(set_to_none=True)
                with autocast_context(precision, runtime.device):
                    with torch.no_grad():
                        fake_for_d = unwrap(generator)(source)
                    real_logits = discriminator(source, target)
                    fake_logits = discriminator(source, fake_for_d)
                    d_real = adversarial(real_logits, torch.ones_like(real_logits))
                    d_fake = adversarial(fake_logits, torch.zeros_like(fake_logits))
                    d_loss = 0.5 * (d_real + d_fake)
                d_grad = _backward_and_prepare_step(
                    loss=d_loss,
                    optimizer=d_opt,
                    parameters=discriminator.parameters(),
                    scaler=scaler,
                    grad_clip=config.grad_clip,
                )

                _set_requires_grad(discriminator, False)
                g_opt.zero_grad(set_to_none=True)
                with autocast_context(precision, runtime.device):
                    fake = generator(source)
                    fake_logits = unwrap(discriminator)(source, fake)
                    g_adv = adversarial(fake_logits, torch.ones_like(fake_logits))
                    g_l1 = l1(fake, target)
                    g_loss = config.lambda_gan * g_adv + config.lambda_l1 * g_l1
                g_grad = _backward_and_prepare_step(
                    loss=g_loss,
                    optimizer=g_opt,
                    parameters=generator.parameters(),
                    scaler=scaler,
                    grad_clip=config.grad_clip,
                )
                _set_requires_grad(discriminator, True)

                if scaler.is_enabled():
                    scaler.update()

                d_after = require_optimizer_advanced(
                    d_opt,
                    before=d_before,
                    name="discriminator",
                )
                g_after = require_optimizer_advanced(
                    g_opt,
                    before=g_before,
                    name="generator",
                )
                if d_after != g_after:
                    raise RuntimeError(
                        "Generator and discriminator optimizer counters diverged: "
                        f"generator={g_after}, discriminator={d_after}"
                    )
                global_step += 1
                if g_after != global_step:
                    raise RuntimeError(
                        "Successful optimizer updates do not match global_step: "
                        f"optimizer_updates={g_after}, global_step={global_step}"
                    )
                consumed_batches = batch_index + 1
                train_batches += 1
                samples_seen += int(source.shape[0]) * int(runtime.world_size)
                values = {
                    "train_g_total": float(g_loss.detach().item()),
                    "train_g_adversarial": float(g_adv.detach().item()),
                    "train_g_l1": float(g_l1.detach().item()),
                    "train_d_total": float(d_loss.detach().item()),
                    "train_d_real": float(d_real.detach().item()),
                    "train_d_fake": float(d_fake.detach().item()),
                    "train_g_grad_norm": float(g_grad.detach().item()),
                    "train_d_grad_norm": float(d_grad.detach().item()),
                    "train_negative_fraction": float((fake.detach() < 0).float().mean().item()),
                    "train_foreground_fraction": foreground_fraction,
                }
                for key, value in values.items():
                    train_sums[key] = train_sums.get(key, 0.0) + value

                if config.log_every_steps > 0 and global_step % config.log_every_steps == 0:
                    reduced_step = _reduce_metric_sums(values, 1.0, runtime)
                    if runtime.is_main:
                        _append_csv(
                            step_metrics_path,
                            {
                                "epoch": epoch,
                                "batch_in_epoch": consumed_batches,
                                "global_step": global_step,
                                "samples_seen": samples_seen,
                                "lr_g": g_opt.param_groups[0]["lr"],
                                "lr_d": d_opt.param_groups[0]["lr"],
                                "precision": precision.resolved,
                                **reduced_step,
                            },
                        )
                        print(
                            f"epoch={epoch} batch={consumed_batches}/{len(train_loader)} "
                            f"step={global_step} samples_seen={samples_seen} "
                            f"g={reduced_step['train_g_total']:.6f} "
                            f"d={reduced_step['train_d_total']:.6f} "
                            f"g_l1={reduced_step['train_g_l1']:.6f}",
                            flush=True,
                        )

                if (
                    config.checkpoint_every_steps > 0
                    and global_step % config.checkpoint_every_steps == 0
                ):
                    _save_progress_checkpoint(
                        out_dir=out_dir,
                        generator=generator,
                        discriminator=discriminator,
                        g_opt=g_opt,
                        d_opt=d_opt,
                        scaler=scaler,
                        g_scheduler=g_scheduler,
                        d_scheduler=d_scheduler,
                        epoch=epoch,
                        batch_in_epoch=consumed_batches,
                        epoch_complete=False,
                        global_step=global_step,
                        samples_seen=samples_seen,
                        runtime=runtime,
                        config=config,
                        best=best,
                        precision_payload=precision.as_dict(),
                        training_sums=train_sums,
                        training_batches=train_batches,
                        save_named_step=True,
                    )

                if hard_limit > 0 and global_step >= hard_limit:
                    stop_training = True
                    break

            if train_batches == 0:
                raise RuntimeError("Training epoch completed without a successful optimizer update")
            epoch_complete = consumed_batches >= len(train_loader)
            train_metrics = _reduce_metric_sums(train_sums, train_batches, runtime)
            val_metrics = _validate(
                generator=generator,
                val_loader=val_loader,
                config=config,
                runtime=runtime,
                precision=precision,
            )
            selection_metric = val_metrics["val_l1_normalized"]
            improved = selection_metric < best
            if improved:
                best = selection_metric

            if epoch_complete:
                g_scheduler.step()
                d_scheduler.step()

            _save_progress_checkpoint(
                out_dir=out_dir,
                generator=generator,
                discriminator=discriminator,
                g_opt=g_opt,
                d_opt=d_opt,
                scaler=scaler,
                g_scheduler=g_scheduler,
                d_scheduler=d_scheduler,
                epoch=epoch,
                batch_in_epoch=consumed_batches,
                epoch_complete=epoch_complete,
                global_step=global_step,
                samples_seen=samples_seen,
                runtime=runtime,
                config=config,
                best=best,
                precision_payload=precision.as_dict(),
                training_sums=train_sums,
                training_batches=train_batches,
                save_named_step=False,
            )
            if improved:
                save_checkpoint(
                    out_dir / "checkpoints" / "best.pt",
                    generator=generator,
                    discriminator=discriminator,
                    g_optimizer=g_opt,
                    d_optimizer=d_opt,
                    scaler=scaler,
                    g_scheduler=g_scheduler,
                    d_scheduler=d_scheduler,
                    epoch=epoch,
                    batch_in_epoch=consumed_batches,
                    epoch_complete=epoch_complete,
                    global_step=global_step,
                    samples_seen=samples_seen,
                    runtime=runtime,
                    config=asdict(config),
                    best_metric=best,
                    precision=precision.as_dict(),
                    training_state=_checkpoint_training_state(train_sums, train_batches),
                )

            if runtime.is_main:
                row = {
                    "epoch": epoch,
                    "epoch_complete": int(epoch_complete),
                    "batch_in_epoch": consumed_batches,
                    "batches_per_epoch": len(train_loader),
                    "global_step": global_step,
                    "samples_seen": samples_seen,
                    "g_optimizer_updates": optimizer_max_step(g_opt),
                    "d_optimizer_updates": optimizer_max_step(d_opt),
                    "per_rank_batch_size": config.batch_size,
                    "global_batch_size": config.batch_size * runtime.world_size,
                    "lr_g": g_opt.param_groups[0]["lr"],
                    "lr_d": d_opt.param_groups[0]["lr"],
                    "epoch_seconds": time.monotonic() - epoch_start,
                    "precision": precision.resolved,
                    "best_val_l1_normalized": best,
                    **train_metrics,
                    **val_metrics,
                }
                _append_csv(metrics_path, row)
                print(
                    f"epoch={epoch} epoch_complete={int(epoch_complete)} "
                    f"step={global_step} samples_seen={samples_seen} "
                    f"g_updates={optimizer_max_step(g_opt)} d_updates={optimizer_max_step(d_opt)} "
                    f"train_g={train_metrics['train_g_total']:.6f} "
                    f"train_d={train_metrics['train_d_total']:.6f} "
                    f"val_l1={val_metrics['val_l1_normalized']:.6f} "
                    f"val_psnr_suv={val_metrics['val_prediction_psnr_db_suv']:.4f} "
                    f"val_ssim_suv={val_metrics['val_prediction_ssim_suv']:.5f} "
                    f"precision={precision.resolved}",
                    flush=True,
                )

            should_preview = (
                config.preview_every_epochs > 0
                and ((epoch + 1) % config.preview_every_epochs == 0 or stop_training)
            )
            if should_preview and runtime.is_main:
                run_epoch_preview(
                    model=unwrap(generator),
                    val_manifest=config.val_csv,
                    mni_reference=config.mni_reference,
                    out_dir=out_dir,
                    device=runtime.device,
                    precision=precision,
                    patch_size=config.patch_size,
                    stride=config.preview_stride,
                    asinh_scale=config.asinh_scale,
                    epoch=epoch,
                    global_step=global_step,
                    subject_id=config.preview_subject_id,
                    selection=config.preview_selection,
                    seed=config.preview_seed,
                    save_nifti=config.preview_save_nifti,
                    vgg19_weights=config.preview_vgg19_weights,
                )
            barrier(runtime)
            resume_batch = 0
            resumed_sums = {}
            resumed_batches = 0
            if stop_training:
                break

        barrier(runtime)
        return out_dir
    finally:
        cleanup(runtime)
