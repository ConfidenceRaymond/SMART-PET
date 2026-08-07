from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from smartpet.checkpoint_io import safe_torch_load, sha256_file
from smartpet.data.nifti import MNIContract, load_mni_volume, save_like
from smartpet.inference.domains import (
    prepare_inference_input,
    prepare_shared_inference_outputs,
)
from smartpet.inference.outputs import prediction_identifier
from smartpet.inference.sliding_window import predict_volume
from smartpet.inference.weights import SUPPORTED_INFERENCE_WEIGHTS_FORMATS
from smartpet.models import OUTPUT_MODES, SmartPETGenerator, normalize_architecture_config
from smartpet.training.precision import PrecisionPolicy, resolve_precision

_REQUIRED_INFERENCE_CONFIG = (
    "base_channels",
    "attention_levels",
    "output_mode",
    "asinh_scale",
    "patch_size",
)


@dataclass(frozen=True)
class PredictionResult:
    reference_image: nib.Nifti1Image
    normalized: np.ndarray
    suv: np.ndarray
    negative_count: int
    prediction_id: str


def _inference_config(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    artifact_type = payload.get("artifact_type")
    allowed_types = {"smartpet_training_checkpoint", "smartpet_inference_weights"}
    if artifact_type not in allowed_types:
        raise RuntimeError(
            f"{path} has unsupported artifact_type={artifact_type!r}; "
            f"expected one of {sorted(allowed_types)}"
        )
    format_version = int(payload.get("format_version", 0))
    if artifact_type == "smartpet_training_checkpoint":
        if format_version < 4:
            raise RuntimeError(
                f"{path} format_version={format_version} predates the supported "
                "smartpet_training_checkpoint format 4"
            )
        allow_v030_architecture = True
    else:
        if format_version not in SUPPORTED_INFERENCE_WEIGHTS_FORMATS:
            raise RuntimeError(
                f"{path} has unsupported inference weights format_version={format_version}; "
                f"expected one of {SUPPORTED_INFERENCE_WEIGHTS_FORMATS}"
            )
        allow_v030_architecture = format_version == 1

    raw = payload.get("config")
    if not isinstance(raw, dict):
        raise RuntimeError(f"{path} does not contain a valid configuration mapping")
    missing = [field for field in _REQUIRED_INFERENCE_CONFIG if raw.get(field) is None]
    if missing:
        raise RuntimeError(
            f"{path} is missing inference-critical configuration fields: {missing}. "
            "SMART-PET will not guess values that determine architecture or SUV scaling."
        )
    try:
        architecture = normalize_architecture_config(
            raw,
            allow_v030_defaults=allow_v030_architecture,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{path} has invalid architecture configuration: {exc}") from exc
    if not payload.get("generator_state"):
        raise RuntimeError(f"{path} contains no generator_state")
    return {**raw, **architecture}


class InferenceEngine:
    """Load one checkpoint once and apply deterministic sliding-window inference."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        mni_reference: str | Path,
        patch_size: tuple[int, int, int] | None = None,
        stride: tuple[int, int, int] | None = None,
        asinh_scale: float | None = None,
        amp: bool = True,
        amp_dtype: str = "auto",
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.mni_reference = Path(mni_reference)
        if not self.mni_reference.is_file():
            raise FileNotFoundError(self.mni_reference)

        raw = safe_torch_load(self.checkpoint_path)
        config = _inference_config(raw, self.checkpoint_path)
        self.config = config

        recorded_scale = float(config["asinh_scale"])
        if recorded_scale <= 0:
            raise ValueError("Checkpoint asinh_scale must be positive")
        if asinh_scale is not None and float(asinh_scale) != recorded_scale:
            raise RuntimeError(
                "Requested asinh_scale does not match the checkpoint contract: "
                f"requested={float(asinh_scale)}, checkpoint={recorded_scale}"
            )
        self.asinh_scale = recorded_scale

        recorded_patch_size = tuple(int(v) for v in config["patch_size"])
        if len(recorded_patch_size) != 3 or any(v <= 0 for v in recorded_patch_size):
            raise ValueError(f"Invalid checkpoint patch_size={recorded_patch_size}")
        self.patch_size = (
            tuple(int(v) for v in patch_size) if patch_size is not None else recorded_patch_size
        )
        if len(self.patch_size) != 3 or any(v <= 0 for v in self.patch_size):
            raise ValueError("patch_size must contain three positive integers")

        configured_stride = stride or tuple(max(1, value // 2) for value in self.patch_size)
        self.stride = tuple(int(v) for v in configured_stride)
        if len(self.stride) != 3 or any(v <= 0 for v in self.stride):
            raise ValueError("stride must contain three positive integers")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = torch.device(device)
        self.precision: PrecisionPolicy = resolve_precision(
            amp=amp,
            amp_dtype=amp_dtype,
            device=self.device,
        )
        self.contract = MNIContract.from_reference(self.mni_reference)

        self.output_mode = str(config["output_mode"])
        if self.output_mode not in OUTPUT_MODES:
            raise RuntimeError(
                f"Checkpoint output_mode={self.output_mode!r} is unsupported; "
                f"expected {OUTPUT_MODES}"
            )
        attention_levels = tuple(int(v) for v in config["attention_levels"])
        if len(set(attention_levels)) != len(attention_levels):
            raise RuntimeError("Checkpoint attention_levels contains duplicate entries")
        self.model = SmartPETGenerator(
            int(config["base_channels"]),
            attention_levels=attention_levels,
            output_mode=self.output_mode,
            similarity_mode=str(config["similarity_mode"]),
            encoder_convs_per_level=int(config["encoder_convs_per_level"]),
            channel_spatial_input_projection=bool(
                config["channel_spatial_input_projection"]
            ),
            generator_spectral_norm=bool(config["generator_spectral_norm"]),
        ).to(self.device)
        self.model.load_state_dict(raw["generator_state"], strict=True)
        self.model.eval()
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)

    def predict(self, input_path: str | Path, *, input_domain: str) -> PredictionResult:
        image, input_volume = load_mni_volume(input_path, self.contract)
        source = prepare_inference_input(
            input_volume,
            domain=input_domain,
            asinh_scale=self.asinh_scale,
        )
        raw = predict_volume(
            self.model,
            source,
            device=self.device,
            patch_size=self.patch_size,
            stride=self.stride,
            amp=self.precision.autocast_enabled,
            amp_dtype=self.precision.autocast_dtype,
        )
        normalized, suv, negative_count = prepare_shared_inference_outputs(
            raw,
            asinh_scale=self.asinh_scale,
        )
        if self.output_mode == "positive_softplus_residual" and negative_count:
            raise RuntimeError(
                "Positive-residual checkpoint produced negative normalized voxels; "
                f"count={negative_count}"
            )
        return PredictionResult(
            reference_image=image,
            normalized=normalized,
            suv=suv,
            negative_count=negative_count,
            prediction_id=prediction_identifier(normalized),
        )

    @staticmethod
    def save(
        result: PredictionResult,
        *,
        normalized_output: str | Path | None,
        suv_output: str | Path | None,
    ) -> dict[str, Path]:
        saved: dict[str, Path] = {}
        if normalized_output is not None:
            saved["normalized"] = save_like(
                result.normalized, result.reference_image, normalized_output
            )
        if suv_output is not None:
            saved["suv"] = save_like(result.suv, result.reference_image, suv_output)
        return saved
