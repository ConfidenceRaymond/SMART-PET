from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from smartpet.data.nifti import MNIContract, load_mni_volume, save_like
from smartpet.inference.domains import (
    prepare_inference_input,
    prepare_shared_inference_outputs,
)
from smartpet.inference.outputs import prediction_identifier
from smartpet.inference.sliding_window import predict_volume
from smartpet.models import SmartPETGenerator
from smartpet.training.precision import PrecisionPolicy, resolve_precision


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionResult:
    reference_image: nib.Nifti1Image
    normalized: np.ndarray
    suv: np.ndarray
    negative_count: int
    prediction_id: str


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

        raw = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        config: dict[str, Any] = dict(raw.get("config", {}))
        self.config = config
        resolved_scale = (
            asinh_scale if asinh_scale is not None else config.get("asinh_scale", 1.0)
        )
        self.asinh_scale = float(resolved_scale)
        if self.asinh_scale <= 0:
            raise ValueError("asinh_scale must be positive")
        configured_patch_size = patch_size or config.get(
            "patch_size", (128, 128, 128)
        )
        self.patch_size = tuple(int(v) for v in configured_patch_size)
        configured_stride = stride or tuple(
            max(1, value // 2) for value in self.patch_size
        )
        self.stride = tuple(int(v) for v in configured_stride)
        if any(v <= 0 for v in (*self.patch_size, *self.stride)):
            raise ValueError("patch_size and stride must be positive")

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
        self.output_mode = str(config.get("output_mode", "linear"))
        attention_levels = tuple(int(v) for v in config.get("attention_levels", (2, 3)))
        self.model = SmartPETGenerator(
            int(config.get("base_channels", 32)),
            attention_levels=attention_levels,
            output_mode=self.output_mode,
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
