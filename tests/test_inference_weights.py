from __future__ import annotations

from pathlib import Path

import pytest
import torch

from smartpet.checkpoint_io import sha256_file
from smartpet.inference.weights import (
    INFERENCE_WEIGHTS_ARTIFACT_TYPE,
    INFERENCE_WEIGHTS_FORMAT_VERSION,
    audit_inference_weights,
    export_inference_weights,
)
from smartpet.models import SmartPETGenerator
from smartpet.training.checkpoint import CHECKPOINT_FORMAT_VERSION, atomic_save


class _WriteMarker:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        statement = f"open({str(self.marker)!r}, 'w', encoding='utf-8').write('executed')"
        return exec, (statement,)


def _full_checkpoint(path: Path) -> None:
    generator = SmartPETGenerator(
        base_channels=1,
        attention_levels=(),
        output_mode="positive_softplus_residual",
    )
    atomic_save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "artifact_type": "smartpet_training_checkpoint",
            "smartpet_version": "0.3.0",
            "generator_state": generator.state_dict(),
            "config": {
                "base_channels": 1,
                "attention_levels": [],
                "output_mode": "positive_softplus_residual",
                "asinh_scale": 1.0,
                "patch_size": [128, 128, 128],
            },
            "global_step": 11,
            "epoch": 4,
            "best_metric": 0.25,
        },
        path,
    )


def test_export_and_audit_inference_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "full.pt"
    weights = tmp_path / "weights.pt"
    _full_checkpoint(checkpoint)

    payload = export_inference_weights(checkpoint, weights)
    result = audit_inference_weights(weights, expected_sha256=sha256_file(weights))

    assert payload["artifact_type"] == INFERENCE_WEIGHTS_ARTIFACT_TYPE
    assert payload["format_version"] == INFERENCE_WEIGHTS_FORMAT_VERSION
    assert result["source_checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["source_global_step"] == 11
    assert result["source_epoch"] == 4
    assert result["config"]["patch_size"] == [128, 128, 128]
    assert result["config"]["similarity_mode"] == "v030_luminance"
    assert result["config"]["encoder_convs_per_level"] == 1
    assert result["config"]["channel_spatial_input_projection"] is False


def test_audit_weights_rejects_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "full.pt"
    weights = tmp_path / "weights.pt"
    _full_checkpoint(checkpoint)
    export_inference_weights(checkpoint, weights)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        audit_inference_weights(weights, expected_sha256="0" * 64)


def test_audit_weights_rejects_missing_config_field(tmp_path: Path) -> None:
    generator = SmartPETGenerator(base_channels=1, attention_levels=())
    path = tmp_path / "invalid.pt"
    atomic_save(
        {
            "format_version": 1,
            "artifact_type": INFERENCE_WEIGHTS_ARTIFACT_TYPE,
            "smartpet_version": "0.3.0",
            "generator_state": generator.state_dict(),
            "config": {
                "base_channels": 1,
                "attention_levels": [],
                "output_mode": "positive_softplus_residual",
                "asinh_scale": 1.0,
            },
            "source_checkpoint_sha256": "1" * 64,
            "source_global_step": 1,
            "source_epoch": 0,
            "source_best_metric": 0.5,
        },
        path,
    )
    with pytest.raises(RuntimeError, match="patch_size"):
        audit_inference_weights(path)


def test_audit_weights_refuses_pickle_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    path = tmp_path / "hostile.pt"
    torch.save({"payload": _WriteMarker(marker)}, path)

    with pytest.raises(RuntimeError, match="weights_only=True"):
        audit_inference_weights(path)

    assert not marker.exists(), "weights payload executed during audit"


def test_format1_weights_resolve_the_frozen_v030_architecture(tmp_path: Path) -> None:
    generator = SmartPETGenerator(base_channels=1, attention_levels=())
    path = tmp_path / "legacy_v1_weights.pt"
    atomic_save(
        {
            "format_version": 1,
            "artifact_type": INFERENCE_WEIGHTS_ARTIFACT_TYPE,
            "smartpet_version": "0.3.0",
            "generator_state": generator.state_dict(),
            "config": {
                "base_channels": 1,
                "attention_levels": [],
                "output_mode": "linear",
                "asinh_scale": 1.0,
                "patch_size": [128, 128, 128],
            },
            "source_checkpoint_sha256": "1" * 64,
            "source_checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "source_smartpet_version": "legacy-v0.3.0",
            "source_global_step": 1,
            "source_epoch": 0,
            "source_best_metric": 0.5,
        },
        path,
    )
    result = audit_inference_weights(path)
    assert result["format_version"] == 1
    assert result["config"]["similarity_mode"] == "v030_luminance"
    assert result["config"]["generator_spectral_norm"] is False


def test_format2_weights_reject_partial_architecture_metadata(tmp_path: Path) -> None:
    generator = SmartPETGenerator(base_channels=1, attention_levels=())
    path = tmp_path / "partial_v2_weights.pt"
    atomic_save(
        {
            "format_version": INFERENCE_WEIGHTS_FORMAT_VERSION,
            "artifact_type": INFERENCE_WEIGHTS_ARTIFACT_TYPE,
            "smartpet_version": "0.3.1-dev",
            "generator_state": generator.state_dict(),
            "config": {
                "base_channels": 1,
                "attention_levels": [],
                "output_mode": "linear",
                "asinh_scale": 1.0,
                "patch_size": [128, 128, 128],
                "similarity_mode": "v030_luminance",
            },
            "source_checkpoint_sha256": "1" * 64,
            "source_checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "source_smartpet_version": "0.3.1-dev",
            "source_global_step": 1,
            "source_epoch": 0,
            "source_best_metric": 0.5,
        },
        path,
    )
    with pytest.raises(RuntimeError, match="partially specified"):
        audit_inference_weights(path)
