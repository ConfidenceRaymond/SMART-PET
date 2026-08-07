from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from smartpet.checkpoint_io import safe_torch_load, sha256_file
from smartpet.cli.convert_legacy_checkpoint import convert_legacy_checkpoint
from smartpet.training.checkpoint import CHECKPOINT_FORMAT_VERSION


class _WriteMarker:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        statement = f"open({str(self.marker)!r}, 'w', encoding='utf-8').write('executed')"
        return exec, (statement,)


def _legacy_payload() -> dict[str, object]:
    return {
        "format_version": 3,
        "generator_state": {"weight": torch.ones(1)},
        "discriminator_state": {"weight": torch.ones(1)},
        "g_optimizer_state": {"state": {}, "param_groups": []},
        "d_optimizer_state": {"state": {}, "param_groups": []},
        "global_step": 0,
        "g_optimizer_updates": 0,
        "d_optimizer_updates": 0,
        "rng_states": [
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.random.get_rng_state(),
            }
        ],
        "world_size": 1,
        "config": {"output_mode": "positive_softplus_residual"},
        "best_metric": 0.5,
    }


def test_converter_refuses_before_unsafe_load_without_confirmation(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    source = tmp_path / "hostile.pt"
    torch.save({"payload": _WriteMarker(marker)}, source)

    with pytest.raises(RuntimeError, match="confirmation token"):
        convert_legacy_checkpoint(
            source,
            tmp_path / "converted.pt",
            expected_sha256=sha256_file(source),
            confirmation="NO",
        )

    assert not marker.exists(), "unsafe legacy payload executed before confirmation"


def test_converter_refuses_before_unsafe_load_on_hash_mismatch(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    source = tmp_path / "hostile.pt"
    torch.save({"payload": _WriteMarker(marker)}, source)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        convert_legacy_checkpoint(
            source,
            tmp_path / "converted.pt",
            expected_sha256="0" * 64,
            confirmation="I_UNDERSTAND_UNSAFE_PICKLE",
        )

    assert not marker.exists(), "unsafe legacy payload executed before digest verification"


def test_converter_rewrites_v3_rng_state_to_safe_v4(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pt"
    output = tmp_path / "converted.pt"
    torch.save(_legacy_payload(), source)

    result = convert_legacy_checkpoint(
        source,
        output,
        expected_sha256=sha256_file(source),
        confirmation="I_UNDERSTAND_UNSAFE_PICKLE",
    )
    converted = safe_torch_load(output)

    assert result["output_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert converted["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert converted["artifact_type"] == "smartpet_training_checkpoint"
    assert isinstance(converted["rng_states"][0]["python"], dict)
    assert isinstance(converted["rng_states"][0]["numpy"], dict)
    assert converted["conversion"]["source_sha256"] == sha256_file(source)
