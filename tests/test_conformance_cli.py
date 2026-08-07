from __future__ import annotations

import json
import sys
from pathlib import Path

from smartpet.cli import conformance

ROOT = Path(__file__).resolve().parents[1]


def _run(monkeypatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["smartpet-conformance", *arguments])
    conformance.main()


def test_verify_legacy_cli(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "verify.json"
    _run(
        monkeypatch,
        ["verify-legacy", "--root", str(ROOT / "reference" / "legacy"), "--output", str(output)],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verified_file_count"] == 17


def test_gradient_cli(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "gradients.json"
    _run(monkeypatch, ["gradients", "--seed", "2023", "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["historical_executed"]["gan_real_pair_gradient_norm"] == 0


def test_corrected_architecture_cli(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "corrected_architecture.json"
    _run(
        monkeypatch,
        [
            "architecture",
            "--base-channels",
            "1",
            "--attention-levels",
            "2",
            "3",
            "--similarity-mode",
            "scale_consistent",
            "--encoder-convs-per-level",
            "2",
            "--channel-spatial-input-projection",
            "--discriminator-spectral-norm",
            "--legacy-contract",
            str(ROOT / "reference" / "legacy" / "architecture_contract.json"),
            "--output",
            str(output),
        ],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generator"]["encoder_conv3d_counts"] == [2] * 7
    assert payload["generator"]["similarity_mode"] == "scale_consistent"
    assert payload["generator"]["channel_spatial_input_projection"] is True
    assert len(payload["discriminator"]["spectral_norm_paths"]) == 5
