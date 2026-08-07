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
