from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "reference" / "legacy"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_snapshot_manifest_and_hashes_match() -> None:
    manifest = json.loads((LEGACY / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["source_count"] == 17
    records = manifest["files"]
    assert len(records) == 17
    for record in records:
        path = ROOT / record["path"]
        assert path.is_file(), record["path"]
        assert path.stat().st_size == record["size_bytes"]
        assert _sha256(path) == record["sha256"]


def test_sha256sums_lists_exactly_the_frozen_sources() -> None:
    entries = {}
    for line in (LEGACY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        entries[path] = digest
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in (LEGACY / "source").iterdir()
        if path.is_file()
    }
    assert set(entries) == expected
    assert all(_sha256(ROOT / path) == digest for path, digest in entries.items())
