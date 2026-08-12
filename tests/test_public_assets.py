from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from smartpet.cli import download_assets
from smartpet.public_assets import PUBLIC_ASSETS, assets_for_profile


def test_asset_profiles_are_nested() -> None:
    inference = {asset.relative_path for asset in assets_for_profile("inference")}
    finetune = {asset.relative_path for asset in assets_for_profile("finetune")}
    all_assets = {asset.relative_path for asset in assets_for_profile("all")}
    assert inference < finetune < all_assets
    assert "weights/smartpet_g001_parent_v0.3.1.pt" in inference
    assert "checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt" in finetune


def test_entry_for_path_matches_nested_folder_suffix() -> None:
    entries = [
        {
            "path": "SMART-PET_REPRODUCIBILITY_ASSETS_v0.3.0/templates/csymT.nii.gz",
            "url": "https://example.invalid/csymT",
        }
    ]
    entry = download_assets._entry_for_path(entries, "templates/csymT.nii.gz")
    assert entry["url"].endswith("/csymT")


def test_verify_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"smartpet")
    expected = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_assets._verify(path, expected)


def test_repository_checksum_manifest_matches_pinned_assets() -> None:
    manifest = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "PUBLIC_ASSET_SHA256_v0.3.1.txt"
    )
    rows = {}
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        checksum, relative_path = line.split(maxsplit=1)
        rows[relative_path] = checksum

    expected = {asset.relative_path: asset.sha256 for asset in PUBLIC_ASSETS}
    assert rows == expected
