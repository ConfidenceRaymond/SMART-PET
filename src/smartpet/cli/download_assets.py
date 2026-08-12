from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from smartpet.public_assets import PUBLIC_ASSET_FOLDER_URL, assets_for_profile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def _gdown_executable() -> str:
    executable = shutil.which("gdown")
    if executable is None:
        raise RuntimeError(
            "gdown is required for automated public-asset downloads. Install the "
            "optional asset dependency with: python -m pip install '.[assets]'"
        )
    return executable


def _remote_entries(gdown: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [gdown, PUBLIC_ASSET_FOLDER_URL, "--folder", "--json", "--quiet"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Could not parse gdown folder listing as JSON") from exc
    if not isinstance(entries, list):
        raise RuntimeError("Unexpected gdown folder listing format")
    return [entry for entry in entries if isinstance(entry, dict)]


def _entry_for_path(entries: list[dict[str, Any]], relative_path: str) -> dict[str, Any]:
    suffix = relative_path.replace("\\", "/")
    matches = []
    for entry in entries:
        path = str(entry.get("path", "")).replace("\\", "/")
        if path == suffix or path.endswith("/" + suffix):
            matches.append(entry)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one public Drive entry for {relative_path}, found {len(matches)}"
        )
    if not matches[0].get("url"):
        raise RuntimeError(f"Drive entry has no URL for {relative_path}")
    return matches[0]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download and SHA-256 verify the pinned SMART-PET public reproducibility assets."
        )
    )
    p.add_argument(
        "--output-dir",
        default="resources",
        help="Destination root. Default: ./resources",
    )
    p.add_argument(
        "--profile",
        choices=["inference", "finetune", "all"],
        default="inference",
        help=(
            "inference downloads the parent model + templates; finetune adds the full "
            "parent checkpoint; all adds domain-specific and historical weights."
        ),
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify expected local files without network access.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list", action="store_true", help="List the pinned files and exit.")
    return p


def main() -> None:
    args = parser().parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    assets = assets_for_profile(args.profile)

    if args.list:
        for asset in assets:
            print(f"{asset.sha256}  {asset.relative_path}  # {asset.description}")
        return

    if args.verify_only:
        for asset in assets:
            path = output_root / asset.relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            _verify(path, asset.sha256)
            print(f"[OK] {asset.relative_path}")
        print(f"[OK] verified {len(assets)} asset(s) under {output_root}")
        return

    gdown = _gdown_executable()
    entries = _remote_entries(gdown)
    output_root.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        destination = output_root / asset.relative_path
        if destination.is_file() and not args.overwrite:
            _verify(destination, asset.sha256)
            print(f"[OK] existing {asset.relative_path}")
            continue

        entry = _entry_for_path(entries, asset.relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        subprocess.run(
            [gdown, str(entry["url"]), "-O", str(temporary)],
            check=True,
        )
        try:
            _verify(temporary, asset.sha256)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"[OK] downloaded {asset.relative_path}")

    print(f"[OK] downloaded and verified {len(assets)} asset(s) under {output_root}")


if __name__ == "__main__":
    main()
