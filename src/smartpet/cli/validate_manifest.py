from __future__ import annotations

import argparse

import numpy as np

from smartpet.data.dataset import read_manifest
from smartpet.data.nifti import MNIContract, load_mni_volume


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate SMART-PET manifest geometry and intensity domain."
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--mni-reference", required=True)
    p.add_argument("--other-manifest", help="Optional second split; subject overlap is rejected")
    p.add_argument("--require-nonnegative", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--negative-tolerance", type=float, default=1e-6)
    args = p.parse_args()
    if args.negative_tolerance < 0:
        p.error("--negative-tolerance must be non-negative")

    records = read_manifest(args.manifest)
    if args.other_manifest:
        other = read_manifest(args.other_manifest)
        overlap = sorted(
            {record.subject_id for record in records}
            & {record.subject_id for record in other}
        )
        if overlap:
            raise ValueError(f"Subject overlap between manifests: {overlap[:10]}")

    contract = MNIContract.from_reference(args.mni_reference)
    global_min = float("inf")
    global_max = float("-inf")
    tiny_negative_voxels = 0
    for record in records:
        if record.source_path.resolve() == record.target_path.resolve():
            raise ValueError(f"Source and target are the same file for {record.subject_id}")
        for role, path in (("source", record.source_path), ("target", record.target_path)):
            _, data = load_mni_volume(path, contract)
            minimum = float(data.min())
            maximum = float(data.max())
            global_min = min(global_min, minimum)
            global_max = max(global_max, maximum)
            if args.require_nonnegative and minimum < -args.negative_tolerance:
                count = int(np.count_nonzero(data < -args.negative_tolerance))
                raise ValueError(
                    "Negative PET values exceed tolerance: "
                    f"subject={record.subject_id}, role={role}, path={path}, "
                    f"minimum={minimum:.8g}, count={count}"
                )
            tiny_negative_voxels += int(
                np.count_nonzero(
                    (data < 0) & (data >= -args.negative_tolerance)
                )
            )

    print(f"[OK] {len(records)} paired subjects validated")
    print(f"[OK] intensity minimum={global_min:.8g}")
    print(f"[OK] intensity maximum={global_max:.8g}")
    if args.other_manifest:
        print("[OK] subject overlap=0")
    if args.require_nonnegative:
        print(f"[OK] non-negative domain; tiny_negative_voxels={tiny_negative_voxels}")


if __name__ == "__main__":
    main()
