# Frozen historical SMART-PET source snapshot

This directory is an immutable forensic reference used to determine how the
modern SMART-PET implementation differs from the uploaded historical code.

## Safety and execution policy

The files under `source/` are preserved byte-for-byte. They are **not production
code**, are not imported by the `smartpet` package, and must not be executed.
They contain historical unsafe and non-reproducible patterns, including `eval`,
legacy checkpoint loading assumptions, machine-specific paths, randomized GAN
labels, and target-dependent inverse scaling.

The snapshot is excluded from Ruff and marked as binary-diff-protected because
formatting, line-ending conversion, or whitespace cleanup would invalidate the
forensic evidence. The files remain directly readable from a local checkout.
`scripts/validate_release.sh` verifies every file against `SHA256SUMS.txt`.

Verify manually from the repository root:

```bash
sha256sum -c reference/legacy/SHA256SUMS.txt
smartpet-conformance verify-legacy
```

`manifest.json` records the file roles, hashes, and limitations.
`architecture_contract.json` records source-derived architectural and training
facts that the conformance harness compares against the modern implementation.

The snapshot is not a complete historical environment. In particular, the
uploaded `train16_8.py` currently invokes test mode, and no separate historical
ADVMSE training script was supplied.
