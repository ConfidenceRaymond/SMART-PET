# Legacy checkpoint conversion

Normal SMART-PET commands load artifacts with `weights_only=True`. A legacy
format-3 training checkpoint may contain Python and NumPy RNG objects that do
not satisfy that security boundary.

`smartpet-convert-legacy-checkpoint` is a narrowly scoped migration command. It
contains the repository's only intentional `weights_only=False` call and must
not be run on a workstation, login node, or compute job that can access patient
data, credentials, SSH agents, cloud tokens, or writable project storage.

Run it only inside a disposable, network-disabled container with no sensitive
host mounts. Record and verify the source digest before execution:

```bash
smartpet-convert-legacy-checkpoint \
  --input /isolated/input/legacy_v0.3.0.pt \
  --output /isolated/output/converted_v4.pt \
  --expected-sha256 <SOURCE_SHA256> \
  --confirmation I_UNDERSTAND_UNSAFE_PICKLE
```

The command:

- verifies the supplied source SHA-256 before unsafe loading;
- accepts only SMART-PET full checkpoints with `format_version=3`;
- converts Python and NumPy RNG state to primitive/tensor-only structures;
- writes a new full checkpoint with `format_version=4`;
- records the source digest and conversion metadata.

After conversion, move only the converted file out of the isolated environment
and run:

```bash
smartpet-audit-checkpoint --checkpoint /path/to/converted_v4.pt
```

Never disable `weights_only=True` in training, inference, evaluation, or audit
code to make an unfamiliar artifact load.
