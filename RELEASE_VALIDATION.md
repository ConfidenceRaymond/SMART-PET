# Release validation status

## v0.3.1 published baseline

The v0.3.1 source release was validated before publication with:

- Ruff passing under the configured rules;
- the complete pytest suite passing;
- version-contract checks passing;
- `git diff --check` passing;
- checkpoint and inference-weight audits passing;
- single- and multi-GPU smoke validation;
- whole-volume inference and inference-output auditing.

Published v0.3.1 source commit:

```text
cb92e9774a94af71b1bf31be2e792500e4a6829d
```

## Post-release clean-room validation

A fresh public clone was then exercised as an external user. The clean-room path validated installation, public assets, native activity preprocessing, ANTs shared-transform registration, canonical orientation handling, SUV conversion, asinh normalization, G0.01-parent inference, output auditing, and fixed-mask evaluation.

The clean-room pass also identified documentation/environment/usability issues. The current correction set addresses those issues before the next release. See:

```text
docs/EXTERNAL_USER_VALIDATION.md
```

## Required gate for the next release

The correction set is **not** considered released merely because source edits exist. Before a new tag:

```bash
bash scripts/validate_release.sh
git diff --check
```

Then repeat the single-GPU, DDP, exact-resume, fine-tuning initialization, preprocessing, inference, asset-download/verification, and output-audit checks in `docs/RELEASE_CHECKLIST.md`.
