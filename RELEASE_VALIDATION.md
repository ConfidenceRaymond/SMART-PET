# Release validation status

Build date: 2026-08-06

Completed for this corrected source revision:

- Python source and tests compiled successfully;
- all shell launchers passed `bash -n`;
- 44 tests passed in the packaging environment;
- the 61 Ruff diagnostics reported by the first Narval release check were addressed without weakening the configured lint rules;
- the `smartpet-0.3.0-py3-none-any.whl` wheel built successfully with the license file embedded;
- no NIfTI, checkpoint, patient manifest, SLURM log, private subject identifier, or private absolute cluster path was found in the source tree.

Required before a public GitHub tag:

```bash
bash scripts/validate_release.sh
```

Run that command on Narval or another environment with all declared dependencies and Ruff installed. The release gate must additionally complete the single-GPU, DDP, exact-resume, fine-tuning initialization, and inference smoke tests listed in `docs/RELEASE_CHECKLIST.md`.
