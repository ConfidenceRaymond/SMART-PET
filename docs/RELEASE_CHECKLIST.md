# Public release checklist

Before pushing a tag:

1. verify that `LICENSE` and the README state CC BY-NC-SA 4.0;
2. run `bash scripts/validate_release.sh` in the target environment;
3. run a single-GPU two-step smoke;
4. run a two-GPU DDP two-step smoke;
5. run exact continuation from the DDP smoke checkpoint;
6. run fine-tuning initialization into a new directory;
7. run one-volume and batch inference;
8. audit the full checkpoint with `smartpet-audit-checkpoint`;
9. export and audit inference-only weights with `smartpet-export-weights` and `smartpet-audit-weights`;
10. audit both output NIfTIs and metadata;
11. confirm no patient data, manifests, checkpoints, logs, or absolute private paths are tracked;
12. publish the asset archive at an immutable citable location;
13. commit the source, checkpoint, inference-weight, template, mask, and model-card SHA-256 values to Git.

NiBabel-dependent tests have passed on Narval and in the release validation workflow. Every later release candidate must repeat the complete validation before tagging.
