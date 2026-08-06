# Public release checklist

Before pushing a tag:

1. verify that `LICENSE` and the README state CC BY-NC-SA 4.0;
2. run `bash scripts/validate_release.sh` in the target environment;
3. run a single-GPU two-step smoke;
4. run a two-GPU DDP two-step smoke;
5. run exact continuation from the DDP smoke checkpoint;
6. run fine-tuning initialization into a new directory;
7. run one-volume and batch inference;
8. audit both output NIfTIs and metadata;
9. confirm no patient data, manifests, checkpoints, logs, or absolute private paths are tracked;
10. record source archive, checkpoint, and model-card SHA-256 hashes.

The local build used to prepare this source tree could not execute NiBabel-dependent tests because NiBabel was unavailable. Those tests must pass on Narval or in GitHub Actions before the repository is tagged as validated.
