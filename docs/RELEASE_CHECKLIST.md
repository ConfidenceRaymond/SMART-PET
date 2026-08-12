# Public release checklist

Before pushing a new tag:

1. verify that `LICENSE` and the README state CC BY-NC-SA 4.0;
2. verify the version agrees across `pyproject.toml`, `src/smartpet/__init__.py`, and `CITATION.cff`;
3. run `bash scripts/validate_release.sh` in the target environment;
4. run `git diff --check` and confirm no generated/private files are tracked;
5. create a fresh external-user environment with `scripts/setup_environment.sh`;
6. verify the public `inference` asset profile with `smartpet-download-assets --verify-only`;
7. audit the parent inference weights with the pinned SHA-256;
8. run one raw-activity preprocessing smoke using ANTs and inspect registration provenance;
9. verify one `ADMIN` activity example and one `NONE` frame-average decay example;
10. verify XLSX and CSV metadata parsing;
11. verify dynamic 4D input fails with the documented scalar-3D guidance;
12. run a single-GPU two-step training smoke;
13. run a two-GPU DDP two-step smoke;
14. run exact continuation from the DDP smoke checkpoint;
15. run fine-tuning initialization into a new directory from the full parent checkpoint;
16. run one-volume and batch inference;
17. audit the full checkpoint with `smartpet-audit-checkpoint`;
18. export and audit inference-only weights with `smartpet-export-weights` and `smartpet-audit-weights`;
19. audit output NIfTIs and metadata with `smartpet-audit-inference`;
20. run fixed-mask evaluation with the pinned public brain mask;
21. confirm no patient data, manifests, checkpoints, logs, environment directories, or absolute private paths are tracked;
22. ensure the public asset mirror title matches the intended asset release;
23. replace the mirror checksum file with the complete repository-pinned checksum list;
24. add/update the external-activity XLSX template in the public asset mirror;
25. preserve third-party notices for template resources;
26. publish an immutable citable asset archive when available and record its identifier.

Every release candidate must repeat this gate before tagging. A passing older release does not validate later source changes automatically.
