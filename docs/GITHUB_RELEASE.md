# Offline HPC-to-GitHub release

Compute nodes do not need internet access. Perform source validation on the HPC system, create a Git bundle, then transfer the bundle to an internet-connected machine for the GitHub push.

Set the intended release version explicitly:

```bash
VERSION=vX.Y.Z
```

Before bundling:

```bash
bash scripts/validate_release.sh
git diff --check
git status --short
```

The source tree should be clean and the version must agree across `pyproject.toml`, `src/smartpet/__init__.py`, and `CITATION.cff`.

Create and checksum the bundle:

```bash
git tag -a "$VERSION" -m "SMART-PET $VERSION"
git bundle create "SMART-PET-${VERSION}.bundle" main --tags
sha256sum "SMART-PET-${VERSION}.bundle" > "SMART-PET-${VERSION}.bundle.sha256"
```

Copy the bundle to an internet-connected computer, clone it, attach GitHub, and push:

```bash
git clone "SMART-PET-${VERSION}.bundle" SMART-PET
cd SMART-PET
git remote add origin git@github.com:ConfidenceRaymond/SMART-PET.git
git push -u origin main
git push origin --tags
```

Do not commit checkpoints, PET/NIfTI data, patient manifests, environment directories, logs, temporary workspaces, or machine-specific absolute paths.

Public model/template assets are released separately. Before release, update the public asset mirror title and checksum file to agree with the pinned repository asset manifest.
