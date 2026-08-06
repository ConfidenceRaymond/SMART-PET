# Offline Narval-to-GitHub release

Narval compute nodes do not need internet access.

```bash
cd SMART-PET
git init -b main
git add .
git commit -m "Release SMART-PET v0.3.0"
git tag -a v0.3.0 -m "SMART-PET v0.3.0"
git bundle create SMART-PET-v0.3.0.bundle main --tags
sha256sum SMART-PET-v0.3.0.bundle > SMART-PET-v0.3.0.bundle.sha256
```

Copy the bundle to an internet-connected computer with `scp`, clone it, add the empty GitHub repository as `origin`, and push `main` plus tags:

```bash
git clone SMART-PET-v0.3.0.bundle SMART-PET
cd SMART-PET
git remote add origin git@github.com:YOUR_USERNAME/SMART-PET.git
git push -u origin main
git push origin --tags
```

Do not commit checkpoints, NIfTI volumes, patient manifests, local environment files, logs, or `work/`.
