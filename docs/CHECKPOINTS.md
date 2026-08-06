# Checkpoint contract

Full checkpoints use format version 3 and contain:

- generator and discriminator states;
- both Adam optimizer states;
- learning-rate schedulers;
- AMP scaler state when applicable;
- epoch, batch position, global step, and samples seen;
- optimizer-update integrity counters;
- rank-specific Python, NumPy, CPU Torch, and CUDA RNG states;
- training configuration and precision policy;
- current best validation metric.

Use `smartpet-audit-checkpoint` before continuation. Do not store multi-gigabyte checkpoints directly in Git. Publish model weights through a release archive, Zenodo, Hugging Face, or institutional storage and record SHA-256 hashes.
