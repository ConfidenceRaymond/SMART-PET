# Changes from the published SMART-PET implementation

The 2024 article described subject-space PET volumes manually stripped of non-brain tissue, resampled directly to `128 × 128 × 128`, standardized by mean and standard deviation, and scaled to `[-1, 1]`. Version 0.3.0 is a modernized implementation and is not a bitwise reproduction.

| Component | Published implementation | v0.3.0 |
|---|---|---|
| Image space | Subject space | User-supplied common reference geometry |
| Training samples | Whole image resized to 128³ | Paired foreground-aware 128³ crops |
| Intensity transform | Mean/std and `[-1,1]` | Reversible non-negative asinh-SUV |
| Output | Historical bounded output | Identity-centred positive softplus residual |
| Attention | SSAB concept; multiple ablations reported | Combined SSAB3D at configurable encoder levels; baseline `[2,3]` = 16³ and 8³ |
| Adversarial loss | Best reported configuration used adversarial MSE + L1 | LSGAN MSE + `100 × L1` |
| Multi-GPU | Legacy implementation | One process per GPU with PyTorch DDP |
| Inference | Fixed-size image | Full-volume overlapping Hann reconstruction |
| Resume | Limited historical state | Exact optimizer/scheduler/RNG continuation |
| Fine-tuning | Not separated from continuation | Explicit `--init-checkpoint` mode |
| Evaluation | Historical image and clinical analyses | Fixed-mask SUV evaluation plus provenance |

The supplementary material distinguishes individual SAM, SSA, and CSA variants from combined S4/S5 configurations. The production baseline uses the combined SSAB3D block, not a standalone module called “SAM4.”
