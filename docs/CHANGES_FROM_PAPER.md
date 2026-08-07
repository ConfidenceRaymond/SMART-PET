# Changes from the published SMART-PET implementation

The 2024 article described subject-space PET volumes manually stripped of non-brain tissue, resampled directly to `128 × 128 × 128`, standardized by mean and standard deviation, and scaled to `[-1, 1]`. Version 0.3.0 is a modernized implementation and is not a bitwise reproduction.

| Component | Published implementation | v0.3.0 | v0.3.1 configurable contract |
|---|---|---|---|
| Image space | Subject space | User-supplied common reference geometry | Preserved |
| Training samples | Whole image resized to 128³ | Paired foreground-aware 128³ crops | Preserved |
| Intensity transform | Mean/std and `[-1,1]` | Reversible non-negative asinh-SUV | Preserved |
| Output | Historical bounded output | Identity-centred positive softplus residual | Preserved |
| Attention | SSAB concept; multiple ablations reported | Combined SSAB3D at configurable encoder levels; baseline `[2,3]` = 16³ and 8³ | Explicit `v030_luminance`, `paper_exact`, and `scale_consistent` similarity modes; corrected Equation 5 wiring |
| Adversarial loss | Best reported configuration used adversarial MSE + L1 | LSGAN MSE + `100 × L1` | Preserved |
| Multi-GPU | Legacy implementation | One process per GPU with PyTorch DDP | Preserved |
| Inference | Fixed-size image | Full-volume overlapping Hann reconstruction | Preserved |
| Resume | Limited historical state | Exact optimizer/scheduler/RNG continuation | Extended architecture compatibility checks |
| Fine-tuning | Not separated from continuation | Explicit `--init-checkpoint` mode | Extended architecture compatibility checks |
| Evaluation | Historical image and clinical analyses | Fixed-mask SUV evaluation plus provenance | Preserved |

The supplementary material distinguishes individual SAM, SSA, and CSA variants from combined S4/S5 configurations. The production baseline uses the combined SSAB3D block, not a standalone module called “SAM4.”

## v0.3.1 architectural corrections

The corrected S4 candidate restores two convolutions per encoder level, the
channel-spatial `conv1x1(x)` branch, and discriminator spectral normalization.
It keeps the modern axial self-attention operator and removes the historical
second multiplication by the feature tensor. Generator spectral normalization
is configurable but disabled in the primary candidate pending a controlled
stability gate. See [Corrected configurable S4](CORRECTED_S4.md).

Patch-based training changes the receptive-field-to-anatomy ratio relative to
the article's whole-volume 128³ resampling. Therefore, descriptions of
self-attention as global apply within each training patch, not necessarily the
complete native volume. Registration into the common reference also interpolates
PET values and can alter the noise texture presented to the model.
