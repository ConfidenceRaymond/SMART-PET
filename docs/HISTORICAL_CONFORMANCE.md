# Historical SMART-PET conformance contract

This document separates three things that must not be conflated:

1. what the uploaded historical source actually computes;
2. what SMART-PET v0.3.0 modernized intentionally; and
3. what v0.3.1 will restore, correct, or test experimentally.

The raw snapshot is frozen under `reference/legacy/source/` and protected by
`reference/legacy/SHA256SUMS.txt`. It is reference-only and must not be run.

## Source-derived comparison

| Component | Uploaded historical code | v0.3.0 | v0.3.1 decision | Classification |
|---|---|---|---|---|
| Input domain | TorchIO Z-normalization and independent `[-1,1]` rescaling | MNI SUV with reversible asinh transform | Preserve v0.3.0 | Intentional modernization |
| Output | Transposed convolution plus Tanh | Positive softplus residual | Preserve v0.3.0 | Intentional modernization |
| Encoder depth | Two convolutions per level | One convolution per level | Make configurable; primary candidate uses two | Architectural restoration |
| Generator spectral normalization | Encoder and decoder convolutions | Absent | Controlled scope after stability gate | Controlled restoration |
| Discriminator spectral normalization | Active discriminator blocks | Absent | Restore | Stabilizer restoration |
| Discriminator output | Sigmoid probability | Raw logits | Preserve raw logits | Historical double-sigmoid defect correction |
| S4 composition | Self + similarity + channel-spatial, followed by convolution | Same three-branch composition | Preserve | Concept preserved |
| Similarity statistic | Variance-squared Equation 4 | Luminance-like local comparison | Implement explicit modes in Phase 2B | Required conformance correction |
| Similarity Equation 5 | Both convolutions receive similarity map | Simplified descriptor fusion | Similarity convolution plus feature convolution | Historical wiring correction |
| Self-attention | Shared query/key module and custom plane/depth products | Axial scaled dot-product attention | Preserve v0.3.0 operator | Intentional modernization |
| Channel-spatial attention | Includes `conv1x1(x)` spatial branch | Branch omitted | Restore branch | Required architectural correction |
| Feature gating | Branches gate internally and S4 multiplies by `x` again | Single internal gating | Preserve single gating | Historical `x²`-type defect correction |
| Attention placement | 16³ and 8³ active; 4³ declared but commented | 16³ and 8³ | Compare `[2,3]` against `[4]` | Controlled experiment |
| Generator objective | Effective gradient is `0.01 × L1` | `100 × L1 + 1 × LSGAN` | Preserve corrected objective | Historical implementation defect correction |
| VGG term | Slice feature L1 detached with `no_grad`, `.item()`, and NumPy mean | Evaluation-only legacy metric | Evaluation/ablation only | Renamed and separated from FID |
| FID | The class named `FID_vgg19` is paired feature L1, not FID | Separate evaluation contract | True distribution-level FID/KID | Metric correction |
| Inverse scaling | Reads standard-dose target maximum | Deterministic inverse asinh | Preserve v0.3.0 | Deployment-critical modernization |

## Why the historical generator was effectively L1-only

The active generator step computes pixel L1, a VGG-derived scalar, and a GAN
term. However, the GAN term evaluates the real low-dose/standard-dose pair, so it
has no path to generator parameters. The VGG implementation runs under
`torch.no_grad()`, calls `.item()`, and averages through NumPy. Only the weighted
pixel term remains connected to the generator graph.

Relevant source:

- [`main16_8.py`, active generator step](../reference/legacy/source/main16_8.py#L146-L163)
- [`metrics.py`, detached VGG feature term](../reference/legacy/source/metrics.py#L262-L315)
- [`param.py`, active weights and optimizer](../reference/legacy/source/param.py#L28-L44)

`smartpet-conformance gradients` turns this graph-connectivity claim into a
machine-checkable report. It uses tiny deterministic modules so the audit does
not require VGG weights or a GPU.

## Attention discrepancies recorded before implementation

The historical self-similarity helper accepts a sigma argument but constructs
its Gaussian window with sigma 1.5. Its published variance-squared formula is
scale-fragile. v0.3.0 instead computes a luminance-like statistic. These are
separate quantities and must remain separately named.

The historical Equation 5 implementation also passes the similarity map to both
convolutions, rather than passing the feature tensor to the second convolution.
The S4 wrapper then multiplies already gated branch outputs by the feature tensor
again.

Relevant source:

- [`SSIMattn.py`, window and statistic](../reference/legacy/source/SSIMattn.py#L15-L31)
- [`SSIMattn.py`, variance-squared self-similarity](../reference/legacy/source/SSIMattn.py#L94-L141)
- [`Attns.py`, similarity wiring](../reference/legacy/source/Attns.py#L180-L205)
- [`Attns.py`, S4 composition and outer multiplication](../reference/legacy/source/Attns.py#L208-L258)

`smartpet-conformance attention` evaluates the historical-code statistic, the
published equation with explicit sigma 3, and the v0.3.0 luminance statistic on
the same deterministic tensor and intensity scales.

## Static architecture audit

`smartpet-conformance architecture` reports parameter counts, module inventory,
encoder convolution counts, attention levels, output mode, discriminator output
contract, and spectral-normalization paths. The output includes the frozen
legacy architecture contract for direct comparison.

Example:

```bash
smartpet-conformance verify-legacy \
  --output work/audits/phase2a/legacy_verification.json

smartpet-conformance architecture \
  --output work/audits/phase2a/architecture_v030.json

smartpet-conformance attention \
  --seed 2023 \
  --output work/audits/phase2a/attention_numerics.json

smartpet-conformance gradients \
  --seed 2023 \
  --output work/audits/phase2a/gradient_attribution.json
```

Phase 2A is observational: it does not modify the model architecture or training
objective. Phase 2B will implement the corrected configurable S4 architecture
behind explicit configuration fields and acceptance tests.
