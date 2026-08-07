# Corrected configurable S4 architecture

SMART-PET v0.3.1 separates the frozen v0.3.0 architecture from the corrected
S4 candidate. Architecture choices are explicit configuration fields, recorded
in full checkpoints and inference-only weights, and checked during resume,
fine-tuning, export, audit, and inference.

## Frozen v0.3.0 profile

`configs/train_from_scratch.json` preserves the published v0.3.0 release
contract:

```json
{
  "attention_levels": [2, 3],
  "similarity_mode": "v030_luminance",
  "encoder_convs_per_level": 1,
  "channel_spatial_input_projection": false,
  "generator_spectral_norm": false,
  "discriminator_spectral_norm": false
}
```

These defaults preserve strict loading of the existing epoch-4 generator.
Legacy format-4 checkpoints and format-1 inference weights that predate these
fields resolve only to this complete immutable profile. Partially specified
architecture metadata is rejected.

## Corrected S4 candidate

`configs/train_corrected_s4.json` defines the primary Phase 2B candidate:

```json
{
  "attention_levels": [2, 3],
  "similarity_mode": "scale_consistent",
  "encoder_convs_per_level": 2,
  "channel_spatial_input_projection": true,
  "generator_spectral_norm": false,
  "discriminator_spectral_norm": true
}
```

The corrected candidate retains the modern MNI/SUV/asinh data contract,
positive softplus residual output, axial scaled dot-product self-attention,
single branch gating, raw discriminator logits, and the corrected LSGAN plus
L1 objective.

## Similarity modes

`similarity_mode` is one of:

- `v030_luminance`: the exact v0.3.0 local luminance-like gate and state-dict
  layout. Under strict deterministic execution, its parameter-free `AvgPool3d`
  smoothing is evaluated by a mathematically equivalent grouped box convolution
  because CUDA does not provide deterministic `avg_pool3d` backward;
- `paper_exact`: the variance-squared Equation 4 as written in the article,
  with `c2 = (0.03 L)^2` derived from each sample/channel dynamic range;
- `scale_consistent`: a dimensionally consistent SSIM contrast form comparing
  local variance with its spatial reference variance, using the same dynamic
  range definition for `c2`.

The corrected modes implement Equation 5 with distinct learned paths:

```text
sigmoid(conv_similarity(similarity_map) + conv_feature(feature_map)) * feature_map
```

The second convolution receives the feature map, not a second copy of the
similarity map.

## Other architectural controls

`encoder_convs_per_level` accepts `1` or `2`. The two-convolution setting
restores the historical encoder depth while retaining the modern skip topology
and output contract.

`channel_spatial_input_projection=true` restores the learned `1x1x1` projection
of the input feature into the spatial-attention logits.

`discriminator_spectral_norm=true` applies spectral normalization to every
active discriminator convolution. Generator spectral normalization remains an
explicit experimental switch and is disabled in the primary corrected profile
until a controlled stability smoke test supports enabling it.

Under DDP, discriminator buffer broadcast is enabled when spectral normalization
is active so the power-iteration buffers remain rank-synchronized across
checkpoint boundaries.

## Acceptance boundary

Phase 2B changes architecture code and metadata contracts only. It does not
claim improved image quality. The candidate must first pass CPU unit tests,
release validation, static conformance reports, and then the predefined short
E0-E4 experimental ladder before any full Option C training run.
