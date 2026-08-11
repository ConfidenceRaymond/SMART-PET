# Corrected configurable S4 architecture

SMART-PET v0.3.1 makes the S4 architecture choices explicit in configuration,
full checkpoints, and inference-only weights. These fields are validated during
resume, fine-tuning, export, audit, and inference.

## Released G0.01-parent architecture

The current canonical configuration is `configs/train_from_scratch.json`. It
matches the released G0.01-parent model:

```json
{
  "attention_levels": [2, 3],
  "similarity_mode": "scale_consistent",
  "encoder_convs_per_level": 2,
  "channel_spatial_input_projection": true,
  "generator_spectral_norm": false,
  "discriminator_spectral_norm": true,
  "lambda_gan": 0.01
}
```

The public fine-tuning configuration uses the same architecture contract and
initializes both trained networks from the full G0.01-parent checkpoint.

## Historical v0.3.0 compatibility profile

The published v0.3.0 model used:

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

This profile remains an immutable compatibility contract for loading historical
v0.3.0 checkpoints and inference weights. It is not the default v0.3.1
from-scratch configuration.

Legacy format-4 checkpoints and format-1 inference weights that predate the
explicit S4 fields resolve only to this complete historical profile. Partially
specified architecture metadata is rejected.

## Historical Phase 2B candidate

`configs/phase2b_corrected_s4_candidate.json` is retained for development
provenance. It introduced the corrected S4 architecture before the final
G0.01-parent training recipe was established.

It is **not** the recommended public training configuration. In particular, its
optimization settings differ from the released G0.01-parent recipe, including
a larger adversarial weight and a shorter training schedule.

For new training use:

```bash
smartpet-train --config configs/train_from_scratch.json
```

For fine-tuning use:

```bash
smartpet-train \
  --config configs/finetune.json \
  --init-checkpoint /path/to/smartpet_g001_parent_v0.3.1_full_checkpoint.pt
```

## Similarity modes

`similarity_mode` is one of:

- `v030_luminance`: the historical v0.3.0 local luminance-like gate and
  state-dict layout. Under strict deterministic execution, its parameter-free
  `AvgPool3d` smoothing is evaluated by a mathematically equivalent grouped box
  convolution because CUDA does not provide deterministic `avg_pool3d`
  backward;
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

`channel_spatial_input_projection=true` enables the learned `1x1x1` projection
of the input feature into the spatial-attention logits.

`discriminator_spectral_norm=true` applies spectral normalization to every
active discriminator convolution. Generator spectral normalization remains an
explicit switch and is disabled in the released G0.01-parent profile.

Under DDP, discriminator buffer broadcast is enabled when spectral
normalization is active so the power-iteration buffers remain synchronized
across ranks and checkpoint boundaries.

## Compatibility boundary

Architecture compatibility is checked explicitly. SMART-PET rejects
fine-tuning or exact-resume attempts when the checkpoint architecture does not
match the requested configuration.

The historical v0.3.0 profile, the historical Phase 2B candidate, and the
released G0.01-parent recipe should therefore be treated as distinct,
versioned contracts rather than interchangeable presets.
