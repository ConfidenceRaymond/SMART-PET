# SMART-PET model provenance

This document distinguishes the original G0.01 model from Lawson-specific
fine-tuned checkpoints. These models must not be treated as interchangeable.

## Model hierarchy

| Model | Role | Fine-tuning | Checkpoint SHA-256 | Confirmatory status |
|---|---|---|---|---|
| **G0.01-parent** | Original/internal reference model | None | `2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee` | Original reference |
| **G0.01-Lawson-15ep-1e-5** | Independently confirmed Lawson-specific model | 15 epochs, initial LR `1e-5`, 52 train / 8 validation | `17026f7bda6c0023f886a67a88775f6dee8007526f6ab67e7f5f853fc9b30499` | **Confirmed on locked Lawson-15** |
| **G0.01-Lawson-3e-5** | Newer Lawson development candidate | 100 epochs, constant LR `3e-5`, 52 train / 8 validation | `504e3b4f30f7b25535a22b8d2844a43dcc8895d7201804627404376733e0be22` | **Development only; not independently tested** |

## Independently confirmed Lawson model

The one-shot independent Lawson evaluation used
**G0.01-Lawson-15ep-1e-5**, not G0.01-Lawson-3e-5.

Primary endpoint:

- cohort: 15 previously locked independent Lawson subjects;
- metric: brain-masked, unclipped physical-SUV NMAE;
- comparison: Lawson fine-tuned minus G0.01-parent;
- G0.01-parent mean NMAE: `13.3407%`;
- G0.01-Lawson-15ep-1e-5 mean NMAE: `10.8344%`;
- mean paired difference: `-2.5063` percentage points;
- bootstrap 95% CI: `[-2.9734, -2.0770]` percentage points;
- subject wins: `15/15`;
- exact sign-flip two-sided p-value: `6.1035e-05`.

The preregistered external-superiority criterion passed.

The same checkpoint did not satisfy the predefined internal-retention
noninferiority criterion:

- G0.01-parent internal-73 mean brain NMAE: `4.4344%`;
- Lawson fine-tuned mean: `5.0792%`;
- paired difference: `+0.6448` percentage points;
- bootstrap 95% CI: `[+0.6072, +0.6842]`;
- predefined retention margin: `+0.5` percentage points.

The resulting interpretation is therefore:

> Use cohort-specific checkpoints: the Lawson-adapted checkpoint for Lawson
> data and the original G0.01-parent checkpoint for the original/internal
> domain.

## G0.01-Lawson-3e-5 development candidate

G0.01-Lawson-3e-5 was initialized directly from G0.01-parent and trained for
100 complete epochs on the 52-subject Lawson training set with constant
learning rate `3e-5`.

It was developed without accessing either the locked Lawson-15 or the
internal-73 cohort.

On the eight-subject Lawson development-validation cohort, whole-volume
physical-SUV evaluation showed:

- G0.01-parent brain NMAE: `12.7624%`;
- G0.01-Lawson-3e-5 brain NMAE: `10.2800%`;
- paired improvement: `-2.4824` percentage points.

This supports G0.01-Lawson-3e-5 as a promising Lawson development candidate,
but it does **not** constitute independent confirmation.

## Locked-test governance

The Lawson locked-15 was opened once in the preregistered final evaluation
(job `591999`).

It is now consumed as an independent test set and must not be reused for:

- learning-rate selection;
- epoch selection;
- checkpoint selection;
- loss-weight selection;
- L2-SP tuning;
- architecture selection;
- comparison of newer development candidates.

In particular:

> **G0.01-Lawson-3e-5 must not be described as independently validated on the
> Lawson locked-15.**

Any future Lawson-model development requires a new independent confirmation
cohort if confirmatory claims are desired.

## Naming contract

Use these names consistently:

- `G0.01-parent`
- `G0.01-Lawson-15ep-1e-5`
- `G0.01-Lawson-3e-5`

Do not use the historical alias `A` for G0.01-parent in new reports,
documentation, figures, or released artifacts.
