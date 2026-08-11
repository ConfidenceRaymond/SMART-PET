# SMART-PET model provenance

SMART-PET distributes a general parent model, a domain-specific adapted model,
and a historical inference artifact. They should not be treated as
interchangeable.

## Public model hierarchy

| Public model | Role | Training status | Source checkpoint SHA-256 |
|---|---|---|---|
| **G0.01-parent** | General pretrained model | Original reference model | `2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee` |
| **G0.01-external-adapted** | Domain-specific adapted model | 15-epoch fine-tune from G0.01-parent, initial LR `1e-5` | `17026f7bda6c0023f886a67a88775f6dee8007526f6ab67e7f5f853fc9b30499` |
| **v0.3.0 epoch-4 historical** | Historical inference reference | Earlier v0.3.0 release artifact | See release SHA-256 manifest |

The public v0.3.1 inference artifacts are:

```text
G0.01-parent
smartpet_g001_parent_v0.3.1.pt
SHA-256: f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0

G0.01-external-adapted
smartpet_g001_external_adapted_v0.3.1.pt
SHA-256: aecd3b0c15f0b0b90fc6e2142412562ceacc7a5aacd440d37c3476e7dc89b797
```

The parent full checkpoint used for public fine-tuning is:

```text
smartpet_g001_parent_v0.3.1_full_checkpoint.pt
SHA-256: 2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee
```

## External adaptation result

The released external-adapted checkpoint was evaluated once on a previously
locked 15-subject external cohort.

For brain-masked, unclipped physical-SUV NMAE:

- parent mean NMAE: `13.3407%`;
- adapted mean NMAE: `10.8344%`;
- mean paired difference: `-2.5063` percentage points;
- bootstrap 95% CI: `[-2.9734, -2.0770]`;
- adapted model improved all `15/15` subjects;
- exact sign-flip two-sided p-value: `6.1035e-05`.

The external-superiority criterion passed.

The same adapted checkpoint did not satisfy the predefined internal-retention
criterion:

- parent internal mean brain NMAE: `4.4344%`;
- adapted model: `5.0792%`;
- paired difference: `+0.6448` percentage points;
- bootstrap 95% CI: `[+0.6072, +0.6842]`;
- predefined retention margin: `+0.5` percentage points.

The practical interpretation is therefore:

> Use G0.01-parent as the general pretrained model. Use the external-adapted
> checkpoint only when the target domain supports that choice.

## Model-selection governance

A locked test cohort used for a final adaptation claim is considered consumed.
It must not subsequently be reused for hyperparameter selection, checkpoint
selection, loss-weight tuning, or architecture selection.

Future adaptation studies should reserve a new independent test cohort if a new
confirmatory claim is required.

## Naming contract

Use these public names consistently:

- `G0.01-parent`
- `G0.01-external-adapted`
- `v0.3.0 epoch-4 historical`

Do not use historical internal aliases in new public documentation, figures, or
released artifacts.
