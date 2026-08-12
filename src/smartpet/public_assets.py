from __future__ import annotations

from dataclasses import dataclass

PUBLIC_ASSET_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1XqEI6W30OsrWusMycX0QB8E8DoFURhWh?usp=drive_link"
)


@dataclass(frozen=True)
class PublicAsset:
    relative_path: str
    sha256: str
    profiles: tuple[str, ...]
    description: str


PUBLIC_ASSETS: tuple[PublicAsset, ...] = (
    PublicAsset(
        relative_path="templates/csymT.nii.gz",
        sha256="d28d312d3c895c226dbd61947b77691c6d850396c035015399bd4cfdeed4c291",
        profiles=("inference", "finetune", "all"),
        description="SMART-PET MNI reference",
    ),
    PublicAsset(
        relative_path="templates/MNI152_T1_1mm_brain_mask.nii.gz",
        sha256="274b41c4cf787ada4ce683524301ee052d1ef64b208569c05ce7e9c00717404e",
        profiles=("inference", "finetune", "all"),
        description="Fixed whole-brain evaluation mask",
    ),
    PublicAsset(
        relative_path="weights/smartpet_g001_parent_v0.3.1.pt",
        sha256="f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0",
        profiles=("inference", "finetune", "all"),
        description="Recommended G0.01-parent inference weights",
    ),
    PublicAsset(
        relative_path="checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt",
        sha256="2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee",
        profiles=("finetune", "all"),
        description="Full G0.01-parent checkpoint for fine-tuning",
    ),
    PublicAsset(
        relative_path="weights/smartpet_g001_external_adapted_v0.3.1.pt",
        sha256="aecd3b0c15f0b0b90fc6e2142412562ceacc7a5aacd440d37c3476e7dc89b797",
        profiles=("all",),
        description="Domain-specific external-adapted inference weights",
    ),
    PublicAsset(
        relative_path="weights/smartpet_v0.3.0_epoch4_inference.pt",
        sha256="ddc79a1940032754f5b719688f6affd2612d3566b9c4f03a0d2e41ce1f5b1d25",
        profiles=("all",),
        description="Historical v0.3.0 epoch-4 inference weights",
    ),
)


def assets_for_profile(profile: str) -> tuple[PublicAsset, ...]:
    if profile not in {"inference", "finetune", "all"}:
        raise ValueError("profile must be inference, finetune, or all")
    return tuple(asset for asset in PUBLIC_ASSETS if profile in asset.profiles)
