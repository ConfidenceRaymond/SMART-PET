from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SIMILARITY_MODES = ("v030_luminance", "paper_exact", "scale_consistent")
ARCHITECTURE_CONFIG_FIELDS = (
    "similarity_mode",
    "encoder_convs_per_level",
    "channel_spatial_input_projection",
    "generator_spectral_norm",
    "discriminator_spectral_norm",
)
V030_ARCHITECTURE_DEFAULTS: dict[str, Any] = {
    "similarity_mode": "v030_luminance",
    "encoder_convs_per_level": 1,
    "channel_spatial_input_projection": False,
    "generator_spectral_norm": False,
    "discriminator_spectral_norm": False,
}


def normalize_architecture_config(
    raw: Mapping[str, Any],
    *,
    allow_v030_defaults: bool,
) -> dict[str, Any]:
    """Validate the architecture fields that determine model reconstruction.

    SMART-PET v0.3.0 artifacts predate these explicit fields. They may be
    resolved to the immutable v0.3.0 contract only when the caller opts into
    that schema migration. Partially specified architecture metadata is always
    rejected.
    """

    present = {field for field in ARCHITECTURE_CONFIG_FIELDS if raw.get(field) is not None}
    if not present:
        if not allow_v030_defaults:
            raise ValueError(
                "Architecture config is missing required fields: "
                f"{list(ARCHITECTURE_CONFIG_FIELDS)}"
            )
        return dict(V030_ARCHITECTURE_DEFAULTS)

    missing = [field for field in ARCHITECTURE_CONFIG_FIELDS if raw.get(field) is None]
    if missing:
        raise ValueError(f"Architecture config is partially specified; missing fields: {missing}")

    similarity_mode = raw["similarity_mode"]
    if not isinstance(similarity_mode, str) or similarity_mode not in SIMILARITY_MODES:
        raise ValueError(
            f"Unsupported similarity_mode={similarity_mode!r}; expected {SIMILARITY_MODES}"
        )

    encoder_convs = raw["encoder_convs_per_level"]
    if isinstance(encoder_convs, bool) or not isinstance(encoder_convs, int):
        raise TypeError("encoder_convs_per_level must be an integer")
    if encoder_convs not in {1, 2}:
        raise ValueError("encoder_convs_per_level must be 1 or 2")

    boolean_fields = (
        "channel_spatial_input_projection",
        "generator_spectral_norm",
        "discriminator_spectral_norm",
    )
    for field in boolean_fields:
        if not isinstance(raw[field], bool):
            raise TypeError(f"{field} must be a boolean")

    return {
        "similarity_mode": similarity_mode,
        "encoder_convs_per_level": int(encoder_convs),
        "channel_spatial_input_projection": bool(raw["channel_spatial_input_projection"]),
        "generator_spectral_norm": bool(raw["generator_spectral_norm"]),
        "discriminator_spectral_norm": bool(raw["discriminator_spectral_norm"]),
    }
