from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SSAB3D
from .blocks import DecoderBlock, EncoderBlock
from .contracts import SIMILARITY_MODES

OUTPUT_MODES = ("linear", "positive_softplus_residual")


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus for strictly positive tensors."""

    return value + torch.log(-torch.expm1(-value))


def positive_softplus_residual(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Apply an identity-centred residual while enforcing non-negative output.

    At residual=0 the output is approximately the non-negative source. This is a
    better fit for denoising in the non-negative asinh-SUV domain than either the
    historical tanh output or post-hoc clipping of a linear prediction.
    """

    base = source.clamp_min(float(epsilon))
    logits = _inverse_softplus(base) + residual
    return F.softplus(logits)


class SmartPETGenerator(nn.Module):
    """Seven-level 3D SMART-PET encoder-decoder for 128-cube patches."""

    def __init__(
        self,
        base_channels: int = 32,
        attention_levels: Sequence[int] = (2, 3),
        *,
        output_mode: str = "linear",
        output_epsilon: float = 1e-6,
        similarity_mode: str = "v030_luminance",
        encoder_convs_per_level: int = 1,
        channel_spatial_input_projection: bool = False,
        generator_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode={output_mode!r}; expected {OUTPUT_MODES}")
        if similarity_mode not in SIMILARITY_MODES:
            raise ValueError(
                f"Unsupported similarity_mode={similarity_mode!r}; expected {SIMILARITY_MODES}"
            )
        if encoder_convs_per_level not in {1, 2}:
            raise ValueError("encoder_convs_per_level must be 1 or 2")

        b = int(base_channels)
        channels = [b, 2 * b, 4 * b, 8 * b, 16 * b, 16 * b, 16 * b]
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    1,
                    channels[0],
                    first=True,
                    convs_per_level=encoder_convs_per_level,
                    use_spectral_norm=generator_spectral_norm,
                )
            ]
            + [
                EncoderBlock(
                    channels[i - 1],
                    channels[i],
                    use_norm=(i != len(channels) - 1),
                    convs_per_level=encoder_convs_per_level,
                    use_spectral_norm=generator_spectral_norm,
                )
                for i in range(1, len(channels))
            ]
        )
        self.attention_levels = tuple(int(v) for v in attention_levels)
        self.attention = nn.ModuleDict(
            {
                str(i): SSAB3D(
                    channels[i],
                    similarity_mode=similarity_mode,
                    channel_spatial_input_projection=channel_spatial_input_projection,
                )
                for i in self.attention_levels
            }
        )
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    channels[6],
                    channels[5],
                    dropout=0.5,
                    use_spectral_norm=generator_spectral_norm,
                ),
                DecoderBlock(
                    channels[5] + channels[5],
                    channels[4],
                    dropout=0.5,
                    use_spectral_norm=generator_spectral_norm,
                ),
                DecoderBlock(
                    channels[4] + channels[4],
                    channels[3],
                    dropout=0.5,
                    use_spectral_norm=generator_spectral_norm,
                ),
                DecoderBlock(
                    channels[3] + channels[3],
                    channels[2],
                    use_spectral_norm=generator_spectral_norm,
                ),
                DecoderBlock(
                    channels[2] + channels[2],
                    channels[1],
                    use_spectral_norm=generator_spectral_norm,
                ),
                DecoderBlock(
                    channels[1] + channels[1],
                    channels[0],
                    use_spectral_norm=generator_spectral_norm,
                ),
            ]
        )
        self.output = nn.ConvTranspose3d(channels[0] + channels[0], 1, 4, 2, 1)
        self.output_mode = str(output_mode)
        self.output_epsilon = float(output_epsilon)
        self.similarity_mode = str(similarity_mode)
        self.encoder_convs_per_level = int(encoder_convs_per_level)
        self.channel_spatial_input_projection = bool(channel_spatial_input_projection)
        self.generator_spectral_norm = bool(generator_spectral_norm)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim != 5 or source.shape[1] != 1:
            raise ValueError(f"Expected [B,1,D,H,W], got {tuple(source.shape)}")
        x = source
        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            if str(level) in self.attention:
                x = self.attention[str(level)](x)
            skips.append(x)
        for index, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-index - 2])
        residual_or_prediction = self.output(x)
        if self.output_mode == "linear":
            return residual_or_prediction
        return positive_softplus_residual(
            source,
            residual_or_prediction,
            epsilon=self.output_epsilon,
        )
