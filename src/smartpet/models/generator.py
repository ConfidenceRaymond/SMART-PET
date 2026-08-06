from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SSAB3D
from .blocks import DecoderBlock, EncoderBlock

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
    ) -> None:
        super().__init__()
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode={output_mode!r}; expected {OUTPUT_MODES}")
        b = int(base_channels)
        channels = [b, 2 * b, 4 * b, 8 * b, 16 * b, 16 * b, 16 * b]
        self.encoders = nn.ModuleList(
            [EncoderBlock(1, channels[0], first=True)]
            + [
                EncoderBlock(
                    channels[i - 1],
                    channels[i],
                    use_norm=(i != len(channels) - 1),
                )
                for i in range(1, len(channels))
            ]
        )
        self.attention_levels = tuple(int(v) for v in attention_levels)
        self.attention = nn.ModuleDict({str(i): SSAB3D(channels[i]) for i in self.attention_levels})
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(channels[6], channels[5], dropout=0.5),
                DecoderBlock(channels[5] + channels[5], channels[4], dropout=0.5),
                DecoderBlock(channels[4] + channels[4], channels[3], dropout=0.5),
                DecoderBlock(channels[3] + channels[3], channels[2]),
                DecoderBlock(channels[2] + channels[2], channels[1]),
                DecoderBlock(channels[1] + channels[1], channels[0]),
            ]
        )
        self.output = nn.ConvTranspose3d(channels[0] + channels[0], 1, 4, 2, 1)
        self.output_mode = str(output_mode)
        self.output_epsilon = float(output_epsilon)

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
