from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def norm(channels: int) -> nn.Module:
    return nn.InstanceNorm3d(channels, affine=True)


def maybe_spectral_norm(module: nn.Module, *, enabled: bool) -> nn.Module:
    return spectral_norm(module) if enabled else module


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        first: bool = False,
        use_norm: bool = True,
        convs_per_level: int = 1,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if convs_per_level not in {1, 2}:
            raise ValueError("convs_per_level must be 1 or 2")

        layers: list[nn.Module] = []
        if first and convs_per_level == 2:
            layers.extend(
                [
                    maybe_spectral_norm(
                        nn.Conv3d(in_channels, out_channels, 7, 1, 3),
                        enabled=use_spectral_norm,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                    maybe_spectral_norm(
                        nn.Conv3d(out_channels, out_channels, 4, 2, 1),
                        enabled=use_spectral_norm,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        else:
            kernel = 7 if first else 4
            padding = 3 if first else 1
            layers.append(
                maybe_spectral_norm(
                    nn.Conv3d(in_channels, out_channels, kernel, 2, padding),
                    enabled=use_spectral_norm,
                )
            )
            if (not first) and use_norm:
                layers.append(norm(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if convs_per_level == 2:
                layers.append(
                    maybe_spectral_norm(
                        nn.Conv3d(out_channels, out_channels, 3, 1, 1),
                        enabled=use_spectral_norm,
                    )
                )
                if use_norm:
                    layers.append(norm(out_channels))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            maybe_spectral_norm(
                nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1),
                enabled=use_spectral_norm,
            ),
            norm(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if x.shape[2:] != skip.shape[2:]:
            raise ValueError(f"Decoder/skip shape mismatch: {x.shape} vs {skip.shape}")
        return torch.cat([x, skip], dim=1)
