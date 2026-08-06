from __future__ import annotations

import torch
import torch.nn as nn


def norm(channels: int) -> nn.Module:
    return nn.InstanceNorm3d(channels, affine=True)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        first: bool = False,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        kernel = 7 if first else 4
        padding = 3 if first else 1
        stride = 2
        layers: list[nn.Module] = [nn.Conv3d(in_channels, out_channels, kernel, stride, padding)]
        if (not first) and use_norm:
            layers.append(norm(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1),
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
