from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator3D(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        b = int(base_channels)

        def block(cin: int, cout: int, stride: int, use_norm: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv3d(cin, cout, 4, stride, 1)]
            if use_norm:
                layers.append(nn.InstanceNorm3d(cout, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(2, b, 2, False),
            *block(b, 2 * b, 2, True),
            *block(2 * b, 4 * b, 2, True),
            *block(4 * b, 8 * b, 1, True),
            nn.Conv3d(8 * b, 1, 4, 1, 1),
        )

    def forward(self, source: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([source, candidate], dim=1))
