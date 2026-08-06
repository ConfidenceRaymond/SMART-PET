from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialSelfAttention3D(nn.Module):
    """Memory-bounded axial attention over depth and in-plane positions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Conv3d(channels, channels, 1)
        self.key = nn.Conv3d(channels, channels, 1)
        self.value = nn.Conv3d(channels, channels, 3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        # Depth attention independently at each in-plane location.
        qd = q.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
        kd = k.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
        vd = v.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
        depth = self._attention(qd, kd, vd).reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)
        # In-plane attention independently for each depth plane.
        n = h * w
        qp = q.permute(0, 2, 3, 4, 1).reshape(b * d, n, c)
        kp = k.permute(0, 2, 3, 4, 1).reshape(b * d, n, c)
        vp = v.permute(0, 2, 3, 4, 1).reshape(b * d, n, c)
        plane = self._attention(qp, kp, vp).reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return x + self.gamma * (depth + plane)


class SimilarityAttention3D(nn.Module):
    """Local SSIM-inspired self-similarity gate from the SMART-PET design."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.smooth = nn.AvgPool3d(kernel_size, stride=1, padding=padding)
        self.gate = nn.Conv3d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.smooth(x)
        variance = torch.clamp(self.smooth(x * x) - mean * mean, min=0.0)
        similarity = (2.0 * x * mean + 1e-4) / (x * x + mean * mean + 1e-4)
        descriptor = torch.cat([similarity, torch.sqrt(variance + 1e-6)], dim=1)
        return x * torch.sigmoid(self.gate(descriptor))


class ChannelSpatialAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(
            nn.Conv3d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv3d(hidden, channels, 1)
        )
        self.spatial = nn.Conv3d(2, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool3d(x, 1)
        mx = F.adaptive_max_pool3d(x, 1)
        x_channel = x * torch.sigmoid(self.channel(avg) + self.channel(mx))
        spatial = torch.cat(
            [
                x_channel.mean(1, keepdim=True),
                x_channel.amax(1, keepdim=True),
            ],
            dim=1,
        )
        return x_channel * torch.sigmoid(self.spatial(spatial))


class SSAB3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_attention = AxialSelfAttention3D(channels)
        self.similarity = SimilarityAttention3D(channels)
        self.channel_spatial = ChannelSpatialAttention3D(channels)
        self.fuse = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.self_attention(x) + self.similarity(x) + self.channel_spatial(x))
