from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import SIMILARITY_MODES


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


def _box_average_3d(
    x: torch.Tensor,
    *,
    kernel_size: int,
    padding: int,
) -> torch.Tensor:
    """Deterministic grouped-convolution equivalent of AvgPool3d with padded zeros."""

    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor [B,C,D,H,W], got shape={tuple(x.shape)}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    channels = int(x.shape[1])
    divisor = float(kernel_size**3)
    weight = torch.full(
        (channels, 1, kernel_size, kernel_size, kernel_size),
        1.0 / divisor,
        dtype=x.dtype,
        device=x.device,
    )
    return F.conv3d(x, weight, stride=1, padding=padding, groups=channels)


def _gaussian_window_3d(
    kernel_size: int,
    sigma: float,
    channels: int,
) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coordinates = torch.arange(kernel_size, dtype=torch.float32)
    coordinates = coordinates - kernel_size // 2
    one_d = torch.exp(-(coordinates.square()) / (2.0 * float(sigma) ** 2))
    one_d = one_d / one_d.sum()
    window = torch.einsum("i,j,k->ijk", one_d, one_d, one_d)
    return window.reshape(1, 1, kernel_size, kernel_size, kernel_size).expand(
        channels, 1, -1, -1, -1
    )


class SimilarityAttention3D(nn.Module):
    """SMART-PET similarity gate with explicit historical and corrected modes.

    ``v030_luminance`` preserves the v0.3.0 module and state-dict contract.
    ``paper_exact`` implements the published variance-squared Equation 4.
    ``scale_consistent`` uses the SSIM contrast form between local variance and
    its spatial reference, so numerator, denominator, and c2 share units.

    Both corrected modes implement Equation 5 as two separate convolutions:
    one receives the similarity map and the other receives the feature map.
    """

    def __init__(
        self,
        channels: int,
        *,
        mode: str = "v030_luminance",
        window_size: int = 11,
        sigma: float = 3.0,
        gate_kernel_size: int = 7,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        if mode not in SIMILARITY_MODES:
            raise ValueError(f"Unsupported similarity mode={mode!r}; expected {SIMILARITY_MODES}")
        if gate_kernel_size <= 0 or gate_kernel_size % 2 == 0:
            raise ValueError("gate_kernel_size must be a positive odd integer")
        if k2 <= 0:
            raise ValueError("k2 must be positive")

        self.channels = int(channels)
        self.mode = str(mode)
        self.k2 = float(k2)

        if self.mode == "v030_luminance":
            padding = gate_kernel_size // 2
            self.smooth = nn.AvgPool3d(gate_kernel_size, stride=1, padding=padding)
            self.gate = nn.Conv3d(
                self.channels * 2,
                self.channels,
                gate_kernel_size,
                padding=padding,
            )
        else:
            self.padding = window_size // 2
            self.register_buffer(
                "window",
                _gaussian_window_3d(window_size, sigma, self.channels),
                persistent=False,
            )
            gate_padding = gate_kernel_size // 2
            self.conv_similarity = nn.Conv3d(
                self.channels,
                self.channels,
                gate_kernel_size,
                padding=gate_padding,
            )
            self.conv_feature = nn.Conv3d(
                self.channels,
                self.channels,
                gate_kernel_size,
                padding=gate_padding,
            )

    def _smooth_v030(self, x: torch.Tensor) -> torch.Tensor:
        if torch.are_deterministic_algorithms_enabled():
            kernel_size = int(self.smooth.kernel_size)
            padding = int(self.smooth.padding)
            return _box_average_3d(
                x,
                kernel_size=kernel_size,
                padding=padding,
            )
        return self.smooth(x)

    def _variance(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window.to(device=x.device, dtype=x.dtype)
        mean = F.conv3d(x, window, padding=self.padding, groups=self.channels)
        second = F.conv3d(x.square(), window, padding=self.padding, groups=self.channels)
        return (second - mean.square()).clamp_min(0.0)

    def similarity_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},D,H,W], got {tuple(x.shape)}"
            )
        if self.mode == "v030_luminance":
            mean = self._smooth_v030(x)
            return (2.0 * x * mean + 1e-4) / (x.square() + mean.square() + 1e-4)

        variance = self._variance(x)
        data_range = x.amax(dim=(2, 3, 4), keepdim=True) - x.amin(
            dim=(2, 3, 4), keepdim=True
        )
        epsilon = torch.finfo(x.dtype).eps
        c2 = (self.k2 * data_range.clamp_min(epsilon)).square()
        if self.mode == "paper_exact":
            return (2.0 * variance + c2) / (2.0 * variance.square() + c2)

        reference_variance = variance.mean(dim=(2, 3, 4), keepdim=True)
        numerator = 2.0 * torch.sqrt(variance * reference_variance) + c2
        denominator = variance + reference_variance + c2
        return numerator / denominator.clamp_min(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        similarity = self.similarity_map(x)
        if self.mode == "v030_luminance":
            mean = self._smooth_v030(x)
            variance = torch.clamp(self._smooth_v030(x.square()) - mean.square(), min=0.0)
            descriptor = torch.cat([similarity, torch.sqrt(variance + 1e-6)], dim=1)
            return x * torch.sigmoid(self.gate(descriptor))
        gate = torch.sigmoid(self.conv_similarity(similarity) + self.conv_feature(x))
        return x * gate


class ChannelSpatialAttention3D(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        *,
        input_projection: bool = False,
    ) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(
            nn.Conv3d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv3d(hidden, channels, 1)
        )
        self.spatial = nn.Conv3d(2, 1, 7, padding=3)
        self.input_projection = nn.Conv3d(channels, 1, 1) if input_projection else None

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
        spatial_logits = self.spatial(spatial)
        if self.input_projection is not None:
            spatial_logits = spatial_logits + self.input_projection(x)
        return x_channel * torch.sigmoid(spatial_logits)


class SSAB3D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        similarity_mode: str = "v030_luminance",
        channel_spatial_input_projection: bool = False,
    ) -> None:
        super().__init__()
        self.self_attention = AxialSelfAttention3D(channels)
        self.similarity = SimilarityAttention3D(channels, mode=similarity_mode)
        self.channel_spatial = ChannelSpatialAttention3D(
            channels,
            input_projection=channel_spatial_input_projection,
        )
        self.fuse = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each branch performs its own single gating operation. Do not multiply
        # by x again here; the historical wrapper's outer multiplication created
        # an unintended x-squared gain profile.
        return self.fuse(self.self_attention(x) + self.similarity(x) + self.channel_spatial(x))
