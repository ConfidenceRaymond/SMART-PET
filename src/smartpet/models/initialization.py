from __future__ import annotations

import torch
import torch.nn as nn


def _weight_parameter(module: nn.Module) -> torch.Tensor:
    weight_orig = getattr(module, "weight_orig", None)
    if isinstance(weight_orig, torch.Tensor):
        return weight_orig
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is not None and hasattr(parametrizations, "weight"):
        original = getattr(parametrizations.weight, "original", None)
        if isinstance(original, torch.Tensor):
            return original
    return module.weight


def initialize_gan_weights(
    module: nn.Module,
    *,
    conv_mean: float = 0.0,
    conv_std: float = 0.02,
    norm_mean: float = 1.0,
    norm_std: float = 0.02,
) -> None:
    """Initialize SMART-PET/pix2pix-style GAN modules explicitly.

    Convolution and transposed-convolution weights use Normal(0, 0.02).
    Affine normalization scales use Normal(1, 0.02). All available biases are zero.
    Spectral-normalized modules initialize their underlying original parameter.
    """

    if isinstance(module, nn.Conv3d | nn.ConvTranspose3d | nn.Linear):
        nn.init.normal_(_weight_parameter(module), mean=conv_mean, std=conv_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, nn.BatchNorm3d | nn.InstanceNorm3d | nn.GroupNorm):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=norm_mean, std=norm_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def initialize_identity_residual_head(module: nn.Module) -> None:
    """Zero a residual prediction head so the initial model is the identity map."""

    if not isinstance(module, nn.Conv3d | nn.ConvTranspose3d):
        raise TypeError("Residual head must be Conv3d or ConvTranspose3d")
    nn.init.zeros_(_weight_parameter(module))
    if module.bias is not None:
        nn.init.zeros_(module.bias)
