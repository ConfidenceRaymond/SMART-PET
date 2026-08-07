from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_gradient_norm(
    loss: torch.Tensor | float,
    parameters: Iterable[nn.Parameter],
    *,
    retain_graph: bool = True,
) -> float:
    """Return the L2 norm of a loss gradient without mutating ``.grad`` fields."""

    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return 0.0
    parameter_list = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss,
        parameter_list,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().double().square().sum()
    return float(torch.sqrt(total))


def generator_gradient_attribution(*, seed: int = 2023) -> dict[str, Any]:
    """Demonstrate historical and corrected generator gradient connectivity.

    This deliberately uses tiny deterministic modules. It audits computation-graph
    connectivity, not image quality and not the numerical value of VGG features.
    """

    torch.manual_seed(int(seed))
    generator = nn.Conv3d(1, 1, kernel_size=1, bias=False)
    discriminator = nn.Conv3d(2, 1, kernel_size=1, bias=False)
    source = torch.randn(1, 1, 4, 4, 4)
    target = torch.randn(1, 1, 4, 4, 4)
    fake = generator(source)

    l1 = F.l1_loss(fake, target)
    historical_real_score = discriminator(torch.cat([source, target], dim=1))
    historical_gan = F.binary_cross_entropy_with_logits(
        historical_real_score,
        torch.ones_like(historical_real_score),
    )
    with torch.no_grad():
        historical_feature_tensor = F.l1_loss(fake, target)
    historical_feature_value = float(historical_feature_tensor.item())
    historical_total = historical_feature_value + 5e-3 * historical_gan + 1e-2 * l1

    corrected_fake_score = discriminator(torch.cat([source, fake], dim=1))
    corrected_gan = F.mse_loss(corrected_fake_score, torch.ones_like(corrected_fake_score))
    corrected_total = 100.0 * l1 + corrected_gan

    real_score = discriminator(torch.cat([source, target], dim=1))
    detached_fake_score = discriminator(torch.cat([source, fake.detach()], dim=1))
    discriminator_loss = 0.5 * (
        F.mse_loss(real_score, torch.ones_like(real_score))
        + F.mse_loss(detached_fake_score, torch.zeros_like(detached_fake_score))
    )

    generator_parameters = tuple(generator.parameters())
    l1_norm = loss_gradient_norm(l1, generator_parameters)
    historical_total_norm = loss_gradient_norm(historical_total, generator_parameters)
    report = {
        "seed": int(seed),
        "historical_executed": {
            "l1_gradient_norm": l1_norm,
            "gan_real_pair_gradient_norm": loss_gradient_norm(
                historical_gan, generator_parameters
            ),
            "vgg_detached_gradient_norm": loss_gradient_norm(
                historical_feature_value, generator_parameters
            ),
            "total_gradient_norm": historical_total_norm,
            "expected_l1_scale": 1e-2,
            "total_to_l1_gradient_ratio": historical_total_norm / l1_norm,
        },
        "corrected_lsgan": {
            "l1_gradient_norm": l1_norm,
            "gan_fake_pair_gradient_norm": loss_gradient_norm(corrected_gan, generator_parameters),
            "total_gradient_norm": loss_gradient_norm(corrected_total, generator_parameters),
            "lambda_l1": 100.0,
            "lambda_gan": 1.0,
        },
        "discriminator_step": {
            "generator_gradient_norm_with_detached_fake": loss_gradient_norm(
                discriminator_loss, generator_parameters, retain_graph=False
            )
        },
    }
    return report
