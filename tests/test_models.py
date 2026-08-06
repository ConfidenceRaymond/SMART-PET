import torch

from smartpet.models import PatchDiscriminator3D, SmartPETGenerator


def test_model_shapes_on_128_cube():
    generator = SmartPETGenerator(base_channels=2, attention_levels=())
    discriminator = PatchDiscriminator3D(base_channels=2)
    x = torch.randn(1, 1, 128, 128, 128)
    with torch.no_grad():
        y = generator(x)
        logits = discriminator(x, y)
    assert y.shape == x.shape
    assert logits.ndim == 5 and logits.shape[1] == 1
