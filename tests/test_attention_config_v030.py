from smartpet.models import SmartPETGenerator


def test_attention_levels_are_explicit():
    model = SmartPETGenerator(base_channels=1, attention_levels=(2, 3))
    assert model.attention_levels == (2, 3)
    assert set(model.attention) == {"2", "3"}
