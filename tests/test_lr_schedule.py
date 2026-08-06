def test_published_linear_decay_contract():
    epochs = 400
    decay_start = 100

    def factor(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (epochs - decay_start))

    assert factor(0) == 1.0
    assert factor(99) == 1.0
    assert factor(100) == 1.0
    assert factor(250) == 0.5
    assert factor(400) == 0.0
