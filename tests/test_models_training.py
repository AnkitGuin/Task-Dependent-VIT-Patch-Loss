import numpy as np
import torch
import pytest
from patchloss.models.tiny_vit import TinyViT
from patchloss.models.small_cnn import SmallCNN
from patchloss.training.train import fit_model


@pytest.mark.parametrize("p", [2, 4])
def test_tiny_vit_forward_backward_cpu(p):
    torch.set_num_threads(2)
    model = TinyViT(h=8, w=8, p=p, d=16, heads=2)
    logits = model(torch.randn(5, 1, 8, 8))
    assert logits.shape == (5, 2)
    logits.square().mean().backward()
    assert model.tokenizer.linear.weight.grad is not None


def test_frozen_remains_unchanged_learned_updates():
    torch.set_num_threads(2)
    x = np.random.default_rng(2).normal(size=(32, 1, 8, 8)).astype("float32")
    y = (x[:, 0, 0, 0] > 0).astype(np.int64)
    c = {"epochs": 2, "batch_size": 16, "lr": 0.003}
    for frozen in (True, False):
        model = TinyViT(h=8, w=8, p=4, d=16, heads=2, trainable=not frozen)
        old = model.tokenizer.matrix.copy()
        fit_model(model, (x, y), (x, y), (x, y), c, seed=1)
        assert (np.allclose(old, model.tokenizer.matrix)) == frozen


def test_smallcnn_shape():
    model = SmallCNN()
    assert model(torch.randn(4, 1, 8, 8)).shape == (4, 2)


def test_validation_only_checkpoint_selection():
    """Alter held-out labels; validation-selected epoch and training history must not move."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=(32, 1, 8, 8)).astype("float32")
    y = (x[:, 0, 0, 0] > 0).astype(np.int64)
    cfg = {"epochs": 3, "batch_size": 16, "lr": 0.002, "patience": 3}

    def run(test_labels):
        torch.manual_seed(33)
        model = TinyViT(h=8, w=8, p=4, d=16, heads=2, seed=5)
        return fit_model(model, (x, y), (x, y), (x, test_labels), cfg, seed=17)

    m1, h1, _ = run(y)
    m2, h2, _ = run(1 - y)
    assert m1["best_validation_epoch"] == m2["best_validation_epoch"]
    assert h1 == h2
    assert np.isclose(m1["accuracy"] + m2["accuracy"], 1.0)
