import numpy as np
import pytest
from patchloss.theory.bayes_gaussian import bayes_error, raw_error, independent_monte_carlo
from patchloss.theory.linear_algebra import row_projector
from patchloss.operators.linear_patch import random_projection
from patchloss.data.gaussian import build_signal, generate


def test_bayes_formula_endpoints_and_interior():
    W = random_projection(3, 8, 1)
    for a in [0, 0.5, 1]:
        v = build_signal(W, a, 33)
        assert np.isclose(np.linalg.norm(row_projector(W) @ v), a, atol=1e-10)
        prediction = bayes_error(v, W, 0.4)
        m = independent_monte_carlo(W, v, 0.4, 20000, seed=int(a * 100) + 5)
        assert abs(m["measured_error"] - prediction) < 0.025
        if a == 0:
            assert abs(prediction - 0.5) < 1e-10
        if a == 1:
            assert abs(prediction - raw_error(v, 0.4)) < 1e-10


def test_full_column_rank_no_loss():
    W = np.eye(8)
    v = np.arange(8.0)
    assert abs(bayes_error(v, W, 1) - raw_error(v, 1)) < 1e-12
    with pytest.raises(ValueError):
        bayes_error(v, W, 0)


def test_gaussian_not_clipped():
    x, y, m = generate(
        3,
        "train",
        256,
        {"height": 8, "width": 8, "patch_size": 8, "embedding_dim": 16, "alpha": 0.5, "sigma": 1.0},
    )
    assert (x < 0).any() and (x > 1).any()
    assert y.sum() == 128 and m["unclipped"]
