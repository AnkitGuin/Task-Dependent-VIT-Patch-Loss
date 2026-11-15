import numpy as np
import pytest
from patchloss.operators.linear_patch import random_projection
from patchloss.operators.reference_matrix import block_patch_matrix
from patchloss.theory.linear_algebra import (
    numerical_rank,
    row_projector,
    null_vector,
    alignment,
    spectrum_info,
)


def test_global_rank_and_nullity():
    W = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    A = block_patch_matrix(W, 4, 4, 2)
    assert numerical_rank(A) == 4 * numerical_rank(W) == 8
    assert A.shape[1] - numerical_rank(A) == 8


def test_null_vector_projector_and_spectrum():
    W = random_projection(3, 9, seed=1)
    u = null_vector(W)
    P = row_projector(W)
    assert np.linalg.norm(W @ u) < 1e-11
    assert np.allclose(P, P.T)
    assert np.allclose(P @ P, P)
    assert np.linalg.norm(P @ u) < 1e-10
    v = W[0]
    assert np.allclose(P @ v, v)
    assert abs(alignment(v, W) - 1) < 1e-10
    assert abs(alignment(u, W)) < 1e-10
    assert spectrum_info(W)["nullity"] == 6


def test_full_rank_positive_control():
    W = np.eye(4)
    assert numerical_rank(W) == 4
    with pytest.raises(ValueError):
        null_vector(W)
    assert np.allclose(np.linalg.pinv(W) @ W, np.eye(4))
