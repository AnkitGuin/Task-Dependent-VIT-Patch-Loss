"""Standalone finite-dimensional verification, not a substitute for proofs."""

import numpy as np
import torch
from .linear_algebra import numerical_rank, row_projector, null_vector
from .bayes_gaussian import bayes_error
from ..operators.patchify import patchify, unpatchify
from ..operators.linear_patch import random_projection
from ..operators.reference_matrix import block_patch_matrix


def verify(seed=0):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.normal(size=(3, 2, 8, 12)))
    for p in [1, 2, 4]:
        assert torch.equal(unpatchify(patchify(x, p), p, 8, 12), x)
        assert torch.allclose(
            torch.linalg.vector_norm(patchify(x, p).reshape(3, -1), dim=1),
            torch.linalg.vector_norm(x.reshape(3, -1), dim=1),
        )
    W = random_projection(3, 4, seed)
    A = block_patch_matrix(W, 4, 4, 2)
    assert numerical_rank(A) == 4 * numerical_rank(W)
    P = row_projector(W)
    assert np.allclose(P @ P, P)
    u = null_vector(W)
    assert np.linalg.norm(W @ u) < 1e-9
    assert abs(bayes_error(u, W, 0.4) - 0.5) < 1e-8
    return {
        "checked": ["roundtrip", "isometry", "global_rank", "projector", "nullspace", "gaussian_chance"],
        "rank_global": numerical_rank(A),
        "max_null_residual": float(np.linalg.norm(W @ u)),
    }
