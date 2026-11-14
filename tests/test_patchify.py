import numpy as np
import torch
import pytest
from patchloss.operators.patchify import patchify, unpatchify
from patchloss.operators.reference_matrix import block_patch_matrix


@pytest.mark.parametrize(
    "shape,p", [((2, 3, 8, 12), 2), ((1, 1, 16, 16), 4), ((3, 2, 6, 9), 3), ((2, 1, 3, 5), 1)]
)
def test_exact_roundtrip_and_isometry(shape, p):
    x = torch.randn(shape, dtype=torch.float64)
    z = patchify(x, p)
    assert torch.equal(unpatchify(z, p, shape[2], shape[3]), x)
    assert torch.allclose(x.flatten(1).norm(dim=1), z.flatten(1).norm(dim=1), atol=1e-12)


def test_raw_reshape_never_rank_reducing():
    for p in (1, 2, 4):
        A = block_patch_matrix(np.eye(p * p), 4, 4, p)
        assert np.linalg.matrix_rank(A) == 16
        assert np.allclose(A.T @ A, np.eye(16))


def test_invalid_grid_rejected():
    with pytest.raises(ValueError):
        patchify(torch.zeros(1, 1, 7, 8), 2)
