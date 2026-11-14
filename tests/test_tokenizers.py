import numpy as np
import torch
import torch.nn.functional as F
from patchloss.operators.linear_patch import LinearPatchTokenizer
from patchloss.operators.overlap import OverlapTokenizer
from patchloss.operators.reference_matrix import materialize_operator, block_patch_matrix


def test_conv2d_and_dense_patch_linear_equivalence():
    tok = LinearPatchTokenizer(p=2, channels=2, d=3, seed=0)
    x = torch.randn(2, 2, 6, 8)
    conv = F.conv2d(x, tok.linear.weight.reshape(3, 2, 2, 2), stride=2).flatten(2).transpose(1, 2)
    assert torch.allclose(tok(x), conv, atol=1e-6)


def test_small_reference_operator_matrix():
    tok = LinearPatchTokenizer(p=2, d=3, seed=0).double()
    A = materialize_operator(tok, (1, 4, 4), dtype=torch.float64)
    expected = block_patch_matrix(tok.matrix, 4, 4, 2)
    assert np.allclose(A, expected)
    assert A.shape == (12, 16)


def test_overlap_not_automatically_invertible():
    op = OverlapTokenizer(p=4, stride=2, d=3).double()
    A = materialize_operator(op, (1, 8, 8), dtype=torch.float64)
    assert A.shape == (27, 64)
    assert np.linalg.matrix_rank(A) < 64


def test_frozen_vs_trainable_gradient():
    a = LinearPatchTokenizer(p=2, d=3, trainable=False)
    b = LinearPatchTokenizer(p=2, d=3, trainable=True)
    assert not a.linear.weight.requires_grad and b.linear.weight.requires_grad
