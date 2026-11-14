"""Exactly materialize an operator for small arrays; column i is output of image basis e_i."""

import numpy as np
import torch


def materialize_operator(operator, shape=(1, 8, 8), dtype=torch.float32):
    c, h, w = shape
    n = c * h * w
    training = operator.training
    operator.eval()
    try:
        with torch.no_grad():
            basis = torch.eye(n, dtype=dtype).reshape(n, c, h, w)
            offset = operator(torch.zeros(1, c, h, w, dtype=dtype)).reshape(1, -1)
            out = (operator(basis).reshape(n, -1) - offset).T
    finally:
        operator.train(training)
    return out.detach().cpu().numpy().astype(np.float64)


def block_patch_matrix(W, h, w, p, channels=1):
    from .patchify import patchify_np

    n = h * w * channels
    patches = patchify_np(np.eye(n, dtype=np.float64).reshape(n, channels, h, w), p)
    return np.einsum("bnm,dm->bnd", patches, W).reshape(n, -1).T
