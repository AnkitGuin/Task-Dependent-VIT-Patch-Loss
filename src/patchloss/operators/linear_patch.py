"""Frozen, learnable and full-rank shared patch projectors."""

import numpy as np
import torch
from torch import nn
from .patchify import patchify


def random_projection(d: int, m: int, seed: int, orthonormal: bool = True):
    g = np.random.default_rng(seed)
    mat = g.standard_normal((d, m))
    if orthonormal:
        u, _, vt = np.linalg.svd(mat, full_matrices=False)
        mat = u @ vt
    return mat.astype(np.float64)


class LinearPatchTokenizer(nn.Module):
    def __init__(self, p=8, channels=1, d=16, seed=0, trainable=False, identity=False):
        super().__init__()
        self.p = p
        m = channels * p * p
        if identity and d != m:
            raise ValueError("Identity width must be m")
        self.linear = nn.Linear(m, d, bias=False)
        weights = np.eye(m) if identity else random_projection(d, m, seed)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weights, dtype=self.linear.weight.dtype))
        self.linear.weight.requires_grad_(trainable)

    def forward(self, images):
        return self.linear(patchify(images, self.p))

    @property
    def matrix(self):
        return self.linear.weight.detach().cpu().numpy().astype(np.float64)
