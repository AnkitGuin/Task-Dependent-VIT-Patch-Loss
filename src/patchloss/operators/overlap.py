"""Overlapping shared linear projection. Rank may or may not increase."""

import torch
import torch.nn.functional as F
from torch import nn
from .linear_patch import random_projection


class OverlapTokenizer(nn.Module):
    def __init__(self, p=4, stride=2, channels=1, d=8, seed=0, trainable=False):
        super().__init__()
        if stride < 1 or stride > p:
            raise ValueError("Stride must be in [1,p]")
        self.p, self.stride = p, stride
        self.linear = nn.Linear(channels * p * p, d, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(random_projection(d, channels * p * p, seed)).float())
        self.linear.weight.requires_grad_(trainable)

    def forward(self, x):
        if (
            x.ndim != 4
            or min(x.shape[-2:]) < self.p
            or any((dim - self.p) % self.stride for dim in x.shape[-2:])
        ):
            raise ValueError("Overlapping windows must cover the full image without dropping border pixels")
        return self.linear(F.unfold(x, self.p, stride=self.stride).transpose(1, 2))
