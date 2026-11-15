"""Fixed binomial smoothing with circular boundaries, then tokenization."""

import torch
import torch.nn.functional as F
from torch import nn


class BinomialLowPass(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
        ker = torch.outer(k, k)
        self.register_buffer("kernel", ker[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="circular"), self.kernel, groups=self.channels)
