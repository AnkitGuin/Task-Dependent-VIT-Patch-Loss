"""Convolutional stem and anti-aliased tokenization variants."""

from torch import nn
from ..operators.lowpass import BinomialLowPass
from ..operators.linear_patch import LinearPatchTokenizer


class FilteredTokenizer(nn.Module):
    def __init__(self, p=4, channels=1, d=8, seed=0):
        super().__init__()
        self.filter = BinomialLowPass(channels)
        self.tokenizer = LinearPatchTokenizer(p, channels, d, seed, False)

    def forward(self, x):
        return self.tokenizer(self.filter(x))


class ConvStem(nn.Module):
    def __init__(self, channels=1, d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, d, 3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x).flatten(2).transpose(1, 2)
