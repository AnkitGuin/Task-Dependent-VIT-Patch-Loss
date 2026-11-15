"""Transparent small ViT with positional encodings and mean pooling."""

import torch
from torch import nn
from ..operators.linear_patch import LinearPatchTokenizer


class TinyViT(nn.Module):
    def __init__(
        self, h=16, w=16, p=4, channels=1, d=32, heads=2, depth=2, trainable=True, seed=0, classes=2
    ):
        super().__init__()
        if h % p or w % p or d % heads:
            raise ValueError("Invalid image/patch grid or heads")
        self.tokenizer = LinearPatchTokenizer(p, channels, d, seed, trainable=trainable)
        n = (h // p) * (w // p)
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, classes)

    def forward(self, images):
        return self.head(self.norm(self.encoder(self.tokenizer(images) + self.pos).mean(1)))
