"""Lossless ordered channel-major pixel patch extraction."""

import torch
import torch.nn.functional as F
import numpy as np


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    if (
        not isinstance(p, int)
        or p < 1
        or x.ndim != 4
        or min(x.shape[1:]) < 1
        or x.shape[2] % p
        or x.shape[3] % p
    ):
        raise ValueError("Expect B,C,H,W with H,W divisible by p")
    return F.unfold(x, kernel_size=p, stride=p).transpose(1, 2).contiguous()


def unpatchify(z: torch.Tensor, p: int, height: int, width: int) -> torch.Tensor:
    if (
        not isinstance(p, int)
        or p < 1
        or min(height, width) < 1
        or height % p
        or width % p
        or z.ndim != 3
        or z.shape[1] != height * width // (p * p)
    ):
        raise ValueError("Invalid patch grid")
    if z.shape[-1] % (p * p):
        raise ValueError("Patch feature dimension is not C*p*p")
    return F.fold(z.transpose(1, 2), output_size=(height, width), kernel_size=p, stride=p)


def patchify_np(x: np.ndarray, p: int) -> np.ndarray:
    return patchify(torch.from_numpy(np.ascontiguousarray(x)), p).numpy()


def unpatchify_np(z: np.ndarray, p: int, h: int, w: int) -> np.ndarray:
    return unpatchify(torch.from_numpy(np.ascontiguousarray(z)), p, h, w).numpy()
