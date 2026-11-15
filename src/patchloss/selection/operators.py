"""Equal-feature-budget patch maps; weights are fit without diagnostic/test examples."""

from pathlib import Path
import numpy as np
import torch
from scipy.fft import dct
from ..operators.patchify import patchify_np
from ..operators.linear_patch import random_projection


def dct_rows(p, channels, d):
    basis = dct(np.eye(p), type=2, norm="ortho", axis=0)
    color = dct(np.eye(channels), type=2, norm="ortho", axis=0)
    order = sorted(
        (i + j, i * i + j * j, c, i, j) for c in range(channels) for i in range(p) for j in range(p)
    )
    rows = []
    for _, __, c, i, j in order[:d]:
        row = color[c, :, None, None] * np.outer(basis[i], basis[j])[None]
        rows.append(row.reshape(-1))
    return np.stack(rows)


def fit_bank(operator_images, budget, patch_sizes, seed, config):
    c, h, w = operator_images.shape[1:]
    bank = []
    for p in patch_sizes:
        n = h * w // (p * p)
        if h % p or w % p or budget % n:
            raise ValueError("Budget must be an integer multiple of the token count")
        d = budget // n
        m = c * p * p
        if not 0 < d <= m:
            raise ValueError("Requested patch dimension is invalid")
        patches = patchify_np(operator_images, p)
        flat = patches.reshape(-1, m)
        rng = np.random.default_rng(seed + 100 * p)
        if len(flat) > int(config["pca_patch_cap"]):
            flat = flat[rng.choice(len(flat), int(config["pca_patch_cap"]), replace=False)]
        centered = flat - flat.mean(0)
        covariance = centered.T @ centered / len(centered)
        _, vectors = np.linalg.eigh(covariance)
        pca = vectors[:, -d:][:, ::-1].T
        matrices = {"random": random_projection(d, m, seed + 1000 * p), "dct": dct_rows(p, c, d), "pca": pca}
        for kind, W in matrices.items():
            bank.append(
                {
                    "candidate": f"p{p:02d}_{kind}",
                    "family": kind,
                    "patch_size": p,
                    "token_count": n,
                    "embedding_dim": d,
                    "patch_dim": m,
                    "matrix": W,
                    "token_budget": budget,
                    "projection_flops_proxy": 2 * n * d * m,
                }
            )
    if config.get("pretrained", True):
        p = 16
        if c != 3 or p not in patch_sizes:
            raise ValueError("Pretrained DeiT patch transfer requires RGB and p=16")
        state = torch.load(Path(config["data_root"]) / "deit_tiny.pth", map_location="cpu", weights_only=True)
        matrix = state["model"]["patch_embed.proj.weight"].numpy().astype(np.float64).reshape(192, 768)
        _, s, vt = np.linalg.svd(matrix, full_matrices=False)
        n = h * w // (p * p)
        d = budget // n
        if d > 192:
            raise ValueError("Requested DeiT compression width exceeds its native 192 rows")
        matrices = {"deit_native_svd": s[:d, None] * vt[:d], "deit_rowspace": vt[:d]}
        for kind, W in matrices.items():
            bank.append(
                {
                    "candidate": f"p{p:02d}_{kind}",
                    "family": kind,
                    "patch_size": p,
                    "token_count": n,
                    "embedding_dim": d,
                    "patch_dim": 768,
                    "matrix": W,
                    "token_budget": budget,
                    "projection_flops_proxy": 2 * n * d * 768,
                }
            )
    return sorted(bank, key=lambda item: item["candidate"])


def transform(images, candidate):
    patches = patchify_np(images, candidate["patch_size"])
    return (patches @ candidate["matrix"].T).reshape(len(images), -1)
