"""Exact two-point label collisions at each independently generated base image."""

import numpy as np
from .base import rng_for, metadata
from ..operators.patchify import unpatchify_np, patchify_np
from ..theory.linear_algebra import null_vector, row_projector
from ..operators.linear_patch import random_projection


def generate(seed, split, n_samples, config):
    h = int(config.get("height", 32))
    w = int(config.get("width", 32))
    p = int(config.get("patch_size", 8))
    d = int(config.get("embedding_dim", 16))
    if n_samples < 2 or n_samples % 2 or d >= p * p or h % p or w % p:
        raise ValueError("S1 needs even sample count, d < patch dimension, divisible H/W")
    g = rng_for(seed, split)
    W = np.asarray(config.get("projection", random_projection(d, p * p, seed)), dtype=float)
    if W.shape != (d, p * p):
        raise ValueError("Projection shape must match embedding and patch dimensions")
    u = null_vector(W)
    if config.get("pattern", "svd") == "checker":
        yy, xx = np.indices((p, p))
        checker = ((-1.0) ** (xx + yy)).reshape(-1)
        u = (np.eye(p * p) - row_projector(W)) @ checker
        if np.linalg.norm(u) <= 1e-10 * np.linalg.norm(checker):
            raise ValueError("Checker pattern has no numerically resolvable null component")
        u /= np.linalg.norm(u)
    patch_idx = int(config.get("patch_index", 0))
    if not 0 <= patch_idx < (h // p) * (w // p):
        raise ValueError("Invalid planted patch index")
    images = []
    labels = []
    base_ids = []
    for i in range(n_samples // 2):
        base = g.uniform(0.34, 0.66, size=(1, 1, h, w))
        z = patchify_np(base, p)
        # Keep all bases on the same hyperplane in the planted patch. Otherwise
        # random variation along u makes the raw class distributions overlap,
        # invalidating the claimed raw-image zero-risk oracle for this mixture.
        centered = z[0, patch_idx] - 0.5
        centered -= (centered @ u) * u
        centered *= min(1.0, 0.16 / max(float(np.max(np.abs(centered))), 1e-15))
        z[0, patch_idx] = 0.5 + centered
        eps = min(0.20 / max(1e-15, np.max(np.abs(u))), 0.20)
        for y, sign in ((0, -1), (1, 1)):
            z2 = z.copy()
            z2[0, patch_idx, :] += sign * eps * u
            images.append(unpatchify_np(z2, p, h, w)[0])
            labels.append(y)
            base_ids.append(f"{split}-{seed}-{i}")
    x = np.stack(images).astype(np.float64)
    md = metadata(seed, split, n_samples, config, "nullspace")
    md.update(
        {"projection": W, "null_direction": u, "base_ids": base_ids, "epsilon": eps, "patch_index": patch_idx}
    )
    assert x.min() >= 0 and x.max() <= 1
    return x, np.array(labels, dtype=np.int64), md
