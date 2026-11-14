"""Unclipped isotropic Gaussian equal-prior location family."""

import numpy as np
from .base import rng_for, metadata, balanced_binary
from ..operators.linear_patch import random_projection
from ..theory.linear_algebra import row_projector


def build_signal(W, alpha, seed):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    g = np.random.default_rng(seed)
    m = W.shape[1]
    P = row_projector(W)
    row_draw, null_draw = g.normal(size=m), g.normal(size=m)
    row = np.zeros(m)
    null = np.zeros(m)
    if alpha > 0:
        row = P @ row_draw
        if np.linalg.norm(row) < 1e-10:
            raise ValueError("Positive alpha requires a nontrivial row space")
        row /= np.linalg.norm(row)
    if alpha < 1:
        null = (np.eye(m) - P) @ null_draw
        if np.linalg.norm(null) < 1e-10:
            raise ValueError("Alpha below one requires a nontrivial null space")
        null /= np.linalg.norm(null)
    return alpha * row + np.sqrt(max(0, 1 - alpha**2)) * null


def generate(seed, split, n_samples, config):
    h = int(config.get("height", 8))
    w = int(config.get("width", 8))
    p = int(config.get("patch_size", 8))
    if p != h or p != w:
        raise ValueError("Gaussian generator uses a single patch; set H=W=p")
    d = int(config.get("embedding_dim", 16))
    W = random_projection(d, p * p, seed)
    a = float(config.get("alpha", 0.5))
    v = build_signal(W, a, seed + 42)
    sigma = float(config.get("sigma", 0.5))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    g = rng_for(seed, split)
    labels = balanced_binary(n_samples, g)
    signs = labels * 2 - 1
    x = signs[:, None] * v[None, :] / 2 + g.normal(0, sigma, (n_samples, p * p))
    md = metadata(seed, split, n_samples, config, "gaussian")
    md.update({"projection": W, "signal": v, "sigma": sigma, "alpha": a, "unclipped": True})
    return x.reshape(-1, 1, h, w).astype(np.float64), labels, md
