"""High-frequency signal vs nuisance; matched low-pass and raw input evaluations."""

import numpy as np
from .base import rng_for, metadata, balanced_binary


def generate(seed, split, n_samples, config):
    h = int(config.get("height", 16))
    w = int(config.get("width", 16))
    g = rng_for(seed, split)
    labels = balanced_binary(n_samples, g)
    family = config.get("family", "signal")
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    checker = (-1.0) ** (xx + yy)
    low = np.cos(2 * np.pi * xx / w)
    images = []
    for lab in labels:
        sign = 2 * lab - 1
        # zero-mean high frequency contrasts, noise/phase from class-independent distribution
        if family == "signal":
            im = 0.5 + sign * 0.21 * checker + 0.16 * g.normal(size=(h, w))
        elif family == "nuisance":
            phase = int(g.integers(2))
            im = (
                0.5
                + sign * 0.18 * low
                + (0.21 if phase == 0 else -0.21) * checker
                + 0.12 * g.normal(size=(h, w))
            )
        else:
            raise ValueError("family must be signal or nuisance")
        images.append(np.clip(im, 0, 1))
    md = metadata(seed, split, n_samples, config, "alias_tradeoff")
    md["family"] = family
    return np.asarray(images, dtype=np.float32)[:, None], labels, md
