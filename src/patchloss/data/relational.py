"""Balanced XOR with analytically identical class means at every pixel."""

import numpy as np
from .base import rng_for, metadata


def generate(seed, split, n_samples, config):
    h = int(config.get("height", 16))
    w = int(config.get("width", 16))
    if n_samples % 4 or min(h, w) < 8:
        raise ValueError("XOR requires a sample count divisible by 4 and 8x8 or larger images")
    g = rng_for(seed, split)
    shifts = int(config.get("phase_shift", 0))
    noise = float(config.get("noise", 0.03))
    # All four bit combinations repeat equally => class mean (on signal pixels) matches.
    pairs = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (n_samples // 4, 1))
    g.shuffle(pairs)
    imgs = []
    labels = []
    for a, b in pairs:
        im = np.full((h, w), 0.5)
        im[2:4, 2:4] = 0.2 + 0.6 * a
        im[h - 4 : h - 2, w - 4 : w - 2] = 0.2 + 0.6 * b
        im = np.roll(im, shifts, axis=0)
        im += g.normal(0, noise, (h, w))
        imgs.append(np.clip(im, 0, 1))
        labels.append(int(a != b))
    md = metadata(seed, split, n_samples, config, "relational")
    md["phase_shift"] = shifts
    return np.array(imgs, dtype=np.float32)[:, None], np.array(labels, dtype=np.int64), md
