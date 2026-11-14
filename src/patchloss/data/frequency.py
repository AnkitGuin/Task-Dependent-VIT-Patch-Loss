"""Balanced grating-orientation, marker, nuisance, and translation tasks."""

import numpy as np
from .base import rng_for, metadata, balanced_binary


def generate(seed, split, n_samples, config):
    h = int(config.get("height", 32))
    w = int(config.get("width", 32))
    g = rng_for(seed, split)
    labels = balanced_binary(n_samples, g)
    mode = config.get("mode", "frequency")
    if mode not in {"frequency", "marker", "nuisance"}:
        raise ValueError(f"Unknown frequency task mode: {mode}")
    freq = int(config.get("frequency", 4))
    shift = int(config.get("phase_shift", 0))
    size = int(config.get("marker_size", 2))
    noise = float(config.get("noise", 0.035))
    y, x = np.mgrid[:h, :w]
    images = []
    for label in labels:
        if mode == "marker":
            im = np.full((h, w), 0.5)
            # class-selecting center/other location; fixed template-size signal
            cx = w // 4 if label == 0 else 3 * w // 4
            cy = h // 2
            xx = (cx + shift) % w
            for dy in range(size):
                for dx in range(size):
                    im[(cy + dy) % h, (xx + dx) % w] = 0.85
        elif mode == "nuisance":
            # low-freq class contrast, grating independently random class-agnostic nuisance
            im = 0.5 + 0.22 * (2 * label - 1) * np.cos(2 * np.pi * (x + shift) / w)
            phi = float(g.uniform(0, 2 * np.pi))
            im = im + 0.17 * np.cos(2 * np.pi * freq * y / h + phi)
        else:
            # orientation determines label, random phase is class-independent
            phi = float(g.uniform(0, 2 * np.pi))
            coord = (x + shift) if label == 0 else (y + shift)
            im = 0.5 + 0.25 * np.cos(2 * np.pi * freq * coord / (w if label == 0 else h) + phi)
        im = im + g.normal(0, noise, size=(h, w))
        images.append(np.clip(im, 0, 1))
    md = metadata(seed, split, n_samples, config, "frequency")
    md.update({"frequency": freq, "mode": mode, "phase_shift": shift, "marker_size": size})
    return np.array(images, dtype=np.float32)[:, None, :, :], labels, md
