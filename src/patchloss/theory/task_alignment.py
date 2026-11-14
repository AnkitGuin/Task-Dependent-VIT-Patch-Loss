"""Mean-based between-class signal retention; not a universal MI estimator."""

import numpy as np
from .linear_algebra import row_projector


def tar_mean(tokens_or_patches, labels, W):
    x = np.asarray(tokens_or_patches, dtype=float)
    y = np.asarray(labels)
    W = np.asarray(W, dtype=float)
    if x.ndim != 3 or W.ndim != 2 or x.shape[-1] != W.shape[1]:
        raise ValueError("Expected sample,patch,patch_dim")
    if y.ndim != 1 or len(y) != len(x) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite samples and one label per sample")
    overall = x.mean(axis=0)
    projector = row_projector(W)
    den = 0.0
    num = 0.0
    for cls in np.unique(y):
        part = x[y == cls]
        diff = part.mean(axis=0) - overall
        weight = len(part) / len(x)
        den += weight * float(np.sum(diff**2))
        num += weight * float(np.sum((diff @ projector) ** 2))
    if den == 0:
        return None
    return float(np.clip(num / den, 0, 1))
