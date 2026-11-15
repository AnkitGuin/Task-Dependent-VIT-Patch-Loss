"""SVD geometry using a relative singular-value threshold."""

import numpy as np


def _matrix(W):
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or not np.isfinite(W).all():
        raise ValueError("Expected a finite two-dimensional matrix")
    return W


def numerical_rank(W, rtol=1e-10):
    W = _matrix(W)
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative")
    s = np.linalg.svd(W, compute_uv=False)
    return int(np.count_nonzero(s > rtol * s[0])) if s.size else 0


def row_projector(W, rtol=1e-10):
    W = _matrix(W)
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative")
    _, singular, vt = np.linalg.svd(W, full_matrices=False)
    rank = int((singular > rtol * singular[0]).sum()) if len(singular) else 0
    return vt[:rank].T @ vt[:rank]


def null_vector(W):
    W = np.asarray(W, dtype=np.float64)
    _, _, vt = np.linalg.svd(W, full_matrices=True)
    r = numerical_rank(W)
    if r == W.shape[1]:
        raise ValueError("No nonzero null-space vector")
    v = vt[-1].copy()
    return v / np.linalg.norm(v)


def alignment(v, W):
    v = np.asarray(v, dtype=float)
    if not np.linalg.norm(v):
        raise ValueError("v must be nonzero")
    return float(np.linalg.norm(row_projector(W) @ v) / np.linalg.norm(v))


def block_projection(v, W, n_patches):
    m = W.shape[1]
    v = np.asarray(v).reshape(n_patches, m)
    return (v @ row_projector(W)).reshape(-1)


def spectrum_info(W):
    s = np.linalg.svd(W, compute_uv=False)
    r = numerical_rank(W)
    return {
        "singular_values": s.tolist(),
        "rank": r,
        "nullity": W.shape[1] - r,
        "condition_nonzero": float(s[0] / s[r - 1]) if r else None,
    }
