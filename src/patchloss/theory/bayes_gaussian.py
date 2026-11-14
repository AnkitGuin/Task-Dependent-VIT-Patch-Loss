"""Equal-prior isotropic Gaussian risk and an independent token-space oracle."""

import numpy as np
from scipy.stats import norm, binomtest
from .linear_algebra import row_projector


def _inputs(v, sigma, W=None):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError("v must be a finite vector")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if W is not None:
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2 or W.shape[1] != len(v) or not np.isfinite(W).all():
            raise ValueError("W must be finite with one column per signal coordinate")
    return v, W


def bayes_error(v, W, sigma):
    v, W = _inputs(v, sigma, W)
    return float(norm.cdf(-np.linalg.norm(row_projector(W) @ v) / (2 * sigma)))


def raw_error(v, sigma):
    v, _ = _inputs(v, sigma)
    return float(norm.cdf(-np.linalg.norm(v) / (2 * sigma)))


def independent_monte_carlo(W, v, sigma, n, seed):
    """Classify sampled Z=WX via feature whitening, without row_projector.

    IID equal-prior labels make binomial intervals valid even for a constant
    zero-signal classifier. All decisions use tokens; no raw-space direction is
    available to amplify roundoff in an intended null-space contrast.
    """
    v, W = _inputs(v, sigma, W)
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    g = np.random.default_rng(seed)
    y = 2 * g.integers(0, 2, size=n) - 1
    x = y[:, None] * v[None, :] / 2 + g.normal(0, sigma, (n, len(v)))
    # Normalize before whitening to avoid overflow/underflow under global scale.
    scale = np.max(np.abs(W)) if W.size else 0.0
    normalized = W / scale if scale else W
    u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    keep = singular > (1e-10 * singular[0]) if singular.size else np.zeros(0, dtype=bool)
    z_white = ((x @ normalized.T) @ u[:, keep]) / singular[keep]
    contrast = ((normalized @ v) @ u[:, keep]) / singular[keep]
    decision = (
        np.ones(n, dtype=int) if np.linalg.norm(contrast) == 0 else np.where(z_white @ contrast >= 0, 1, -1)
    )
    errors = int((decision != y).sum())
    p = errors / n
    ci = binomtest(errors, n).proportion_ci(confidence_level=0.95, method="exact")
    return {
        "measured_error": float(p),
        "num_errors": errors,
        "num_test_samples": int(n),
        "mc_se": float(np.sqrt(p * (1 - p) / n)),
        "ci95_low": float(ci.low),
        "ci95_high": float(ci.high),
        "ci_method": "Clopper-Pearson exact binomial",
        "oracle": "token-space SVD whitening",
        "label_sampling": "iid equal prior",
    }
