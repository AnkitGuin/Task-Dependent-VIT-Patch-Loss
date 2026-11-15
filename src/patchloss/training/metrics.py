"""Simple binary/multiclass metrics and model scale proxy."""

import numpy as np


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def transformer_flops_proxy(n, m, d, depth, classes=2):
    """Forward per image, 2 FLOPs/MAC; QKV/output + FFN(2d) + attention.

    Includes patch embedding and classifier. Omits softmax, GELU, layer norms,
    biases, positional addition, pooling, backward, and optimizer operations.
    """
    return int(2 * n * m * d + depth * (16 * n * d * d + 4 * n * n * d) + 2 * d * classes)


def classification_metrics(logits, labels):
    from scipy.special import logsumexp

    lp = np.asarray(logits)
    y = np.asarray(labels)
    if (
        lp.ndim != 2
        or y.ndim != 1
        or len(y) == 0
        or len(lp) != len(y)
        or not np.isfinite(lp).all()
        or not np.issubdtype(y.dtype, np.integer)
        or np.any(y < 0)
        or np.any(y >= lp.shape[1])
    ):
        raise ValueError("Expected finite N x K logits and valid integer labels")
    pr = lp.argmax(1)
    return {
        "accuracy": float((pr == y).mean()),
        "error": float((pr != y).mean()),
        "balanced_accuracy": float(np.mean([(pr[y == v] == v).mean() for v in np.unique(y)])),
        "negative_log_likelihood": float(np.mean(logsumexp(lp, axis=1) - lp[np.arange(len(y)), y])),
    }
