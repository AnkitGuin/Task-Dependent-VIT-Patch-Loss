"""Exactly solved ridge heads and evidence objectives, without stochastic optimization."""

import numpy as np
from scipy.linalg import eigh


def normalize_fit(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0)
    centered = x - mean
    scale = float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(x).all() or scale <= 0:
        raise ValueError("Features must be finite and nonconstant")
    return centered / scale, mean, scale


def ridge_path(train_x, train_y, validation_x, validation_y, alphas):
    x, mean, scale = normalize_fit(train_x)
    classes = np.unique(train_y)
    targets = (np.asarray(train_y)[:, None] == classes).astype(float)
    prior = targets.mean(0)
    centered_targets = targets - prior
    dual = x.shape[1] > len(x)
    eigen, vectors = eigh(x @ x.T if dual else x.T @ x, check_finite=False)
    eigen = np.maximum(eigen, 0.0)
    rhs = vectors.T @ (centered_targets if dual else x.T @ centered_targets)
    xv = (validation_x - mean) / scale
    curve = []
    models = []
    for alpha in sorted(alphas):
        if alpha <= 0:
            raise ValueError("Ridge alpha must be positive")
        coef = vectors @ (rhs / (eigen[:, None] + alpha))
        if dual:
            coef = x.T @ coef
        accuracy = float(np.mean(classes[(xv @ coef + prior).argmax(1)] == validation_y))
        model = {
            "coef": coef,
            "mean": mean,
            "scale": scale,
            "prior": prior,
            "classes": classes,
            "alpha": float(alpha),
        }
        models.append(model)
        curve.append({"alpha": float(alpha), "validation_accuracy": accuracy})
    index = max(range(len(curve)), key=lambda i: (curve[i]["validation_accuracy"], curve[i]["alpha"]))
    return models[index], curve


def predict(model, features):
    logits = ((features - model["mean"]) / model["scale"]) @ model["coef"] + model["prior"]
    return model["classes"][logits.argmax(1)]


def log_evidence_from_spectrum(eigen, projected_rhs, target_norm, n, alpha, beta):
    """Integrated isotropic Gaussian-prior linear likelihood, normalized by n."""
    lam = alpha / beta
    quadratic = beta * (target_norm - np.sum(projected_rhs**2 / (eigen + lam)))
    return float(
        -0.5 * (n * np.log(2 * np.pi) - n * np.log(beta) + np.log1p(eigen / lam).sum() + quadratic) / n
    )


def logme(features, labels):
    """Published LogME objective with 11 fixed-point steps and ICML-style tolerance.

    This independent implementation uses a Gram eigensystem. Feature centering
    and a scalar RMS scale are the common declared probe preprocessing. One-hot
    targets are not centered. Evidence is recomputed exactly at the final alpha,
    beta rather than using a previous iteration's posterior-mode residual.
    """
    x, _, _ = normalize_fit(features)
    n = len(x)
    eigen, vectors = eigh(x.T @ x, check_finite=False)
    eigen = np.maximum(eigen, 0.0)
    classes = np.unique(labels)
    targets = (np.asarray(labels)[:, None] == classes).astype(float)
    rhs = vectors.T @ x.T @ targets
    values = []
    parameters = []
    for j in range(len(classes)):
        target_norm = float(targets[:, j] @ targets[:, j])
        b = rhs[:, j]
        alpha = beta = 1.0
        converged = False
        for iteration in range(11):
            lam = alpha / beta
            denominator = eigen + lam
            gamma = float(np.sum(eigen / denominator))
            weights_norm = float(np.sum((b / denominator) ** 2))
            residual = float(
                target_norm - 2 * np.sum(b * b / denominator) + np.sum(eigen * (b / denominator) ** 2)
            )
            alpha = max(gamma / (weights_norm + 1e-5), 1e-12)
            beta = max((n - gamma) / (max(0.0, residual) + 1e-5), 1e-12)
            if abs(alpha / beta - lam) / lam < 0.01:
                converged = True
                break
        value = log_evidence_from_spectrum(eigen, b, target_norm, n, alpha, beta)
        values.append(value)
        parameters.append({"alpha": alpha, "beta": beta, "iterations": iteration + 1, "converged": converged})
    return float(np.mean(values)), parameters
