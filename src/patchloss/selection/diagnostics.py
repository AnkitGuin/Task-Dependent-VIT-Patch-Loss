"""Task-dependent scores with explicit information, sample, and model assumptions."""

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from ..operators.patchify import patchify_np
from .linear import normalize_fit, logme, ridge_path

METRICS = (
    "rank",
    "reconstruction",
    "mean_tar",
    "covariance_tar",
    "crossfit_tar",
    "hscore",
    "logme",
    "cheap_probe",
)


def class_contrasts(samples, labels):
    samples = np.asarray(samples, dtype=np.float64)
    labels = np.asarray(labels)
    classes, indices = np.unique(labels, return_inverse=True)
    if len(classes) < 2 or len(samples) != len(labels) or not np.isfinite(samples).all():
        raise ValueError("Need finite samples and at least two classes")
    means = np.stack([samples[labels == c].mean(0) for c in classes])
    priors = np.bincount(indices) / len(labels)
    overall = np.tensordot(priors, means, axes=1)
    weights = np.sqrt(priors).reshape((-1,) + (1,) * (samples.ndim - 1))
    return (means - overall) * weights, means, indices, classes


def patch_statistics(operator_images, diagnostic_images, labels, p, folds, shrinkage):
    started = time.perf_counter()
    x = patchify_np(diagnostic_images, p)
    contrast, means, indices, classes = class_contrasts(x, labels)
    m = x.shape[-1]
    b = contrast.reshape(-1, m)
    first, _, _, classes_a = class_contrasts(x[folds["first"]], labels[folds["first"]])
    second, _, _, classes_b = class_contrasts(x[folds["second"]], labels[folds["second"]])
    if not np.array_equal(classes, classes_a) or not np.array_equal(classes, classes_b):
        raise ValueError("Every diagnostic fold must contain every class")
    cross_a, cross_b = first.reshape(-1, m), second.reshape(-1, m)
    residual = (x - means[indices]).reshape(-1, m)
    denominator = x.shape[1] * (len(x) - len(classes))
    if denominator <= 0 or not 0 < shrinkage <= 1:
        raise ValueError("Invalid covariance sample size or shrinkage")
    covariance = residual.T @ residual / denominator
    covariance = (1 - shrinkage) * covariance + shrinkage * np.trace(covariance) / m * np.eye(m)
    if np.trace(covariance) <= 0:
        raise ValueError("Within-class covariance is degenerate")
    raw_whitened = float(np.sum(b.T * cho_solve(cho_factor(covariance), b.T)))
    op = patchify_np(operator_images, p)
    centered = (op - op.mean(0)).reshape(-1, m)
    return {
        "contrast": b,
        "cross_a": cross_a,
        "cross_b": cross_b,
        "covariance": covariance,
        "raw_mean_energy": float(np.sum(b * b)),
        "raw_cross_energy": float(np.sum(cross_a * cross_b)),
        "raw_whitened_energy": raw_whitened,
        "operator_scatter": centered.T @ centered,
        "shared_statistics_sec": time.perf_counter() - started,
    }


def hscore(features, labels, shrinkage):
    x, _, _ = normalize_fit(features)
    contrast, _, _, _ = class_contrasts(x, labels)
    covariance = x.T @ x / len(x)
    covariance = (1 - shrinkage) * covariance + shrinkage * np.trace(covariance) / x.shape[1] * np.eye(
        x.shape[1]
    )
    return float(np.sum(contrast.T * cho_solve(cho_factor(covariance), contrast.T)))


def score_candidate(candidate, stats, features, labels, folds, config):
    W = candidate["matrix"]
    scores = {}
    times = {}
    start = time.perf_counter()
    _, singular, vt = np.linalg.svd(W, full_matrices=False)
    keep = singular > 1e-10 * singular[0]
    basis = vt[keep]
    scores["rank"] = int(keep.sum()) * candidate["token_count"]
    geometry_sec = time.perf_counter() - start
    times["rank"] = geometry_sec
    start = time.perf_counter()
    scatter = stats["operator_scatter"]
    scores["reconstruction"] = float(np.trace(basis @ scatter @ basis.T) / np.trace(scatter))
    times["reconstruction"] = time.perf_counter() - start + geometry_sec
    start = time.perf_counter()
    projected = stats["contrast"] @ basis.T
    den = stats["raw_mean_energy"]
    scores["mean_tar"] = float(np.sum(projected**2) / den) if den > 0 else None
    times["mean_tar"] = time.perf_counter() - start + geometry_sec
    start = time.perf_counter()
    den = stats["raw_cross_energy"]
    cross_numerator = float(np.sum((stats["cross_a"] @ basis.T) * (stats["cross_b"] @ basis.T)))
    scores["crossfit_tar"] = cross_numerator / den if den > 0 else None
    times["crossfit_tar"] = time.perf_counter() - start + geometry_sec
    start = time.perf_counter()
    covariance = W @ stats["covariance"] @ W.T
    b = stats["contrast"] @ W.T
    numerator = float(np.sum(b.T * cho_solve(cho_factor(covariance), b.T)))
    den = stats["raw_whitened_energy"]
    scores["covariance_tar"] = numerator / den if den > 0 else None
    times["covariance_tar"] = time.perf_counter() - start
    start = time.perf_counter()
    scores["hscore"] = hscore(features, labels, float(config["covariance_shrinkage"]))
    times["hscore"] = time.perf_counter() - start
    start = time.perf_counter()
    scores["logme"], evidence = logme(features, labels)
    times["logme"] = time.perf_counter() - start
    start = time.perf_counter()
    _, curve = ridge_path(
        features[folds["first"]],
        labels[folds["first"]],
        features[folds["second"]],
        labels[folds["second"]],
        [float(config["cheap_probe_alpha"])],
    )
    scores["cheap_probe"] = curve[0]["validation_accuracy"]
    times["cheap_probe"] = time.perf_counter() - start
    for name, value in scores.items():
        if value is not None and not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite diagnostic: {name}")
    return (
        scores,
        times,
        {
            "raw_mean_energy": stats["raw_mean_energy"],
            "raw_cross_energy": stats["raw_cross_energy"],
            "crossfit_projected_energy": cross_numerator,
            "logme_parameters": evidence,
            "geometry_sec": geometry_sec,
        },
    )
