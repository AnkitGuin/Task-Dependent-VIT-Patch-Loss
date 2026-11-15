"""Offline checks of the separately labeled adaptive baseline and nested samples."""

import importlib.util
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from patchloss.selection.data import stratified_parts
from patchloss.selection.linear import normalize_fit


def load_script(name="run_selection_sensitivity.py"):
    spec = importlib.util.spec_from_file_location(
        "selection_sensitivity", Path(__file__).resolve().parents[1] / "scripts" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adaptive_hscore_matches_independent_full_matrix_expression():
    rng = np.random.default_rng(177)
    labels = np.arange(120) % 3
    features = rng.normal(size=(120, 9)) * np.arange(1, 10)
    features[:, 0] += labels
    value, alpha = load_script().adaptive_hscore(features, labels)
    x, _, _ = normalize_fit(features)
    covariance = LedoitWolf(assume_centered=True).fit(x)
    means = np.stack([x[labels == cls].mean(0) for cls in np.unique(labels)])
    between = means.T @ means / len(means)
    expected = (1 - covariance.shrinkage_) * np.trace(np.linalg.solve(covariance.covariance_, between))
    np.testing.assert_allclose(value, expected, rtol=1e-11)
    np.testing.assert_allclose(alpha, covariance.shrinkage_)


def test_diagnostic_sensitivity_samples_are_nested_and_class_balanced():
    labels = np.tile(np.arange(100), 10)
    small = stratified_parts(labels, {"sample": 200}, 71)["sample"]
    medium = stratified_parts(labels, {"sample": 400}, 71)["sample"]
    assert set(small).issubset(medium)
    np.testing.assert_array_equal(np.bincount(labels[small]), np.full(100, 2))
    np.testing.assert_array_equal(np.bincount(labels[medium]), np.full(100, 4))


def test_profile_evidence_matches_direct_marginal_likelihood_and_improves_start():
    from patchloss.selection.linear import logme

    rng = np.random.default_rng(411)
    features = rng.normal(size=(30, 7))
    labels = np.arange(30) % 3
    script = load_script("run_logme_convergence.py")
    x, _, _ = normalize_fit(features)
    eigen, vectors = np.linalg.eigh(x.T @ x)
    y = (labels == 0).astype(float)
    rhs = vectors.T @ x.T @ y
    ratio = 2.0
    q = y @ y - np.sum(rhs**2 / (eigen + ratio))
    beta = len(x) / q
    alpha = ratio * beta
    covariance = np.eye(len(x)) / beta + x @ x.T / alpha
    expected = (
        -0.5
        * (len(x) * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1] + y @ np.linalg.solve(covariance, y))
        / len(x)
    )
    actual = script.profile_objective(np.log(ratio), eigen, rhs, y @ y, len(x))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    old, parameters = logme(features, labels)
    new, _ = script.profiled_logme(features, labels, parameters)
    assert new >= old - 1e-10
