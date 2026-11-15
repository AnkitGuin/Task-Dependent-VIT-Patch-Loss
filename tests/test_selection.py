"""Independent numerical and leakage controls for the prospective selection study."""

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from patchloss.selection.data import stratified_parts, diagnostic_folds
from patchloss.selection.operators import dct_rows, fit_bank, transform
from patchloss.selection.linear import ridge_path, predict, normalize_fit, logme
from patchloss.selection.diagnostics import patch_statistics, METRICS
from patchloss.selection.statistics import freeze_selection, evaluate_selection
from patchloss.selection.study import run_unit
from patchloss.reporting.records import read_record
from patchloss.operators.patchify import patchify_np


def test_stratified_partitions_and_folds_are_disjoint_balanced_reproducible():
    labels = np.tile(np.arange(5), 100)
    parts = stratified_parts(labels, {"operator": 50, "diagnostic": 100, "validation": 50}, 7)
    ids = np.concatenate(list(parts.values()))
    assert len(np.unique(ids)) == len(ids)
    for part in parts.values():
        assert len(set(np.bincount(labels[part]))) == 1
    for key, value in parts.items():
        np.testing.assert_array_equal(
            value, stratified_parts(labels, {"operator": 50, "diagnostic": 100, "validation": 50}, 7)[key]
        )
    folds = diagnostic_folds(labels[parts["diagnostic"]], 12)
    assert set(folds["first"]).isdisjoint(folds["second"])
    with pytest.raises(ValueError):
        stratified_parts(labels, {"bad": 501}, 1)


@pytest.mark.parametrize("p,channels,d", [(2, 3, 3), (4, 1, 8), (4, 3, 48)])
def test_dct_orthonormality_and_lossless_complete_basis(p, channels, d):
    W = dct_rows(p, channels, d)
    np.testing.assert_allclose(W @ W.T, np.eye(d), atol=1e-12)
    if d == channels * p * p:
        np.testing.assert_allclose(W.T @ W, np.eye(d), atol=1e-12)
    if channels == 3:
        # The lowest color component mixes RGB equally rather than dropping two colors.
        np.testing.assert_allclose(W[0].reshape(channels, p, p)[0], W[0].reshape(channels, p, p)[2])


@pytest.mark.parametrize("n,d", [(40, 8), (12, 30)])
def test_ridge_primal_and_dual_match_independent_augmented_least_squares(n, d):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, d))
    y = np.arange(n) % 3
    val = rng.normal(size=(13, d))
    vy = np.arange(13) % 3
    model, _ = ridge_path(x, y, val, vy, [2.0])
    normalized, mean, scale = normalize_fit(x)
    target = np.eye(3)[y]
    prior = target.mean(0)
    design = np.vstack([normalized, np.sqrt(2) * np.eye(d)])
    response = np.vstack([target - prior, np.zeros((d, 3))])
    expected = np.linalg.lstsq(design, response, rcond=None)[0]
    np.testing.assert_allclose(model["coef"], expected, atol=1e-11)
    rotation = np.linalg.qr(rng.normal(size=(d, d)))[0]
    rotated, _ = ridge_path(x @ rotation, y, val @ rotation, vy, [2.0])
    np.testing.assert_array_equal(predict(model, val), predict(rotated, val @ rotation))


def test_logme_objective_matches_direct_integrated_gaussian_likelihood():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(30, 7))
    labels = np.arange(30) % 3
    value, parameters = logme(features, labels)
    x, _, _ = normalize_fit(features)
    direct = []
    for cls, param in enumerate(parameters):
        covariance = np.eye(len(x)) / param["beta"] + x @ x.T / param["alpha"]
        sign, logdet = np.linalg.slogdet(covariance)
        assert sign == 1
        y = (labels == cls).astype(float)
        direct.append(
            -0.5 * (len(x) * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(covariance, y)) / len(x)
        )
    assert value == pytest.approx(np.mean(direct), abs=1e-11)


def test_covariance_retention_distinguishes_equal_euclidean_alignment():
    rng = np.random.default_rng(42)
    labels = np.arange(4000) % 2
    contrast = np.array([1.0, 1.0]) / np.sqrt(2)
    covariance = np.diag([0.01, 10.0])
    x = (2 * labels - 1)[:, None] * contrast / 2 + rng.normal(size=(4000, 2)) @ np.linalg.cholesky(
        covariance
    ).T
    # Explicit whitening calculation: projected Mahalanobis signal fractions differ.
    full = contrast @ np.linalg.solve(covariance, contrast)
    retention = [contrast[i] ** 2 / covariance[i, i] / full for i in [0, 1]]
    assert retention[0] > 0.99 and retention[1] < 0.01
    errors = [norm.cdf(-0.5 * abs(contrast[i]) / np.sqrt(covariance[i, i])) for i in [0, 1]]
    assert errors[0] < 0.001 and errors[1] > 0.4
    assert x.shape == (4000, 2)


def test_crossfit_energies_are_not_forced_nonnegative():
    rng = np.random.default_rng(22)
    operator = rng.normal(size=(20, 1, 4, 4))
    diagnostic = np.zeros((20, 1, 4, 4))
    labels = np.arange(20) % 2
    folds = diagnostic_folds(labels, 2)
    diagnostic[folds["first"], 0, 0, 0] = 2 * labels[folds["first"]] - 1
    diagnostic[folds["second"], 0, 0, 0] = -(2 * labels[folds["second"]] - 1)
    diagnostic += rng.normal(0, 0.01, diagnostic.shape)
    stats = patch_statistics(operator, diagnostic, labels, 2, folds, 0.1)
    assert stats["raw_cross_energy"] < 0


def test_selection_uses_scores_only_and_ties_have_uniform_expected_performance():
    rows = [
        {"candidate": "a", "scores": {name: 1.0 for name in METRICS}, "test_accuracy": 0.2},
        {"candidate": "b", "scores": {name: 1.0 for name in METRICS}, "test_accuracy": 0.8},
    ]
    frozen = freeze_selection(rows)
    outcomes = evaluate_selection(rows, frozen)
    assert all(o["regret"] == pytest.approx(0.3) and o["spearman"] is None for o in outcomes)
    rows[0]["test_accuracy"] = 1.0
    assert freeze_selection(rows) == frozen
    rows[0]["scores"]["crossfit_tar"] = None
    assert freeze_selection(rows)["crossfit_tar"]["undefined_fallback"]


def test_small_bank_has_matched_budget_and_real_matrix_predictions():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 1, 8, 8))
    bank = fit_bank(x, 16, [2, 4], 0, {"pca_patch_cap": 100, "pretrained": False})
    assert len(bank) == 6
    for candidate in bank:
        assert transform(x, candidate).shape == (20, 16)
        patches = patchify_np(x, candidate["patch_size"])
        expected = patches.reshape(20, -1) @ block_diag(*([candidate["matrix"]] * candidate["token_count"])).T
        np.testing.assert_allclose(transform(x, candidate), expected, atol=1e-12)


def test_tiny_unit_saves_pretest_choices_and_hash_valid_checkpoints(tmp_path):
    rng = np.random.default_rng(9)
    data = {}
    for name, n in [("operator", 20), ("diagnostic", 40), ("validation", 20), ("test", 20)]:
        y = np.arange(n) % 2
        x = rng.normal(size=(n, 1, 8, 8))
        x[:, 0, 0, 0] += 2 * y - 1
        data[name] = (x, y)
    config = {
        "output": str(tmp_path / "runs"),
        "patch_sizes": [2, 4],
        "pca_patch_cap": 100,
        "pretrained": False,
        "ridge_alphas": [1.0, 10.0],
        "covariance_shrinkage": 0.1,
        "cheap_probe_alpha": 10.0,
    }
    raw_x = np.concatenate([data["operator"][0], data["diagnostic"][0]])
    raw_y = np.concatenate([data["operator"][1], data["diagnostic"][1]])
    raw_model, raw_curve = ridge_path(
        raw_x.reshape(len(raw_x), -1),
        raw_y,
        data["validation"][0].reshape(20, -1),
        data["validation"][1],
        [1.0, 10.0],
    )
    with threadpool_limits(2):
        path, result = run_unit(
            config, "test_fixture", 0, 16, data, {"fixture": True}, (raw_model, raw_curve, 0.0)
        )
    md, saved = read_record(path)
    assert md["status"] == "completed" and saved == result
    assert len(result["rows"]) == 6
    assert "pretest_selection.json" in md["artifact_sha256"]
    assert (path / "predictions.npz").exists()
