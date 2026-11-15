"""Independent numerical checks for the publication revision's material changes."""

import numpy as np
from scipy.optimize import differential_evolution
from threadpoolctl import threadpool_limits
from patchloss.selection.revision_data import clean_training_pool
from patchloss.selection.evidence import profile_logme
from patchloss.selection.linear import normalize_fit
from patchloss.selection.revision_statistics import paired_bootstrap
from patchloss.selection.revision import freeze


def test_content_cleaning_removes_cross_split_repeats_and_label_conflicts():
    images = np.array([0, 0, 1, 2, 2, 3, 4], dtype=np.uint8).reshape(-1, 1, 1, 1)
    labels = np.array([0, 0, 1, 0, 1, 1, 0])
    test = np.array([1, 1, 9], dtype=np.uint8).reshape(-1, 1, 1, 1)
    keep, train_hash, test_hash, audit = clean_training_pool(images, labels, test)
    np.testing.assert_array_equal(keep, [0, 5, 6])
    assert not set(train_hash[keep]) & set(test_hash)
    assert audit["training_rows_matching_test_content_removed"] == 1
    assert audit["training_rows_in_conflicting_label_groups_removed"] == 2
    assert audit["repeated_training_rows_removed"] == 1
    assert audit["repeated_official_test_rows_retained"] == 1


def test_profile_evidence_matches_independent_direct_gaussian_optimization():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(35, 5))
    labels = (features[:, 0] + .2 * rng.normal(size=35) > 0).astype(int)
    with threadpool_limits(1):
        value, parameters = profile_logme(features, labels)
        x, _, _ = normalize_fit(features)
        reference = []
        for cls in [0, 1]:
            y = (labels == cls).astype(float)
            def loss(log_parameters):
                alpha, beta = np.exp(log_parameters)
                covariance = np.eye(len(x)) / beta + x @ x.T / alpha
                sign, logdet = np.linalg.slogdet(covariance)
                assert sign > 0
                return .5 * (len(x) * np.log(2*np.pi) + logdet + y @ np.linalg.solve(covariance, y)) / len(x)
            result = differential_evolution(loss, [(-12, 15), (-12, 15)], seed=3, tol=1e-9)
            reference.append(-result.fun)
        np.testing.assert_allclose(value, np.mean(reference), atol=1e-7)
        assert all(p["converged"] for p in parameters)


def test_bootstrap_uses_common_images_and_seeds_for_paired_methods():
    rng = np.random.default_rng(2)
    correct = rng.integers(0, 2, size=(3, 1, 40)).astype(float)
    correct = np.concatenate([correct, correct, 1-correct], axis=1)
    seed_only, joint = paired_bootstrap(correct, draws=300, seed=9)
    np.testing.assert_array_equal(joint[:, 0], joint[:, 1])
    np.testing.assert_allclose(joint[:, 0] + joint[:, 2], 1)
    assert np.var(joint[:, 0]) > np.var(seed_only[:, 0])
    duplicate_seeds = np.repeat(correct[:1], 5, axis=0)
    _, joint = paired_bootstrap(duplicate_seeds, draws=300, seed=9)
    # Repeating one fitted model must not divide independent image variance by 5.
    p = correct[0, 0].mean()
    np.testing.assert_allclose(joint[:, 0].var(), p*(1-p)/40, rtol=.3)


def test_fixed_policy_and_tie_expectation_do_not_depend_on_outcomes():
    rows = [dict(candidate=c, scores=dict(rank=256, fixed_dct=float(c == "p04_dct")))
            for c in ["p04_dct", "p16_pca"]]
    frozen = freeze(rows, ["rank", "fixed_dct"])
    assert frozen["fixed_dct"]["eligible_candidates"] == ["p04_dct"]
    assert len(frozen["rank"]["eligible_candidates"]) == 2
    for row in rows:
        row["test_accuracy"] = np.random.random()
    assert freeze(rows, ["rank", "fixed_dct"]) == frozen
