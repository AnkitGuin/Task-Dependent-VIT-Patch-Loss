"""Independent checks for scientific errors found in the November 2026 review."""

import json
import numpy as np
import pytest
import torch
import pandas as pd
from scipy.stats import norm
from patchloss.config import validate, load_config
from patchloss.cli import config_files
from patchloss.data.gaussian import build_signal
from patchloss.data.natural import split_indices
from patchloss.data.relational import generate as xor
from patchloss.data.nullspace import generate as collision
from patchloss.operators.patchify import patchify_np
from patchloss.experiments.runner import run_experiment
from patchloss.operators.lowpass import BinomialLowPass
from patchloss.operators.overlap import OverlapTokenizer
from patchloss.operators.reference_matrix import materialize_operator
from patchloss.reporting.aggregate import aggregate
from patchloss.reporting.audit import audit
from patchloss.reporting.records import select_latest
from patchloss.theory import bayes_gaussian
from patchloss.theory.linear_algebra import numerical_rank, row_projector
from patchloss.theory.task_alignment import tar_mean
from patchloss.training.train import fit_model


@pytest.mark.parametrize("scale", [1e-40, 1.0, 1e40])
def test_geometry_and_token_oracle_are_scale_invariant(scale, monkeypatch):
    # Redundant, nonorthonormal rows; contrast in coordinate 2 is discarded.
    W = scale * np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    v = np.array([0.6, 0.8, 1.0])
    expected = norm.cdf(-1.0 / 0.8)
    assert numerical_rank(W) == 2
    np.testing.assert_allclose(row_projector(W), np.diag([1.0, 1.0, 0.0]), atol=1e-12)
    assert bayes_gaussian.bayes_error(v, W, 0.4) == pytest.approx(expected)

    # The MC validation must not be able to reuse the analytic projector.
    def forbidden(*args, **kwargs):
        raise AssertionError("MC must classify from tokens independently")

    monkeypatch.setattr(bayes_gaussian, "row_projector", forbidden)
    result = bayes_gaussian.independent_monte_carlo(W, v, 0.4, 40000, 17)
    assert abs(result["measured_error"] - expected) < 0.01
    assert result["ci95_low"] <= result["measured_error"] <= result["ci95_high"]


def test_tiny_signal_noise_scale_is_not_replaced_by_chance():
    # Small absolute signal can still have high signal/noise ratio.
    result = bayes_gaussian.independent_monte_carlo(np.eye(1), np.array([1e-20]), 1e-21, 1000, 9)
    assert result["measured_error"] < 0.01
    assert result["ci95_high"] > 0  # A zero-error observation must have nonzero uncertainty.


def test_zero_rank_and_endpoint_signal_controls():
    W = np.zeros((2, 3))
    assert bayes_gaussian.bayes_error(np.ones(3), W, 0.4) == 0.5
    result = bayes_gaussian.independent_monte_carlo(W, np.ones(3), 0.4, 20000, 8)
    assert result["ci95_low"] <= 0.5 <= result["ci95_high"]
    assert np.linalg.norm(build_signal(np.eye(3), 1.0, 0)) == pytest.approx(1.0)
    assert np.linalg.norm(build_signal(W, 0.0, 0)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        build_signal(np.eye(3), 0.5, 0)


def test_mean_tar_scale_invariance():
    x = np.zeros((20, 2, 3))
    y = np.arange(20) % 2
    x[y == 1, :, 0] = 1.0
    W = np.array([[1.0, 0.0, 0.0]])
    assert tar_mean(x * 1e-20, y, W) == pytest.approx(tar_mean(x, y, W))


def test_collision_mixture_has_a_genuine_zero_risk_raw_oracle():
    cfg = {"height": 16, "width": 16, "patch_size": 4, "embedding_dim": 8}
    x, y, md = collision(8, "test", 200, cfg)
    patches = patchify_np(x, 4)
    score = (patches[:, md["patch_index"]] - 0.5) @ md["null_direction"]
    np.testing.assert_array_equal(score > 0, y)
    np.testing.assert_allclose(np.abs(score), md["epsilon"], atol=1e-13)
    tokens = patches @ md["projection"].T
    np.testing.assert_allclose(tokens[::2], tokens[1::2], atol=1e-13)


@pytest.mark.parametrize("shift", [12, 13, -3, 31])
def test_xor_wrapped_translation_preserves_both_markers(shift):
    cfg = {"height": 16, "width": 16, "noise": 0.0}
    a, y, _ = xor(42, "test", 16, cfg)
    b, z, _ = xor(42, "test", 16, {**cfg, "phase_shift": shift})
    np.testing.assert_array_equal(y, z)
    np.testing.assert_array_equal(b, np.roll(a, shift, axis=-2))
    np.testing.assert_allclose(b[z == 0].mean(0), b[z == 1].mean(0))


def test_natural_partitions_are_disjoint_and_stable_when_sample_counts_change():
    train = split_indices(42, "train", 500, 1000, 200, 200)
    val = split_indices(42, "validation", 150, 1000, 200, 200)
    assert set(train).isdisjoint(val)
    np.testing.assert_array_equal(train[:20], split_indices(42, "train", 20, 1000, 200, 200))
    with pytest.raises(ValueError):
        split_indices(42, "validation", 201, 1000, 200, 200)


def test_lowpass_annihilates_unclipped_checkerboard_exactly():
    yy, xx = np.indices((16, 16))
    checker = torch.from_numpy((-1.0) ** (xx + yy))[None, None]
    low = BinomialLowPass().double()
    torch.testing.assert_close(low(checker), torch.zeros_like(checker), rtol=0, atol=0)
    # This certifies the linear filter, not Bayes risk of a clipped noisy generator.


def test_overlap_rejects_dropped_border_and_reference_handles_affine_offsets():
    op = OverlapTokenizer(p=4, stride=3, d=2)
    with pytest.raises(ValueError, match="border"):
        op(torch.zeros(1, 1, 8, 8))
    affine = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3)).double()
    affine.train()
    A = materialize_operator(affine, (1, 2, 2), dtype=torch.float64)
    np.testing.assert_allclose(A, affine[1].weight.detach().numpy(), atol=1e-14)
    assert affine.training


def test_declared_configs_validate_and_invalid_experiments_fail_before_running():
    for path in config_files("full"):
        validate(load_config(path))
    base = load_config("configs/tiny_vit.yaml")
    for changes in (
        {"heads": 0},
        {"channels": 3},
        {"tasks": ["typo"]},
        {"train_samples": 31},
        {"height": 16.5},
        {"lr": float("nan")},
        {"epochs": 0},
    ):
        with pytest.raises(ValueError):
            validate({**base, **changes})


def test_interval_membership_and_new_provenance(tmp_path):
    cfg = {
        "experiment": "E02",
        "height": 4,
        "width": 4,
        "patch_size": 4,
        "embedding_dim": 2,
        "train_samples": 32,
        "test_samples": 32,
        "mc_samples": 200,
        "probe_test_samples": 32,
        "ranks": [2],
        "alphas": [0.0, 1.0],
        "sigmas": [0.1],
    }
    path, _ = run_experiment(cfg, 0, tmp_path / "runs")
    data = json.loads((path / "metrics.json").read_text())
    for row in data["rows"]:
        assert row["theory_in_ci"] == (row["ci95_low"] <= row["theory_bayes_error"] <= row["ci95_high"])
    md = json.loads((path / "metadata.json").read_text())
    assert len(md["source_sha256"]) == 64
    assert "source_snapshot.json" in md["artifact_sha256"]


def test_corrupt_runs_are_excluded_and_audit_cannot_pass_empty_hashes(tmp_path):
    cfg = {"experiment": "E00", "height": 8, "width": 8, "patch_size": 2, "embedding_dim": 4}
    path, _ = run_experiment(cfg, 0, tmp_path / "runs")
    (path / "metrics.json").write_text("{}")
    assert aggregate(tmp_path / "runs", tmp_path / "reports")["runs"] == 0
    assert audit(tmp_path / "runs", tmp_path / "audit.md")["failures"]
    md = json.loads((path / "metadata.json").read_text())
    md["artifact_sha256"] = {}
    (path / "metadata.json").write_text(json.dumps(md))
    assert audit(tmp_path / "runs", tmp_path / "audit.md")["failures"]


@pytest.mark.parametrize("field", ["config_hash", "source_sha256"])
def test_reporting_refuses_to_blend_different_conditions(field):
    rows = [
        dict(experiment_id="E06", seed=i, run_id=f"run{i}", config_hash="a", source_sha256="v2")
        for i in range(2)
    ]
    rows[1][field] = "different"
    with pytest.raises(ValueError, match="Multiple configurations"):
        select_latest(pd.DataFrame(rows))


def test_nonfinite_training_is_an_explicit_failure():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight.fill_(float("nan"))
    x = np.zeros((8, 1, 2, 2), dtype=np.float32)
    y = np.arange(8) % 2
    with pytest.raises(FloatingPointError, match="Nonfinite training"):
        fit_model(model, (x, y), (x, y), (x, y), {"epochs": 1}, 0)
