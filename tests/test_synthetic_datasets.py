import numpy as np
from patchloss.data.frequency import generate as freq
from patchloss.data.relational import generate as xor
from patchloss.data.alias_tradeoff import generate as lowpass


def test_deterministic_and_independent_splits():
    c = {"height": 16, "width": 16, "mode": "marker", "marker_size": 2}
    a, y, _ = freq(2, "train", 48, c)
    b, z, _ = freq(2, "train", 48, c)
    q, _, _ = freq(2, "test", 48, c)
    assert np.array_equal(a, b) and np.array_equal(y, z)
    assert not np.array_equal(a, q)


def test_xor_means_and_balance():
    x, y, _ = xor(11, "train", 2000, {"height": 16, "width": 16, "noise": 0.0})
    assert y.sum() == len(y) // 2
    assert np.allclose(x[y == 0].mean(axis=0), x[y == 1].mean(axis=0), atol=1e-5)


def test_label_preserving_translation():
    c = {"height": 16, "width": 16, "noise": 0.0, "marker_size": 2, "mode": "marker"}
    a, y, _ = freq(42, "test", 48, c)
    b, z, _ = freq(42, "test", 48, {**c, "phase_shift": 2})
    assert np.array_equal(y, z)
    assert np.array_equal(np.roll(a, 2, axis=-1), b)


def test_lowpass_family_class_balance():
    for family in ("signal", "nuisance"):
        x, y, _ = lowpass(2, "train", 48, {"height": 16, "width": 16, "family": family})
        assert y.sum() == 24
        assert np.all((x >= 0) & (x <= 1))
