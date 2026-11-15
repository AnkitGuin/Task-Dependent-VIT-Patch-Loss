import numpy as np
from patchloss.data.nullspace import generate
from patchloss.operators.patchify import patchify_np


def test_paired_exact_collision_and_pixel_bounds():
    cfg = {"height": 16, "width": 16, "patch_size": 4, "embedding_dim": 8}
    a, y, m = generate(0, "train", 32, cfg)
    assert a.shape == (32, 1, 16, 16)
    assert np.all(a >= 0) and np.all(a <= 1)
    assert np.linalg.norm(a[0::2] - a[1::2]) > 0
    z = patchify_np(a, 4) @ m["projection"].T
    assert np.max(abs(z[0::2] - z[1::2])) < 1e-9
    assert y.tolist() == [0, 1] * 16
    assert len(set(m["base_ids"])) == 16


def test_pair_splits_have_disjoint_templates():
    cfg = {"height": 16, "width": 16, "patch_size": 4, "embedding_dim": 8}
    _, _, a = generate(4, "train", 24, cfg)
    _, _, b = generate(4, "test", 24, cfg)
    assert set(a["base_ids"]).isdisjoint(b["base_ids"])
