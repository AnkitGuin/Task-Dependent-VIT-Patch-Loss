from patchloss.seed import split_seed
from patchloss.data.base import balanced_binary
import numpy as np


def test_seed_independence_and_balance():
    assert len({split_seed(3, s) for s in ["train", "test", "validation"]}) == 3
    assert not np.array_equal(
        np.random.default_rng(split_seed(1, "train")).random(10),
        np.random.default_rng(split_seed(2, "train")).random(10),
    )
    assert balanced_binary(20, np.random.default_rng(4)).sum() == 10
