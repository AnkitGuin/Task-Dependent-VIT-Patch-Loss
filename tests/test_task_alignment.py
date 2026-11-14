import numpy as np
from patchloss.theory.task_alignment import tar_mean


def test_tar_bounds_and_undefined():
    x = np.zeros((20, 2, 4))
    y = np.array([0, 1] * 10)
    W = np.array([[1.0, 0, 0, 0]])
    assert tar_mean(x, y, W) is None
    x[y == 1, :, 0] = 1
    assert np.isclose(tar_mean(x, y, W), 1)
    W = np.array([[0.0, 1, 0, 0]])
    assert np.isclose(tar_mean(x, y, W), 0)
