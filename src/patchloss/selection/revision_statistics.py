"""Paired bootstrap preserving common test images, training seeds and budgets."""

import numpy as np


def paired_bootstrap(correct, draws=4000, seed=0, image_resampling=True):
    """Bootstrap accuracy for (seed, method, image) arrays, budgets already averaged.

    Each draw uses the SAME resampled images for every method and seed, then
    resamples training seeds. Domains are handled separately by the caller.
    Exact joint correctness patterns compress multinomial image resampling;
    no independent image noise is falsely assigned to overlapping seed runs.
    """
    correct = np.asarray(correct, dtype=np.float64)
    if correct.ndim != 3 or not np.isfinite(correct).all() or np.any((correct < 0) | (correct > 1)):
        raise ValueError("Expected finite correctness array in [0,1]")
    ns, nm, ni = correct.shape
    if min(ns, nm, ni) < 1:
        raise ValueError("Empty bootstrap axis")
    rng = np.random.default_rng(seed)
    sampled_seeds = rng.integers(0, ns, size=(draws, ns))
    seed_only = correct.mean(2)[sampled_seeds].mean(1)
    if not image_resampling:
        return seed_only, seed_only.copy()
    # Identical columns have identical contributions to every paired statistic.
    patterns, counts = np.unique(correct.reshape(ns * nm, ni).T, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    joint = np.empty((draws, nm))
    for start in range(0, draws, 100):
        end = min(draws, start + 100)
        weights = rng.multinomial(ni, probabilities, size=end - start)
        accuracies = (weights @ patterns / ni).reshape(end-start, ns, nm)
        indices = sampled_seeds[start:end]
        joint[start:end] = np.take_along_axis(accuracies, indices[:, :, None], axis=1).mean(1)
    return seed_only, joint
