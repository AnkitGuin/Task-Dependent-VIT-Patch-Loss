"""Selection endpoints; test outcomes cannot affect score-based candidate choices."""

import numpy as np
from scipy.stats import spearmanr
from .diagnostics import METRICS


def freeze_selection(rows, seed=0):
    rng = np.random.default_rng(seed)
    selected = {}
    for metric in METRICS:
        values = np.array(
            [
                r["scores"].get(metric, np.nan) if r["scores"].get(metric) is not None else np.nan
                for r in rows
            ],
            dtype=float,
        )
        if not np.isfinite(values).all():
            eligible = [r["candidate"] for r in rows]
            undefined = True
        else:
            mask = np.isclose(values, np.max(values), rtol=1e-8, atol=1e-12)
            eligible = [r["candidate"] for r, keep in zip(rows, mask) if keep]
            undefined = False
        selected[metric] = {
            "eligible_candidates": eligible,
            "undefined_fallback": undefined,
            "realized_candidate": str(rng.choice(eligible)),
        }
    return selected


def evaluate_selection(rows, frozen):
    accuracy = {r["candidate"]: r["test_accuracy"] for r in rows}
    best = max(accuracy.values())
    result = []
    for metric, selection in frozen.items():
        expected = float(np.mean([accuracy[name] for name in selection["eligible_candidates"]]))
        values = np.array(
            [r["scores"].get(metric) if r["scores"].get(metric) is not None else np.nan for r in rows],
            dtype=float,
        )
        outcomes = np.array([r["test_accuracy"] for r in rows])
        correlation = (
            float(spearmanr(values, outcomes).statistic)
            if np.isfinite(values).all() and np.ptp(values) > 1e-12 and np.ptp(outcomes) > 0
            else None
        )
        result.append(
            {
                "metric": metric,
                "selected_accuracy": expected,
                "oracle_accuracy": best,
                "regret": float(best - expected),
                "spearman": correlation,
                "tie_count": len(selection["eligible_candidates"]),
                "undefined_fallback": selection["undefined_fallback"],
                "eligible_candidates": selection["eligible_candidates"],
                "realized_candidate": selection["realized_candidate"],
                "realized_accuracy": accuracy[selection["realized_candidate"]],
            }
        )
    return result


def block_bootstrap(frame, field, draws=2000, seed=0):
    """Conditional on the observed datasets, resample paired seed blocks within each."""
    rng = np.random.default_rng(seed)
    metrics = sorted(frame.metric.unique())
    datasets = sorted(frame.dataset.unique())
    means = np.zeros((draws, len(metrics)))
    for dataset in datasets:
        part = frame[frame.dataset == dataset]
        seeds = sorted(part.seed.unique())
        cube = np.array(
            [
                [part[(part.seed == s) & (part.metric == metric)][field].mean() for metric in metrics]
                for s in seeds
            ]
        )
        samples = rng.integers(0, len(seeds), size=(draws, len(seeds)))
        means += cube[samples].mean(1) / len(datasets)
    return metrics, means
