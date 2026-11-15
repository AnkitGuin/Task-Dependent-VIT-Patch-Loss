"""Exploratory profile-evidence check after the primary short LogME iteration cap."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
import yaml
from patchloss.reporting.records import read_record
from patchloss.selection.data import load_cifar
from patchloss.selection.linear import normalize_fit
from patchloss.selection.operators import transform
from patchloss.selection.statistics import freeze_selection, evaluate_selection
from patchloss.seed import split_seed


def profile_objective(log_ratio, eigen, rhs, target_norm, n):
    ratio = np.exp(log_ratio)
    quadratic = target_norm - np.sum(rhs * rhs / (eigen + ratio))
    if quadratic <= 0:
        return -np.inf
    return float(
        -0.5 * (n * (np.log(2 * np.pi) + 1 + np.log(quadratic / n)) + np.log1p(eigen / ratio).sum()) / n
    )


def profiled_logme(features, labels, initial_parameters):
    x, _, _ = normalize_fit(features)
    eigen, vectors = eigh(x.T @ x, check_finite=False)
    eigen = np.maximum(eigen, 0.0)
    targets = (labels[:, None] == np.unique(labels)).astype(float)
    rhs = vectors.T @ x.T @ targets
    n = len(x)
    center = np.log(max(float(eigen.max()), 1e-12))
    grid = np.linspace(center - 20, center + 25, 61)
    values = []
    details = []
    for j, initial in enumerate(initial_parameters):
        b = rhs[:, j]
        target_norm = float(targets[:, j] @ targets[:, j])
        initial_ratio = np.log(initial["alpha"] / initial["beta"])

        def objective(log_ratio):
            return profile_objective(log_ratio, eigen, b, target_norm, n)

        grid_values = np.array([objective(t) for t in grid])
        best = float(-0.5 * (np.log(2 * np.pi) + 1 + np.log(target_norm / n)))
        mode = "null"
        best_ratio = None
        candidates = [(float(grid_values[i]), float(grid[i])) for i in np.argsort(grid_values)[-2:]]
        candidates.append((objective(initial_ratio), initial_ratio))
        for index in np.argsort(grid_values)[-2:]:
            left = grid[max(0, index - 1)]
            right = grid[min(len(grid) - 1, index + 1)]
            if right > left:
                fit = minimize_scalar(
                    lambda z: -objective(z), bounds=(left, right), method="bounded", options={"xatol": 1e-7}
                )
                candidates.append((-float(fit.fun), float(fit.x)))
        for score, ratio in candidates:
            if score > best:
                best = score
                best_ratio = ratio
                mode = "profile"
        values.append(best)
        details.append(
            {
                "evidence": best,
                "mode": mode,
                "log_ratio": best_ratio,
                "initial_profiled_evidence": objective(initial_ratio),
            }
        )
    return float(np.mean(values)), details


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    outcomes = []
    started = time.perf_counter()
    with threadpool_limits(config.get("threads", 2)):
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                data, _ = load_cifar(dataset, seed, config)
                images, labels = data["diagnostic"]
                for budget in config["budgets"]:
                    directory = Path(config["output"]) / f"E11-{dataset}-b{budget}-s{seed}"
                    _, primary = read_record(directory)
                    scored = []
                    for original in primary["rows"]:
                        name = original["candidate"]
                        with np.load(directory / f"operator_{name}.npz") as saved:
                            candidate = {"matrix": saved["matrix"], "patch_size": original["patch_size"]}
                        tick = time.perf_counter()
                        value, details = profiled_logme(
                            transform(images, candidate), labels, original["details"]["logme_parameters"]
                        )
                        if value < original["scores"]["logme"] - 1e-8:
                            raise AssertionError(
                                "Profile search returned lower evidence than the original hyperparameters"
                            )
                        scored.append({"candidate": name, "scores": {"logme": value}, "details": details})
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "budget": budget,
                                "candidate": name,
                                "original_logme": original["scores"]["logme"],
                                "profiled_logme": value,
                                "seconds": time.perf_counter() - tick,
                            }
                        )
                    frozen = freeze_selection(scored, split_seed(seed, dataset + f":profile_logme:{budget}"))[
                        "logme"
                    ]
                    pretest = {"scores": scored, "selection": frozen}
                    (output / f"{directory.name}_pretest.json").write_text(json.dumps(pretest, indent=2))
                    accuracies = {r["candidate"]: r["test_accuracy"] for r in primary["rows"]}
                    for row in scored:
                        row["test_accuracy"] = accuracies[row["candidate"]]
                    outcome = evaluate_selection(scored, {"logme": frozen})[0]
                    outcomes.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "budget": budget,
                            **outcome,
                            "metric": "profiled_logme",
                        }
                    )
                    print(f"PROFILE EVIDENCE CHECK COMPLETE {directory.name}", flush=True)
    pd.DataFrame(rows).to_csv(output / "scores.csv", index=False)
    pd.DataFrame(outcomes).to_csv(output / "selection_by_unit.csv", index=False)
    metadata = {
        "status": "completed",
        "cohort": "exploratory profile-evidence sensitivity",
        "config": config,
        "wallclock_sec": time.perf_counter() - started,
        "protocol_sha256": hashlib.sha256(
            Path("docs/selection_convergence_protocol.md").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in output.iterdir()
            if f.is_file() and f.name != "metadata.json"
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--output", type=Path, default=Path("runs/selection_logme_profile"))
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.output)
