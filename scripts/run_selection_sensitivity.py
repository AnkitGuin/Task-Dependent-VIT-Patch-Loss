"""Exploratory diagnostic label-budget/adaptive-H-score sensitivity on frozen primary heads."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import ledoit_wolf
from threadpoolctl import threadpool_limits
import yaml
import pandas as pd
from patchloss.reporting.records import read_record
from patchloss.selection.data import load_cifar, stratified_parts, diagnostic_folds
from patchloss.selection.operators import transform
from patchloss.selection.linear import normalize_fit
from patchloss.selection.diagnostics import class_contrasts, patch_statistics, score_candidate
from patchloss.selection.statistics import freeze_selection, evaluate_selection
from patchloss.seed import split_seed


def adaptive_hscore(features, labels):
    normalized, _, _ = normalize_fit(features)
    covariance, alpha = ledoit_wolf(normalized, assume_centered=True)
    contrasts, _, _, _ = class_contrasts(normalized, labels)
    value = (1 - alpha) * np.sum(contrasts.T * cho_solve(cho_factor(covariance), contrasts.T))
    return float(value), float(alpha)


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    choices = []
    subsets = []
    started = time.perf_counter()
    with threadpool_limits(config.get("threads", 2)):
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                data, _ = load_cifar(dataset, seed, config)
                all_images, all_labels = data["diagnostic"]
                for budget in config["budgets"]:
                    unit = Path(config["output"]) / f"E11-{dataset}-b{budget}-s{seed}"
                    _, primary = read_record(unit)
                    # Test outcomes are joined only after each new score bank is frozen.
                    for count in [200, 400, 1000]:
                        if count > len(all_labels):
                            continue
                        ids = (
                            np.arange(len(all_labels))
                            if count == len(all_labels)
                            else stratified_parts(
                                all_labels,
                                {"sample": count},
                                split_seed(seed, dataset + ":sensitivity_subset"),
                            )["sample"]
                        )
                        images, labels = all_images[ids], all_labels[ids]
                        fold_key = ":diagnostic_folds" if count == 1000 else ":sensitivity_folds"
                        folds = diagnostic_folds(labels, split_seed(seed, dataset + fold_key))
                        subsets.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "budget": budget,
                                "diagnostic_samples": count,
                                "diagnostic_positions": ids.tolist(),
                                "fold_positions": {k: v.tolist() for k, v in folds.items()},
                            }
                        )
                        statistics = (
                            {
                                p: patch_statistics(
                                    data["operator"][0],
                                    images,
                                    labels,
                                    p,
                                    folds,
                                    config["covariance_shrinkage"],
                                )
                                for p in config["patch_sizes"]
                            }
                            if count < 1000
                            else {}
                        )
                        scored = []
                        for original in primary["rows"]:
                            cid = original["candidate"]
                            with np.load(unit / f"operator_{cid}.npz") as record:
                                candidate = {
                                    **{
                                        k: v
                                        for k, v in original.items()
                                        if k not in {"scores", "details", "score_seconds", "test_accuracy"}
                                    },
                                    "matrix": record["matrix"],
                                }
                            features = transform(images, candidate)
                            if count < 1000:
                                scores, timings, _ = score_candidate(
                                    candidate,
                                    statistics[candidate["patch_size"]],
                                    features,
                                    labels,
                                    folds,
                                    config,
                                )
                            else:
                                scores = dict(original["scores"])
                                timings = dict(original["score_seconds"])
                            tick = time.perf_counter()
                            adaptive, shrinkage = adaptive_hscore(features, labels)
                            scored.append(
                                {
                                    "candidate": cid,
                                    "scores": scores,
                                    "adaptive_hscore": adaptive,
                                    "adaptive_shrinkage": shrinkage,
                                    "adaptive_seconds": time.perf_counter() - tick,
                                    "score_seconds": timings,
                                }
                            )
                        frozen = freeze_selection(
                            scored, split_seed(seed, dataset + f":sensitivity:{budget}:{count}")
                        )
                        adaptive_rows = [
                            {
                                "candidate": r["candidate"],
                                "scores": {**r["scores"], "hscore": r["adaptive_hscore"]},
                            }
                            for r in scored
                        ]
                        adaptive_frozen = freeze_selection(
                            adaptive_rows,
                            split_seed(seed, dataset + f":sensitivity_adaptive:{budget}:{count}"),
                        )["hscore"]
                        pretest = {
                            "scores": scored,
                            "selection": frozen,
                            "adaptive_selection": adaptive_frozen,
                        }
                        prefix = f"{dataset}-b{budget}-s{seed}-n{count}"
                        path = output / (prefix + "_pretest.json")
                        path.write_text(json.dumps(pretest, indent=2, allow_nan=False))
                        # Scores/choices are immutable at this point; join frozen primary outcomes.
                        accuracy = {r["candidate"]: r["test_accuracy"] for r in primary["rows"]}
                        for row in scored:
                            row["test_accuracy"] = accuracy[row["candidate"]]
                        outcome = evaluate_selection(scored, frozen)
                        for row in adaptive_rows:
                            row["test_accuracy"] = accuracy[row["candidate"]]
                        adaptive_outcome = evaluate_selection(adaptive_rows, {"hscore": adaptive_frozen})[0]
                        adaptive_outcome["metric"] = "adaptive_hscore"
                        meta = {
                            "dataset": dataset,
                            "seed": seed,
                            "budget": budget,
                            "diagnostic_samples": count,
                        }
                        choices.extend({**meta, **r} for r in outcome + [adaptive_outcome])
                        rows.extend(
                            {
                                **meta,
                                "candidate": r["candidate"],
                                "adaptive_hscore": r["adaptive_hscore"],
                                "adaptive_shrinkage": r["adaptive_shrinkage"],
                                **{f"score_{k}": v for k, v in r["scores"].items()},
                            }
                            for r in scored
                        )
                        print(f"SENSITIVITY {prefix} frozen and evaluated", flush=True)
    pd.DataFrame(choices).to_csv(output / "selection_by_unit.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "scores.csv", index=False)
    (output / "subset_indices.json").write_text(json.dumps(subsets, indent=2))
    metadata = {
        "status": "completed",
        "cohort": "exploratory sensitivity, not primary",
        "wallclock_sec": time.perf_counter() - started,
        "protocol_sha256": hashlib.sha256(
            Path("docs/selection_secondary_protocol.md").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in output.iterdir()
            if f.is_file() and f.name != "metadata.json"
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("EXPLORATORY SENSITIVITY COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--output", type=Path, default=Path("runs/selection_sensitivity"))
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.output)
