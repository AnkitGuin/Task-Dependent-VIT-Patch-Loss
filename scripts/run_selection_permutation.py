"""Post-primary negative control: shuffle diagnostic labels, freeze choices, retain heads."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
import yaml
from patchloss.reporting.records import read_record
from patchloss.selection.data import load_cifar, diagnostic_folds, array_hash
from patchloss.selection.diagnostics import patch_statistics
from patchloss.selection.statistics import freeze_selection, evaluate_selection
from patchloss.seed import split_seed


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
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
                    originals = json.loads((directory / "pretest_selection.json").read_text())["selection"]
                    candidates = []
                    for row in primary["rows"]:
                        with np.load(directory / f"operator_{row['candidate']}.npz") as saved:
                            W = saved["matrix"]
                        _, _, vt = np.linalg.svd(W, full_matrices=False)
                        candidates.append(
                            {
                                "candidate": row["candidate"],
                                "patch_size": row["patch_size"],
                                "matrix": W,
                                "basis": vt,
                            }
                        )
                    frozen_records = []
                    for repetition in range(5):
                        permutation = np.random.default_rng(
                            split_seed(seed, dataset + f":permuted_labels:{repetition}")
                        ).permutation(len(labels))
                        target = labels[permutation]
                        folds = diagnostic_folds(
                            target, split_seed(seed, dataset + f":permutation_folds:{repetition}")
                        )
                        statistics = {
                            p: patch_statistics(
                                data["operator"][0], images, target, p, folds, config["covariance_shrinkage"]
                            )
                            for p in config["patch_sizes"]
                        }
                        scored = []
                        for candidate in candidates:
                            stats = statistics[candidate["patch_size"]]
                            W = candidate["matrix"]
                            b = stats["contrast"] @ W.T
                            covariance = W @ stats["covariance"] @ W.T
                            cov = float(
                                np.sum(b.T * cho_solve(cho_factor(covariance), b.T))
                                / stats["raw_whitened_energy"]
                            )
                            mean = float(
                                np.sum((stats["contrast"] @ candidate["basis"].T) ** 2)
                                / stats["raw_mean_energy"]
                            )
                            scored.append(
                                {
                                    "candidate": candidate["candidate"],
                                    "scores": {"mean_tar": mean, "covariance_tar": cov},
                                }
                            )
                        selected = freeze_selection(scored)
                        frozen = {metric: selected[metric] for metric in ["mean_tar", "covariance_tar"]}
                        frozen_records.append(
                            {
                                "repetition": repetition,
                                "permutation_indices": permutation.tolist(),
                                "permuted_labels_sha256": array_hash(target),
                                "scored": scored,
                                "selection": frozen,
                            }
                        )
                    (output / f"{directory.name}_pretest.json").write_text(
                        json.dumps(frozen_records, indent=2)
                    )
                    accuracy = {r["candidate"]: r["test_accuracy"] for r in primary["rows"]}
                    for record in frozen_records:
                        scored = record["scored"]
                        for row in scored:
                            row["test_accuracy"] = accuracy[row["candidate"]]
                        results = evaluate_selection(scored, record["selection"])
                        for result in results:
                            same = set(result["eligible_candidates"]) == set(
                                originals[result["metric"]]["eligible_candidates"]
                            )
                            outcomes.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "budget": budget,
                                    "repetition": record["repetition"],
                                    "same_choice_as_true_labels": same,
                                    **result,
                                }
                            )
                    print(f"LABEL-PERMUTATION CONTROL COMPLETE {directory.name}", flush=True)
    frame = pd.DataFrame(outcomes)
    frame.to_csv(output / "selection_by_unit.csv", index=False)
    frame.groupby("metric", as_index=False).agg(
        mean_regret=("regret", "mean"), same_choice_fraction=("same_choice_as_true_labels", "mean")
    ).to_csv(output / "summary.csv", index=False)
    metadata = {
        "status": "completed",
        "cohort": "post-primary diagnostic label-permutation negative control",
        "wallclock_sec": time.perf_counter() - started,
        "config": config,
        "protocol_sha256": hashlib.sha256(
            Path("docs/selection_permutation_protocol.md").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in output.iterdir()
            if f.is_file() and f.name != "metadata.json"
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("LABEL-PERMUTATION CONTROL COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--output", type=Path, default=Path("runs/selection_permutation"))
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.output)
