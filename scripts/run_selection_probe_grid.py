"""Stronger exploratory joint candidate/penalty validation selector, without test labels."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml
from patchloss.reporting.records import read_record
from patchloss.selection.data import load_cifar, diagnostic_folds
from patchloss.selection.linear import ridge_path
from patchloss.selection.operators import transform
from patchloss.selection.statistics import freeze_selection
from patchloss.seed import split_seed

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0]


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    scores = []
    choices = []
    histories = []
    started = time.perf_counter()
    with threadpool_limits(config.get("threads", 2)):
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                data, _ = load_cifar(dataset, seed, config)
                images, labels = data["diagnostic"]
                folds = diagnostic_folds(labels, split_seed(seed, dataset + ":diagnostic_folds"))
                for budget in config["budgets"]:
                    directory = Path(config["output"]) / f"E11-{dataset}-b{budget}-s{seed}"
                    _, record = read_record(directory)
                    rows = []
                    for original in record["rows"]:
                        name = original["candidate"]
                        with np.load(directory / f"operator_{name}.npz") as saved:
                            candidate = {"matrix": saved["matrix"], "patch_size": original["patch_size"]}
                        tick = time.perf_counter()
                        features = transform(images, candidate)
                        model, curve = ridge_path(
                            features[folds["first"]],
                            labels[folds["first"]],
                            features[folds["second"]],
                            labels[folds["second"]],
                            ALPHAS,
                        )
                        score = max(item["validation_accuracy"] for item in curve)
                        rows.append({"candidate": name, "scores": {"cheap_probe": score}})
                        meta = {"dataset": dataset, "seed": seed, "budget": budget, "candidate": name}
                        scores.append(
                            {
                                **meta,
                                "score": score,
                                "selected_alpha": model["alpha"],
                                "seconds": time.perf_counter() - tick,
                            }
                        )
                        histories.extend({**meta, **item} for item in curve)
                    selected = freeze_selection(rows)["cheap_probe"]
                    pretest = {"rows": rows, "selection": selected, "alphas": ALPHAS}
                    (output / f"{directory.name}_pretest.json").write_text(json.dumps(pretest, indent=2))
                    choices.append({"dataset": dataset, "seed": seed, "budget": budget, **selected})
                    print(f"DIAGNOSTIC PROBE GRID FROZEN {directory.name}", flush=True)
    pd.DataFrame(scores).to_csv(output / "scores.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "validation_curves.csv", index=False)
    (output / "choices.json").write_text(json.dumps(choices, indent=2))
    metadata = {
        "status": "completed",
        "cohort": "exploratory joint candidate/penalty diagnostic validation baseline",
        "wallclock_sec": time.perf_counter() - started,
        "config": config,
        "alphas": ALPHAS,
        "protocol_sha256": hashlib.sha256(
            Path("docs/selection_probe_grid_protocol.md").read_bytes()
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
    parser.add_argument("--output", type=Path, default=Path("runs/selection_probe_grid"))
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.output)
