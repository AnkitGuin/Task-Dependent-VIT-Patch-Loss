"""Expanded, validation-only ridge grid with primary operators and selector choices fixed."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml
from patchloss.reporting.records import read_record
from patchloss.selection.data import load_cifar
from patchloss.selection.linear import ridge_path, predict
from patchloss.selection.operators import transform
from patchloss.selection.statistics import evaluate_selection

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0]


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_candidates = []
    started = time.perf_counter()
    with threadpool_limits(config.get("threads", 2)):
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                data, data_md = load_cifar(dataset, seed, config)
                fit_images = np.concatenate([data["operator"][0], data["diagnostic"][0]])
                fit_labels = np.concatenate([data["operator"][1], data["diagnostic"][1]])
                xv, yv = data["validation"]
                raw_model, raw_curve = ridge_path(
                    fit_images.reshape(len(fit_images), -1), fit_labels, xv.reshape(len(xv), -1), yv, ALPHAS
                )
                for budget in config["budgets"]:
                    original = Path(config["output"]) / f"E11-{dataset}-b{budget}-s{seed}"
                    primary_md, primary = read_record(original)
                    pretest = json.loads((original / "pretest_selection.json").read_text())
                    directory = output / original.name
                    if directory.exists():
                        raise RuntimeError("Use a fresh immutable follow-up directory")
                    directory.mkdir()
                    models = {}
                    candidates = {}
                    curves = []
                    rows = []
                    for original_row in primary["rows"]:
                        name = original_row["candidate"]
                        with np.load(original / f"operator_{name}.npz") as saved:
                            candidate = {"matrix": saved["matrix"], "patch_size": original_row["patch_size"]}
                        features = transform(fit_images, candidate)
                        validation = transform(xv, candidate)
                        model, curve = ridge_path(features, fit_labels, validation, yv, ALPHAS)
                        models[name] = model
                        candidates[name] = candidate
                        np.savez_compressed(directory / f"head_{name}.npz", **model)
                        curves.extend({"candidate": name, **item} for item in curve)
                        rows.append(
                            {
                                "candidate": name,
                                "scores": original_row["scores"],
                                "selected_alpha": model["alpha"],
                                "best_validation_accuracy": max(r["validation_accuracy"] for r in curve),
                                "upper_grid_accuracy_plateau": len(
                                    {r["validation_accuracy"] for r in curve[-3:]}
                                )
                                == 1,
                            }
                        )
                    np.savez_compressed(directory / "head_raw_reference.npz", **raw_model)
                    frozen = {
                        "rows": rows,
                        "selection": pretest["selection"],
                        "raw_curve": raw_curve,
                        "raw_selected_alpha": raw_model["alpha"],
                        "expanded_alphas": ALPHAS,
                    }
                    (directory / "pretest_selection.json").write_text(json.dumps(frozen, indent=2))
                    test_images, test_labels = data["test"]
                    predictions = {}
                    for row in rows:
                        name = row["candidate"]
                        predicted = predict(models[name], transform(test_images, candidates[name]))
                        predictions[name] = predicted
                        row["test_accuracy"] = float(np.mean(predicted == test_labels))
                    raw_prediction = predict(raw_model, test_images.reshape(len(test_images), -1))
                    np.savez_compressed(
                        directory / "predictions.npz",
                        labels=test_labels,
                        raw_reference=raw_prediction,
                        **predictions,
                    )
                    results = evaluate_selection(rows, pretest["selection"])
                    metadata = {
                        "dataset": dataset,
                        "seed": seed,
                        "budget": budget,
                        "status": "completed",
                        "primary_source_sha256": primary_md["source_sha256"],
                        "primary_metrics_sha256": hashlib.sha256(
                            (original / "metrics.json").read_bytes()
                        ).hexdigest(),
                        "datasets": data_md,
                        "raw_reference_accuracy": float(np.mean(raw_prediction == test_labels)),
                    }
                    (directory / "metrics.json").write_text(
                        json.dumps({"rows": rows, "selection": results}, indent=2)
                    )
                    with (directory / "history.csv").open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["candidate", "alpha", "validation_accuracy"])
                        writer.writeheader()
                        writer.writerows(curves)
                    metadata["artifact_sha256"] = {
                        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in directory.iterdir()
                        if f.is_file() and f.name != "metadata.json"
                    }
                    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
                    meta = {"dataset": dataset, "seed": seed, "budget": budget}
                    all_results.extend({**meta, **r} for r in results)
                    all_candidates.extend({**meta, **r} for r in rows)
                    print(f"EXPANDED VALIDATION GRID COMPLETE {original.name}", flush=True)
    frame = pd.DataFrame(all_results)
    cells = pd.DataFrame(all_candidates)
    frame.to_csv(output / "selection_by_unit.csv", index=False)
    cells.to_csv(output / "candidates.csv", index=False)
    frame.groupby("metric", as_index=False).regret.mean().sort_values("regret").to_csv(
        output / "summary.csv", index=False
    )
    metadata = {
        "status": "completed",
        "cohort": "validation-driven expanded-grid follow-up",
        "wallclock_sec": time.perf_counter() - started,
        "config": config,
        "alphas": ALPHAS,
        "protocol_sha256": hashlib.sha256(
            Path("docs/selection_regularization_protocol.md").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "root_artifact_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in output.iterdir()
            if f.is_file() and f.name != "metadata.json"
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--output", type=Path, default=Path("runs/selection_regularization"))
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.output)
