"""Immutable per-unit records with scores/checkpoints frozen before test evaluation."""

import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
import torch
from ..config import config_hash
from ..seed import split_seed
from ..reporting.records import read_record
from .data import load_cifar, array_hash, diagnostic_folds
from .operators import fit_bank, transform
from .diagnostics import patch_statistics, score_candidate
from .linear import ridge_path, predict
from .statistics import freeze_selection, evaluate_selection


def source_snapshot():
    root = Path(__file__).resolve().parents[1]
    files = {
        str(f.relative_to(root)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(root.rglob("*.py"))
    }
    return files, hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_study(config):
    for key in ["operator_samples", "diagnostic_samples", "validation_samples", "test_samples"]:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not config["datasets"] or set(config["datasets"]) - {"cifar10", "cifar100"}:
        raise ValueError("Only declared CIFAR-10/100 data are supported")
    for name in config["datasets"]:
        k = 10 if name == "cifar10" else 100
        if any(config[key] % k for key in ["operator_samples", "validation_samples", "test_samples"]):
            raise ValueError("Sample counts must divide into balanced classes")
        if config["diagnostic_samples"] % (2 * k):
            raise ValueError("Each diagnostic fold must contain all balanced classes")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Need distinct study seeds")
    if any(not isinstance(s, int) or s < 0 for s in config["seeds"]):
        raise ValueError("Study seeds must be nonnegative integers")
    for budget in config["budgets"]:
        for p in config["patch_sizes"]:
            if p < 1 or 32 % p or budget % (1024 // p**2) or not 0 < budget <= 3072:
                raise ValueError("Invalid patch/budget grid")
    if not 0 < config["covariance_shrinkage"] <= 1:
        raise ValueError("Shrinkage must be in (0,1]")
    if not config["ridge_alphas"] or min(config["ridge_alphas"]) <= 0:
        raise ValueError("Ridge alphas must be positive")
    return config


def _raw_reference(data, config):
    x = np.concatenate([data["operator"][0], data["diagnostic"][0]])
    y = np.concatenate([data["operator"][1], data["diagnostic"][1]])
    start = time.perf_counter()
    model, curve = ridge_path(
        x.reshape(len(x), -1),
        y,
        data["validation"][0].reshape(len(data["validation"][0]), -1),
        data["validation"][1],
        config["ridge_alphas"],
    )
    return model, curve, time.perf_counter() - start


def run_unit(config, name, seed, budget, data, data_md, raw_reference):
    unit_config = {**config, "experiment": "E11", "dataset": name, "seed": seed, "budget": budget}
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    slug = f"E11-{name}-b{budget}-s{seed}"
    path = root / slug
    snapshot, source_hash = source_snapshot()
    if path.exists():
        md, metrics = read_record(path)
        if md["config_hash"] != config_hash(unit_config) or md["source_sha256"] != source_hash:
            raise ValueError(f"Refusing to resume incompatible unit {slug}; use a new output cohort")
        print(f"RESUME {slug}", flush=True)
        return path, metrics
    path.mkdir()
    (path / "config.yaml").write_text(yaml.safe_dump(unit_config, sort_keys=True))
    _json(path / "source_snapshot.json", snapshot)
    _json(path / "datasets.json", data_md)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    meta = {
        "run_id": slug,
        "experiment_id": "E11",
        "config_hash": config_hash(unit_config),
        "seed": seed,
        "source_sha256": source_hash,
        "status": "running",
        "dataset": name,
        "budget": budget,
        "device": "cpu",
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "dependencies": {k: version(k) for k in ["numpy", "scipy", "torch", "torchvision", "scikit-learn"]},
        "protocol_sha256": hashlib.sha256(Path("docs/selection_protocol.md").read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    _json(path / "metadata.json", meta)
    begin = time.perf_counter()
    try:
        operator_images = data["operator"][0]
        diagnostic_images, diagnostic_labels = data["diagnostic"]
        folds = diagnostic_folds(diagnostic_labels, split_seed(seed, name + ":diagnostic_folds"))
        _json(path / "diagnostic_folds.json", {k: v.tolist() for k, v in folds.items()})
        fit_images = np.concatenate([operator_images, diagnostic_images])
        fit_labels = np.concatenate([data["operator"][1], diagnostic_labels])
        bank_start = time.perf_counter()
        bank = fit_bank(operator_images, budget, config["patch_sizes"], seed, config)
        bank_sec = time.perf_counter() - bank_start
        statistics = {
            p: patch_statistics(
                operator_images,
                diagnostic_images,
                diagnostic_labels,
                p,
                folds,
                config["covariance_shrinkage"],
            )
            for p in config["patch_sizes"]
        }
        rows = []
        history = []
        models = {}
        for candidate in bank:
            cid = candidate["candidate"]
            start = time.perf_counter()
            fit_features = transform(fit_images, candidate)
            validation_features = transform(data["validation"][0], candidate)
            transform_sec = time.perf_counter() - start
            scores, times, details = score_candidate(
                candidate,
                statistics[candidate["patch_size"]],
                fit_features[len(operator_images) :],
                diagnostic_labels,
                folds,
                config,
            )
            start = time.perf_counter()
            model, curve = ridge_path(
                fit_features, fit_labels, validation_features, data["validation"][1], config["ridge_alphas"]
            )
            head_sec = time.perf_counter() - start
            models[cid] = model
            np.savez_compressed(path / f"operator_{cid}.npz", matrix=candidate["matrix"])
            np.savez_compressed(path / f"head_{cid}.npz", **model)
            row = {k: v for k, v in candidate.items() if k != "matrix"}
            row.update(
                {
                    "scores": scores,
                    "score_seconds": times,
                    "details": details,
                    "selected_alpha": model["alpha"],
                    "transform_train_validation_sec": transform_sec,
                    "head_fit_sec": head_sec,
                    "feature_train_sha256": array_hash(fit_features),
                    "head_parameters": int(model["coef"].size + len(model["prior"])),
                    "shared_patch_statistics_sec": statistics[candidate["patch_size"]][
                        "shared_statistics_sec"
                    ],
                }
            )
            rows.append(row)
            history.extend({"candidate": cid, **entry} for entry in curve)
            print(f"FIT {slug} {cid} val={max(r['validation_accuracy'] for r in curve):.3f}", flush=True)
        frozen = freeze_selection(rows, split_seed(seed, name + f":budget{budget}:ties"))
        within_frozen = {
            p: freeze_selection(
                [r for r in rows if r["patch_size"] == p],
                split_seed(seed, name + f":budget{budget}:patch{p}:ties"),
            )
            for p in config["patch_sizes"]
        }
        raw_model, raw_curve, raw_sec = raw_reference
        np.savez_compressed(path / "head_raw_reference.npz", **raw_model)
        pretest = {
            "rows": rows,
            "selection": frozen,
            "within_patch_selection": within_frozen,
            "raw_validation_curve": raw_curve,
            "raw_selected_alpha": raw_model["alpha"],
            "bank_fit_sec": bank_sec,
        }
        _json(path / "pretest_selection.json", pretest)
        pretest_hash = hashlib.sha256((path / "pretest_selection.json").read_bytes()).hexdigest()
        # First use of official-test labels in any score/fit/outcome computation.
        predictions = {}
        test_images, test_labels = data["test"]
        for candidate, row in zip(bank, rows):
            start = time.perf_counter()
            features = transform(test_images, candidate)
            predicted = predict(models[candidate["candidate"]], features)
            predictions[candidate["candidate"]] = predicted
            row["test_accuracy"] = float(np.mean(predicted == test_labels))
            row["test_transform_and_predict_sec"] = time.perf_counter() - start
        raw_pred = predict(raw_model, test_images.reshape(len(test_images), -1))
        np.savez_compressed(
            path / "predictions.npz", labels=test_labels, raw_reference=raw_pred, **predictions
        )
        if hashlib.sha256((path / "pretest_selection.json").read_bytes()).hexdigest() != pretest_hash:
            raise AssertionError("Pretest decisions were modified after seeing test labels")
        outcomes = evaluate_selection(rows, frozen)
        # A separate same-p summary matches dense projection dimensions and head size.
        within = []
        for p in config["patch_sizes"]:
            subset = [r for r in rows if r["patch_size"] == p]
            within.extend({"patch_size": p, **r} for r in evaluate_selection(subset, within_frozen[p]))
        elapsed = time.perf_counter() - begin
        metrics = {
            "status": "completed",
            "experiment_id": "E11",
            "seed": seed,
            "dataset": name,
            "budget": budget,
            "num_candidates": len(rows),
            "wallclock_sec": elapsed,
            "raw_reference_fit_sec": raw_sec,
            "raw_reference_accuracy": float(np.mean(raw_pred == test_labels)),
            "pretest_sha256": pretest_hash,
            "selection": outcomes,
            "within_patch_selection": within,
            "bank_fit_sec": bank_sec,
        }
        _json(path / "metrics.json", {"metrics": metrics, "rows": rows})
        with (path / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate", "alpha", "validation_accuracy"])
            writer.writeheader()
            writer.writerows(history)
        meta.update(
            {
                "status": "completed",
                "wallclock_sec": elapsed,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_sha256": {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in path.iterdir()
                    if f.is_file() and f.name != "metadata.json"
                },
            }
        )
        _json(path / "metadata.json", meta)
        read_record(path)
        print(f"COMPLETE {slug}: {len(rows)} candidates, {elapsed:.1f}s", flush=True)
        return path, {"metrics": metrics, "rows": rows}
    except Exception as ex:
        meta.update({"status": "failed", "wallclock_sec": time.perf_counter() - begin})
        _json(path / "failure.json", {"error": str(ex), "traceback": traceback.format_exc()})
        _json(path / "metadata.json", meta)
        raise


def run_study(config):
    validate_study(config)
    torch.set_num_threads(int(config.get("threads", 2)))
    outputs = []
    with threadpool_limits(limits=int(config.get("threads", 2))):
        for name in config["datasets"]:
            for seed in config["seeds"]:
                data, md = load_cifar(name, seed, config)
                raw = _raw_reference(data, config)
                for budget in config["budgets"]:
                    path, _ = run_unit(config, name, seed, budget, data, md, raw)
                    outputs.append(str(path))
    return outputs
