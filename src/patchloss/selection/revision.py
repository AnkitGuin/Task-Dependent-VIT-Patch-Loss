"""Auditable revision cohort; historical study and follow-ups stay immutable."""

import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .data import diagnostic_folds
from .diagnostics import patch_statistics, hscore
from .evidence import profile_logme, adaptive_hscore
from .linear import ridge_path, predict
from .operators import fit_bank, transform
from .statistics import evaluate_selection
from .study import source_snapshot
from ..seed import split_seed

METHODS = ("rank", "reconstruction", "mean_tar", "covariance_tar", "crossfit_tar", "hscore",
           "adaptive_hscore", "profile_logme", "cheap_probe", "tuned_probe", "fixed_dct")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze(rows, methods=METHODS):
    result = {}
    for method in methods:
        values = np.array([r["scores"].get(method, np.nan) for r in rows], dtype=float)
        undefined = not np.isfinite(values).all()
        selected = np.ones(len(rows), dtype=bool) if undefined else np.isclose(values, values.max(), rtol=1e-8, atol=1e-12)
        ids = [r["candidate"] for r, keep in zip(rows, selected) if keep]
        result[method] = dict(eligible_candidates=ids, undefined_fallback=undefined, realized_candidate=ids[0])
    return result


def score_bank(method, bank, data, config, folds, cache=None, shrinkage=None):
    """Return bank scores. An empty cache includes all required extraction/statistics costs."""
    cache = {} if cache is None else cache
    shrinkage = config["covariance_shrinkage"] if shrinkage is None else shrinkage
    values, details = {}, {}
    for candidate in bank:
        cid, p, W = candidate["candidate"], candidate["patch_size"], candidate["matrix"]
        detail = {}
        if method == "fixed_dct":
            value = float(cid == config["fixed_candidate"])
        elif method == "rank":
            value = int(np.linalg.matrix_rank(W)) * candidate["token_count"]
        elif method in {"reconstruction", "mean_tar", "covariance_tar", "crossfit_tar"}:
            key = ("stats", p, shrinkage)
            if key not in cache:
                cache[key] = patch_statistics(data["operator"][0], data["diagnostic"][0],
                                              data["diagnostic"][1], p, folds, shrinkage)
            stats = cache[key]
            if method == "covariance_tar":
                covariance = W @ stats["covariance"] @ W.T
                b = stats["contrast"] @ W.T
                value = float(np.sum(b.T * cho_solve(cho_factor(covariance), b.T)) / stats["raw_whitened_energy"])
            else:
                key = ("basis", cid)
                if key not in cache:
                    _, s, vt = np.linalg.svd(W, full_matrices=False)
                    cache[key] = vt[s > 1e-10 * s[0]]
                basis = cache[key]
                if method == "reconstruction":
                    value = float(np.trace(basis @ stats["operator_scatter"] @ basis.T) / np.trace(stats["operator_scatter"]))
                elif method == "mean_tar":
                    value = float(np.sum((stats["contrast"] @ basis.T)**2) / stats["raw_mean_energy"])
                else:
                    denominator = stats["raw_cross_energy"]
                    value = float(np.sum((stats["cross_a"] @ basis.T) * (stats["cross_b"] @ basis.T)) / denominator) if denominator > 0 else None
        else:
            key = ("features", cid)
            if key not in cache:
                cache[key] = transform(data["diagnostic"][0], candidate)
            features, labels = cache[key], data["diagnostic"][1]
            if method == "hscore":
                value = hscore(features, labels, config["covariance_shrinkage"])
            elif method == "adaptive_hscore":
                value, detail["shrinkage"] = adaptive_hscore(features, labels)
            elif method == "profile_logme":
                value, detail["parameters"] = profile_logme(features, labels)
            elif method in {"cheap_probe", "tuned_probe"}:
                alphas = config["ridge_alphas"] if method == "tuned_probe" else [config["cheap_probe_alpha"]]
                model, curve = ridge_path(features[folds["first"]], labels[folds["first"]],
                                          features[folds["second"]], labels[folds["second"]], alphas)
                value = max(r["validation_accuracy"] for r in curve)
                detail = dict(alpha=model["alpha"], validation_curve=curve)
            else:
                raise ValueError(method)
        if value is not None and not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite score: {method}/{cid}")
        values[cid], details[cid] = value, detail
    return values, details


def run_revision_unit(config, name, seed, budget, data, data_md):
    path = Path(config["output"]) / f"{name}-b{budget}-s{seed}"
    snapshot, source_hash = source_snapshot()
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    protocol_hash = hashlib.sha256(Path("docs/selection_revision_protocol.md").read_bytes()).hexdigest()
    if path.exists():
        meta = json.loads((path / "metadata.json").read_text())
        if (meta["source_sha256"], meta["config_sha256"], meta["protocol_sha256"]) != (source_hash, fingerprint, protocol_hash):
            raise ValueError("Revision cohort fingerprint mismatch; choose a new cohort")
        if meta["status"] != "completed":
            raise ValueError(f"Incomplete unit requires investigation: {path}")
        print(f"RESUME {path.name}", flush=True)
        return
    path.mkdir(parents=True)
    start = time.perf_counter()
    write_json(path / "source_snapshot.json", snapshot)
    write_json(path / "datasets.json", data_md)
    meta = dict(status="running", dataset=name, seed=seed, budget=budget, source_sha256=source_hash,
                config_sha256=fingerprint, protocol_sha256=protocol_hash)
    write_json(path / "metadata.json", meta)
    bank_start = time.perf_counter()
    bank = fit_bank(data["operator"][0], budget, config["patch_sizes"], seed, config)
    bank_sec = time.perf_counter() - bank_start
    folds = diagnostic_folds(data["diagnostic"][1], split_seed(seed, name + ":revision_folds"))
    write_json(path / "diagnostic_folds.json", {k: v.tolist() for k, v in folds.items()})
    rows = [{k: v for k, v in c.items() if k != "matrix"} | {"scores": {}, "details": {}} for c in bank]
    cache = {}
    for method in METHODS:
        values, details = score_bank(method, bank, data, config, folds, cache)
        for row in rows:
            row["scores"][method] = values[row["candidate"]]
            row["details"][method] = details[row["candidate"]]
    for shrinkage in config["shrinkage_sensitivity"]:
        if shrinkage == config["covariance_shrinkage"]:
            continue
        values, _ = score_bank("covariance_tar", bank, data, config, folds, cache, shrinkage)
        for row in rows:
            row["scores"][f"covariance_lambda_{shrinkage}"] = values[row["candidate"]]
    del cache
    x = np.concatenate([data[k][0] for k in ["operator", "diagnostic"]])
    y = np.concatenate([data[k][1] for k in ["operator", "diagnostic"]])
    models, history = {}, []
    for candidate, row in zip(bank, rows):
        cid = candidate["candidate"]
        begin = time.perf_counter()
        features = transform(x, candidate)
        validation = transform(data["validation"][0], candidate)
        row["head_feature_extraction_sec"] = time.perf_counter() - begin
        begin = time.perf_counter()
        model, curve = ridge_path(features, y, validation, data["validation"][1], config["ridge_alphas"])
        row.update(selected_alpha=model["alpha"], head_fit_sec=time.perf_counter() - begin)
        models[cid] = model
        np.savez_compressed(path / f"operator_{cid}.npz", matrix=candidate["matrix"])
        np.savez_compressed(path / f"head_{cid}.npz", **model)
        history.extend(dict(candidate=cid, **entry) for entry in curve)
    methods = list(rows[0]["scores"])
    frozen = freeze(rows, methods)
    within = {str(p): freeze([r for r in rows if r["patch_size"] == p],
                            [m for m in methods if m != "fixed_dct"])
              for p in config["patch_sizes"]}
    pretest = dict(rows=rows, selection=frozen, within_patch_selection=within, validation_curves=history,
                   bank_fit_sec=bank_sec, config=config)
    write_json(path / "pretest.json", pretest)
    pretest_hash = hashlib.sha256((path / "pretest.json").read_bytes()).hexdigest()
    predictions = {}
    test_x, test_y = data["test"]
    batch = config["test_batch_size"]
    for candidate, row in zip(bank, rows):
        cid = candidate["candidate"]
        pred = np.concatenate([predict(models[cid], transform(test_x[i:i+batch], candidate))
                               for i in range(0, len(test_x), batch)])
        predictions[cid] = pred
        row["test_accuracy"] = float(np.mean(pred == test_y))
    np.savez_compressed(path / "predictions.npz", labels=test_y, **predictions)
    if hashlib.sha256((path / "pretest.json").read_bytes()).hexdigest() != pretest_hash:
        raise AssertionError("Pretest artifact changed")
    outcomes = evaluate_selection(rows, frozen)
    within_outcomes = [dict(patch_size=p, **outcome) for p in config["patch_sizes"]
                       for outcome in evaluate_selection([r for r in rows if r["patch_size"] == p], within[str(p)])]
    write_json(path / "results.json", dict(rows=rows, selection=outcomes, within_patch=within_outcomes))
    meta.update(status="completed", wallclock_sec=time.perf_counter() - start, pretest_sha256=pretest_hash,
                bank_fit_sec=bank_sec, test_samples=len(test_y))
    meta["artifact_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in path.iterdir() if f.name != "metadata.json"}
    write_json(path / "metadata.json", meta)
    print(f"COMPLETE {path.name}: {len(bank)} heads, {len(test_y)} test images, {meta['wallclock_sec']:.1f}s", flush=True)
