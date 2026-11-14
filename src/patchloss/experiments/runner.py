"""Auditable experiments E00–E10; no metrics are generated without execution."""

from __future__ import annotations
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from ..config import config_hash, validate
from ..data import nullspace, gaussian as gaussian_data, frequency, relational, alias_tradeoff, natural
from ..operators.patchify import patchify, unpatchify, patchify_np
from ..operators.linear_patch import random_projection
from ..operators.overlap import OverlapTokenizer
from ..operators.lowpass import BinomialLowPass
from ..operators.reference_matrix import materialize_operator
from ..theory.linear_algebra import numerical_rank, alignment
from ..theory.bayes_gaussian import independent_monte_carlo, bayes_error, raw_error
from ..theory.task_alignment import tar_mean
from ..models.probes import probe
from ..models.tiny_vit import TinyViT
from ..models.small_cnn import SmallCNN
from ..training.train import fit_model
from ..training.checkpoint import save_checkpoint
from ..training.metrics import transformer_flops_proxy
from ..seed import seed_all


def _csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _features(x, W, p):
    return (patchify_np(x.astype(np.float64), p) @ W.T).reshape(len(x), -1)


def _dataset(generator, seed, split, count, config, path, condition=""):
    x, y, md = generator(seed, split, count, config)

    def digest(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()

    record = {
        "task": md["task"],
        "condition": condition,
        "split": split,
        "seed": seed,
        "split_seed": md["split_seed"],
        "parameter_hash": md["parameter_hash"],
        "generator_version": md["generator_version"],
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "images_sha256": digest(x),
        "labels_sha256": digest(y),
        "class_counts": np.bincount(y).tolist(),
    }
    if "source_indices" in md:
        record.update({k: md[k] for k in ("source_indices", "official_partition")})
    target = path / "datasets.json"
    records = json.loads(target.read_text()) if target.exists() else []
    target.write_text(json.dumps(records + [record], indent=2))
    return x, y, md


def e00(c, seed, path):
    rows = []
    for p in (1, 2, 4, 8, 16):
        h, w = int(c.get("height", 32)), int(c.get("width", 32))
        if h % p or w % p:
            continue
        gen = np.random.default_rng(seed + p)
        channels = int(c.get("channels", 1))
        x = torch.from_numpy(gen.random((3, channels, h, w)))
        z = patchify(x, p)
        xr = unpatchify(z, p, h, w)
        rows.append(
            {
                "patch_size": p,
                "max_abs_roundtrip": float(torch.max(torch.abs(x - xr))),
                "norm_error": float(
                    torch.max(
                        torch.abs(
                            torch.linalg.vector_norm(x.flatten(1), dim=1)
                            - torch.linalg.vector_norm(z.flatten(1), dim=1)
                        )
                    )
                ),
                "token_count": z.shape[1],
                "token_budget": z.shape[1] * z.shape[2],
                "operator_rank": channels * h * w,
            }
        )
    return (
        {
            "max_abs_roundtrip": max(r["max_abs_roundtrip"] for r in rows),
            "lossless": all(r["max_abs_roundtrip"] == 0 for r in rows),
        },
        rows,
        [],
    )


def e01(c, seed, path):
    n = int(c.get("train_samples", 128))
    v = int(c.get("val_samples", 64))
    t = int(c.get("test_samples", 128))
    datasets = [
        _dataset(nullspace.generate, seed, split, nx, c, path)
        for split, nx in [("train", n), ("validation", v), ("test", t)]
    ]
    x, y, md = datasets[2]
    W = md["projection"]
    p = int(c.get("patch_size", 8))
    f = _features(x, W, p)
    raw_scores = (patchify_np(x, p)[:, md["patch_index"], :] - 0.5) @ md["null_direction"]
    raw_oracle_accuracy = float(((raw_scores > 0).astype(int) == y).mean())
    assert raw_oracle_accuracy == 1.0
    residual = np.linalg.norm(f[0::2] - f[1::2], axis=1)
    xf1 = _features(datasets[0][0], W, p)
    result = probe(xf1, datasets[0][1], f, y, seed=seed)
    # Paired eval must have identical predictions under exactly equal tokens.
    pair_preds = (
        LogisticRegression(max_iter=200).fit(np.round(xf1, 10), datasets[0][1]).predict(np.round(f, 10))
    )
    pair_acc = float((pair_preds == y).mean())
    np.savez_compressed(
        path / "collision_images.npz",
        x_plus=x[1::2][:4],
        x_minus=x[0::2][:4],
        difference=x[1::2][:4] - x[0::2][:4],
        token_diff=f[1::2][:4] - f[0::2][:4],
    )
    rows = [
        {
            "task": "fixed_collision",
            "pair_count": len(x) // 2,
            "max_token_residual": float(residual.max()),
            "relative_residual": float(residual.max() / max(np.linalg.norm(f, axis=1).mean(), 1e-15)),
            "raw_bayes_error": 0.0,
            "raw_sign_oracle_accuracy": raw_oracle_accuracy,
            "token_bayes_error": 0.5,
            "paired_probe_accuracy": pair_acc,
            "unrounded_probe_accuracy": result["accuracy"],
            "operator_rank": (x.shape[-1] // p) * (x.shape[-2] // p) * numerical_rank(W),
        }
    ]
    assert pair_acc == 0.5
    assert rows[0]["relative_residual"] < 1e-8
    return (
        {
            "token_bayes_error": 0.5,
            "raw_bayes_error": 0.0,
            "raw_sign_oracle_accuracy": raw_oracle_accuracy,
            "collision_residual_max": float(residual.max()),
            "paired_probe_accuracy": pair_acc,
            "unrounded_probe_accuracy": result["accuracy"],
        },
        rows,
        [],
    )


def e02(c, seed, path):
    rows = []
    p = int(c.get("patch_size", 8))
    m = p * p
    ranks = c.get("ranks", [8, 16])
    sigmas = c.get("sigmas", [0.25, 0.5, 1.0])
    alphas = c.get("alphas", [0, 0.25, 0.5, 0.75, 1])
    n = int(c.get("mc_samples", 5000))
    ds = int(c.get("train_samples", 256))
    for r in ranks:
        if r >= m:
            raise ValueError("Alignment grid must have nontrivial null space")
        W = random_projection(int(r), m, seed + int(r))
        for sig in sigmas:
            for a in alphas:
                v = gaussian_data.build_signal(W, float(a), seed + 999 + int(r))
                predicted = bayes_error(v, W, float(sig))
                mc = independent_monte_carlo(
                    W, v, float(sig), n, seed + int(a * 100) + int(sig * 1000) + int(r) * 10000
                )
                # Fitted linear probe on separately generated train/test draws (not oracle).
                g = np.random.default_rng(seed + int(a * 10000) + int(sig * 10000) + int(r) * 100000 + 31)
                y = g.integers(0, 2, size=ds, dtype=np.int64)
                xt = (2 * y - 1)[:, None] * v[None, :] / 2 + g.normal(0, sig, size=(ds, m))
                fitted = LogisticRegression(max_iter=250).fit(xt @ W.T, y)
                nprobe = int(c.get("probe_test_samples", 1000))
                yp = g.integers(0, 2, size=nprobe, dtype=np.int64)
                xp = (2 * yp - 1)[:, None] * v[None, :] / 2 + g.normal(0, sig, size=(nprobe, m))
                probe_error = float((fitted.predict(xp @ W.T) != yp).mean())
                rows.append(
                    {
                        "rank": r,
                        "patch_size": p,
                        "embedding_dim": r,
                        "alpha": float(a),
                        "sigma": float(sig),
                        "theory_bayes_error": predicted,
                        "raw_bayes_error": raw_error(v, sig),
                        "mc_bayes_error": mc["measured_error"],
                        "mc_se": mc["mc_se"],
                        "ci95_low": mc["ci95_low"],
                        "ci95_high": mc["ci95_high"],
                        "fitted_probe_error": probe_error,
                        "num_test_samples": n,
                        "num_errors": mc["num_errors"],
                        "ci_method": mc["ci_method"],
                        "oracle": mc["oracle"],
                        "label_sampling": mc["label_sampling"],
                        "theory_in_ci": mc["ci95_low"] <= predicted <= mc["ci95_high"],
                    }
                )
    rows_miss = [r for r in rows if not r["theory_in_ci"]]
    return (
        {
            "num_conditions": len(rows),
            "outside_nominal_95ci": len(rows_miss),
            "max_abs_mc_deviation": float(
                max(abs(r["theory_bayes_error"] - r["mc_bayes_error"]) for r in rows)
            ),
            "mean_abs_mc_deviation": float(
                np.mean([abs(r["theory_bayes_error"] - r["mc_bayes_error"]) for r in rows])
            ),
        },
        rows,
        [],
    )


def e03(c, seed, path):
    p = int(c.get("patch_size", 8))
    d = int(c.get("embedding_dim", 16))
    sigma = float(c.get("sigma", 0.4))
    W = random_projection(d, p * p, seed)
    rows = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = gaussian_data.build_signal(W, a, seed + 101)
        mc = independent_monte_carlo(W, v, sigma, int(c.get("mc_samples", 4000)), seed + int(a * 100))
        rows.append(
            {
                "rank": numerical_rank(W),
                "alpha": a,
                "measured_alignment": alignment(v, W),
                "theory_bayes_error": bayes_error(v, W, sigma),
                "raw_bayes_error": raw_error(v, sigma),
                "mc_bayes_error": mc["measured_error"],
                "mc_se": mc["mc_se"],
                "ci95_low": mc["ci95_low"],
                "ci95_high": mc["ci95_high"],
                "num_errors": mc["num_errors"],
                "num_test_samples": mc["num_test_samples"],
                "ci_method": mc["ci_method"],
                "oracle": mc["oracle"],
            }
        )
    return (
        {
            "chance_error_null": rows[0]["theory_bayes_error"],
            "full_signal_error": rows[-1]["theory_bayes_error"],
            "task_dependent_error_gap": rows[0]["theory_bayes_error"] - rows[-1]["theory_bayes_error"],
        },
        rows,
        [],
    )


def e04(c, seed, path):
    p = int(c.get("patch_size", 4))
    d = int(c.get("embedding_dim", 8))
    W = random_projection(d, p * p, seed)
    rows = []
    for kind in (
        ("frequency", "marker", "relational")
        if c.get("include_marker", False)
        else ("frequency", "relational")
    ):
        cfg = {
            **c,
            "mode": "marker" if kind == "marker" else "frequency",
            "marker_size": 2,
            "noise": 0.0 if kind == "relational" else 0.02,
        }
        generate = relational.generate if kind == "relational" else frequency.generate
        nt = int(c.get("train_samples", 512))
        ne = int(c.get("test_samples", 256))
        # XOR train/test are balanced across four underlying bit pairs.
        x, y, _ = _dataset(generate, seed, "train", nt, cfg, path, kind)
        xt, yt, _ = _dataset(generate, seed, "test", ne, cfg, path, kind)
        patches = patchify_np(x, p)
        tar = tar_mean(patches, y, W)
        linear = probe(_features(x, W, p), y, _features(xt, W, p), yt, seed=seed)
        nonlinear = probe(_features(x, W, p), y, _features(xt, W, p), yt, True, seed=seed)
        mean_diff = float(np.linalg.norm(x[y == 0].mean(axis=0) - x[y == 1].mean(axis=0)))
        rows.append(
            {
                "task": kind,
                "tar_mean": tar,
                "train_mean_difference": mean_diff,
                "logistic_accuracy": linear["accuracy"],
                "mlp_accuracy": nonlinear["accuracy"],
            }
        )
    xor = next(r for r in rows if r["task"] == "relational")
    return (
        {
            "frequency_tar": rows[0]["tar_mean"],
            "marker_tar": next((r["tar_mean"] for r in rows if r["task"] == "marker"), None),
            "xor_tar": xor["tar_mean"],
            "xor_mean_difference": xor["train_mean_difference"],
            "xor_linear_accuracy": xor["logistic_accuracy"],
            "xor_mlp_accuracy": xor["mlp_accuracy"],
        },
        rows,
        [],
    )


def e05(c, seed, path):
    h = int(c.get("height", 16))
    w = int(c.get("width", 16))
    rows = []
    for p in c.get("patch_sizes", [4, 8]):
        for mode, freq, size, shift in [("frequency", f, 2, s) for f in [2, 4, 6] for s in [0, 2]] + [
            ("marker", 1, s, 0) for s in [1, 2, 4]
        ]:
            d = int(c.get("embedding_dim", 8))
            W = random_projection(d, p * p, seed + p)
            cfg = {
                **c,
                "patch_size": p,
                "frequency": freq,
                "marker_size": size,
                "phase_shift": shift,
                "mode": mode,
                "height": h,
                "width": w,
            }
            x, y, _ = _dataset(frequency.generate, seed, "train", int(c.get("train_samples", 256)), cfg, path)
            z, t, _ = _dataset(frequency.generate, seed, "test", int(c.get("test_samples", 128)), cfg, path)
            pr = probe(_features(x, W, p), y, _features(z, W, p), t, seed=seed)
            rows.append(
                {
                    "task": mode,
                    "patch_size": p,
                    "embedding_dim": d,
                    "rank": numerical_rank(W) * h * w // (p * p),
                    "frequency": freq,
                    "marker_size": size,
                    "phase_shift": shift,
                    "shift_design": "matched train/test; uniform grating phase is distribution-invariant",
                    "accuracy": pr["accuracy"],
                    "error": pr["error"],
                    "token_count": h * w // (p * p),
                    "token_budget": h * w // (p * p) * d,
                    "train_samples": len(x),
                    "test_samples": len(z),
                }
            )
    return {"cells": len(rows), "mean_accuracy": float(np.mean([r["accuracy"] for r in rows]))}, rows, []


def e06(c, seed, path, device="cpu"):
    # Frozen null vector selected from initial W, supervised reorientation may recover it.
    p = int(c.get("patch_size", 8))
    d = int(c.get("embedding_dim", 16))
    h = w = p
    cfg = {
        **c,
        "height": h,
        "width": w,
        "patch_size": p,
        "embedding_dim": d,
        "alpha": 0.0,
        "sigma": float(c.get("sigma", 0.45)),
    }
    ds = [
        _dataset(gaussian_data.generate, seed, s, int(c.get(k, 256)), cfg, path)[:2]
        for s, k in [("train", "train_samples"), ("validation", "val_samples"), ("test", "test_samples")]
    ]
    W = random_projection(d, p * p, seed)
    v = gaussian_data.build_signal(W, 0.0, seed + 42)
    rows = []
    history = []
    for frozen in [True, False]:
        seed_all(seed + 17)
        model = TinyViT(
            h,
            w,
            p,
            d=d,
            heads=int(c.get("heads", 2)),
            depth=int(c.get("depth", 2)),
            trainable=not frozen,
            seed=seed,
        )
        result, hist, _ = fit_model(model, ds[0], ds[1], ds[2], c, seed + 12, device, signal=v)
        result["final_alignment"] = alignment(v, model.tokenizer.matrix)
        result["initial_alignment"] = alignment(v, W)
        result["frozen"] = frozen
        result["token_count"] = 1
        result["token_budget"] = d
        result["patch_size"] = p
        result["flops_proxy"] = transformer_flops_proxy(1, p * p, d, int(c.get("depth", 2)))
        save_checkpoint(model, path / ("frozen.pt" if frozen else "learned.pt"))
        rows.append(result)
        for entry in hist:
            history.append({"model": "frozen" if frozen else "learned", **entry})
    return (
        {
            "frozen_accuracy": rows[0]["accuracy"],
            "learned_accuracy": rows[1]["accuracy"],
            "frozen_alignment": rows[0]["final_alignment"],
            "learned_alignment": rows[1]["final_alignment"],
        },
        rows,
        history,
    )


def e07(c, seed, path):
    # Label-relevant high-freq sign loses signal under the binomial 5x5 kernel;
    # label-independent checker nuisance may be safely removed.
    rows = []
    p = int(c.get("patch_size", 4))
    d = int(c.get("embedding_dim", 8))
    h = int(c.get("height", 16))
    w = int(c.get("width", 16))

    low = BinomialLowPass(1).eval()
    for family in ("signal", "nuisance"):
        cfg = {**c, "family": family}
        x, y, _ = _dataset(
            alias_tradeoff.generate, seed, "train", int(c.get("train_samples", 256)), cfg, path, family
        )
        xt, yt, _ = _dataset(
            alias_tradeoff.generate, seed, "test", int(c.get("test_samples", 128)), cfg, path, family
        )

        def bl(a):
            return low(torch.from_numpy(a).float()).numpy()

        for filt in (False, True):
            ax = bl(x) if filt else x
            az = bl(xt) if filt else xt
            W = random_projection(d, p * p, seed + 3)
            pr = probe(_features(ax, W, p), y, _features(az, W, p), yt, seed=seed)
            rows.append(
                {
                    "family": family,
                    "lowpass": filt,
                    "accuracy": pr["accuracy"],
                    "error": pr["error"],
                    "patch_size": p,
                    "token_count": h * w // (p * p),
                    "token_budget": h * w // (p * p) * d,
                    "kernel": "[1,4,6,4,1] outer product / 256; circular; same resolution"
                    if filt
                    else "none",
                }
            )
    return (
        {
            "signal_raw_accuracy": rows[0]["accuracy"],
            "signal_lowpass_accuracy": rows[1]["accuracy"],
            "nuisance_raw_accuracy": rows[2]["accuracy"],
            "nuisance_lowpass_accuracy": rows[3]["accuracy"],
        },
        rows,
        [],
    )


def e08(c, seed, path):
    h = int(c.get("height", 8))
    w = int(c.get("width", 8))
    p = int(c.get("patch_size", 4))
    d = int(c.get("embedding_dim", 8))
    s = int(c.get("stride", 2))
    if (h - p) % s or (w - p) % s:
        raise ValueError("Overlap experiment requires integral overlapping grid")
    rows = []
    cfg = {**c, "height": h, "width": w, "mode": "marker", "noise": 0.04}
    x, y, _ = _dataset(frequency.generate, seed, "train", int(c.get("train_samples", 256)), cfg, path)
    xt, yt, _ = _dataset(frequency.generate, seed, "test", int(c.get("test_samples", 128)), cfg, path)
    for stride in (p, s):
        op = OverlapTokenizer(p, stride, d=d, seed=seed).double()
        A = materialize_operator(op, (1, h, w), dtype=torch.float64)
        with torch.no_grad():
            f = op(torch.from_numpy(x).double()).numpy().reshape(len(x), -1)
            ft = op(torch.from_numpy(xt).double()).numpy().reshape(len(xt), -1)
        pr = probe(f, y, ft, yt, seed=seed)
        rows.append(
            {
                "stride": stride,
                "overlap": stride < p,
                "operator_rank": numerical_rank(A, 1e-9),
                "operator_nullity": h * w - numerical_rank(A, 1e-9),
                "token_count": (1 + (h - p) // stride) * (1 + (w - p) // stride),
                "token_budget": A.shape[0],
                "accuracy": pr["accuracy"],
                "error": pr["error"],
                "operator_shape": list(A.shape),
            }
        )
    return (
        {
            "nonoverlap_rank": rows[0]["operator_rank"],
            "overlap_rank": rows[1]["operator_rank"],
            "nonoverlap_accuracy": rows[0]["accuracy"],
            "overlap_accuracy": rows[1]["accuracy"],
        },
        rows,
        [],
    )


def e09(c, seed, path, device="cpu"):
    p = int(c.get("patch_size", 4))
    h = int(c.get("height", 16))
    w = int(c.get("width", 16))
    d = int(c.get("embedding_dim", 32))
    rows = []
    history = []
    for task in c.get("tasks", ["frequency", "relational"]):
        cfg = {**c, "mode": "frequency", "noise": float(c.get("noise", 0.025))}
        gen = frequency.generate if task == "frequency" else relational.generate
        ds = [
            _dataset(gen, seed, split, int(c.get(k, 256)), cfg, path, task)[:2]
            for split, k in [
                ("train", "train_samples"),
                ("validation", "val_samples"),
                ("test", "test_samples"),
            ]
        ]
        for model_name in ("tiny_vit_frozen", "tiny_vit_learned", "cnn"):
            seed_all(seed + 7)
            model = (
                SmallCNN()
                if model_name == "cnn"
                else TinyViT(
                    h,
                    w,
                    p,
                    d=d,
                    heads=int(c.get("heads", 2)),
                    depth=int(c.get("depth", 2)),
                    trainable=model_name.endswith("learned"),
                    seed=seed,
                )
            )
            result, hist, _ = fit_model(model, ds[0], ds[1], ds[2], c, seed + 13, device)
            result.update(
                {
                    "task": task,
                    "model": model_name,
                    "patch_size": p if model_name != "cnn" else None,
                    "embedding_dim": d if model_name != "cnn" else None,
                    "token_count": (h * w // (p * p)) if model_name != "cnn" else None,
                    "token_budget": (h * w // (p * p) * d) if model_name != "cnn" else None,
                    "flops_proxy": transformer_flops_proxy(h * w // (p * p), p * p, d, int(c.get("depth", 2)))
                    if model_name != "cnn"
                    else None,
                }
            )
            save_checkpoint(model, path / f"{task}_{model_name}.pt")
            rows.append(result)
            for entry in hist:
                history.append({"task": task, "model": model_name, **entry})
    return (
        {"num_trained": len(rows), "mean_accuracy": float(np.mean([r["accuracy"] for r in rows]))},
        rows,
        history,
    )


def e10(c, seed, path, device="cpu"):
    if not c.get("allow_download", False):
        raise RuntimeError("E10 requires explicit allow_download: true; not part of offline core")
    ds = [
        _dataset(natural.generate, seed, split, int(c.get(k, 256)), c, path)[:2]
        for split, k in [("train", "train_samples"), ("validation", "val_samples"), ("test", "test_samples")]
    ]
    h = int(c.get("height", 28))
    p = int(c.get("patch_size", 4))
    d = int(c.get("embedding_dim", 32))
    rows = []
    history = []
    for kind in ("tiny_vit_learned", "cnn"):
        seed_all(seed + 7)
        model = (
            TinyViT(h, h, p, d=d, heads=int(c.get("heads", 2)), classes=10, trainable=True, seed=seed)
            if kind == "tiny_vit_learned"
            else SmallCNN(classes=10)
        )
        metrics, hist, _ = fit_model(model, ds[0], ds[1], ds[2], c, seed + 7, device)
        save_checkpoint(model, path / f"{kind}.pt")
        rows.append(
            {
                **metrics,
                "model": kind,
                "task": "MNIST exploratory, official test split",
                "train_examples": len(ds[0][0]),
                "test_examples": len(ds[2][0]),
            }
        )
        history.extend([{"model": kind, **r} for r in hist])
    return (
        {"num_trained": len(rows), "mean_accuracy": float(np.mean([r["accuracy"] for r in rows]))},
        rows,
        history,
    )


FUNCTIONS = {
    "E00": e00,
    "E01": e01,
    "E02": e02,
    "E03": e03,
    "E04": e04,
    "E05": e05,
    "E06": e06,
    "E07": e07,
    "E08": e08,
    "E09": e09,
    "E10": e10,
}


def run_experiment(config, seed=0, output="runs", device="cpu"):
    validate(config)
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    seed_all(seed)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    slug = f"{config['experiment']}-s{seed}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    path = root / slug
    path.mkdir(exist_ok=False)
    import yaml

    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    meta = {
        "run_id": slug,
        "experiment_id": config["experiment"],
        "config_hash": config_hash(config),
        "seed": seed,
        "device": device,
        "hardware": platform.processor() or platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "reproduction_argv": [
            sys.executable,
            "-m",
            "patchloss.cli",
            "run",
            "--config",
            str((path / "config.yaml").resolve()),
            "--seed",
            str(seed),
            "--device",
            device,
            "--output",
            str(root.resolve()),
        ],
        "invocation_argv": sys.argv,
        "dependencies": {
            name: version(name)
            for name in ("numpy", "scipy", "torch", "scikit-learn", "pandas", "matplotlib", "PyYAML")
        },
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "input_shape": [
            int(config.get("channels", 1)),
            int(config.get("height", 32)),
            int(config.get("width", 32)),
        ],
        "patch_size": int(config.get("patch_size", 8)),
        "stride": int(config.get("stride", config.get("patch_size", 8))),
        "patch_dim": int(config.get("channels", 1)) * int(config.get("patch_size", 8)) ** 2,
        "embedding_dim": int(config.get("embedding_dim", 16)),
        "train_samples": config.get("train_samples"),
        "val_samples": config.get("val_samples"),
        "test_samples": config.get("test_samples"),
    }
    from ..seed import split_seed

    meta["split_seed"] = split_seed(seed, "test")
    (path / "metadata.json").write_text(json.dumps(meta, indent=2))
    begin = time.perf_counter()
    try:
        source_root = Path(__file__).resolve().parents[1]
        source_files = {
            str(f.relative_to(source_root)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(source_root.rglob("*.py"))
        }
        source_hash = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
        (path / "source_snapshot.json").write_text(json.dumps(source_files, indent=2))
        meta["source_sha256"] = source_hash
        meta["result_schema_version"] = 2
        meta["flops_proxy_convention"] = (
            "forward/image; 2 FLOPs/MAC; embedding + depth*(QKV/output + FFN(2d) + attention) + head; excludes elementwise ops"
        )
        fun = FUNCTIONS[config["experiment"]]
        metrics, rows, history = (
            fun(config, seed, path, device)
            if config["experiment"] in {"E06", "E09", "E10"}
            else fun(config, seed, path)
        )
        elapsed = time.perf_counter() - begin
        metrics.update(
            {
                "wallclock_sec": elapsed,
                "status": "completed",
                "experiment_id": config["experiment"],
                "seed": seed,
            }
        )
        results = {"metrics": _jsonable(metrics), "rows": _jsonable(rows)}
        (path / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))
        _csv(path / "history.csv", history)
        meta.update(
            {
                "status": "completed",
                "wallclock_sec": elapsed,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_sha256": {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in path.iterdir()
                    if f.is_file() and f.name not in {"metadata.json"}
                },
            }
        )
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))
        return path, metrics
    except Exception as ex:
        meta.update({"status": "failed", "wallclock_sec": time.perf_counter() - begin})
        (path / "failure.json").write_text(
            json.dumps({"error": str(ex), "traceback": traceback.format_exc()}, indent=2)
        )
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))
        raise
