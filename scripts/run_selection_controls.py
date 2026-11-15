"""Known-model calibration, pretrained geometry, and fresh-noise optimization controls."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution, binomtest, kstest, norm
from scipy.linalg import cho_solve, cho_factor
from threadpoolctl import threadpool_limits
import torch
from patchloss.seed import seed_all, split_seed
from patchloss.selection.study import source_snapshot
from patchloss.theory.linear_algebra import spectrum_info
from patchloss.data.relational import generate as xor_generate
from patchloss.models.tiny_vit import TinyViT
from patchloss.models.small_cnn import SmallCNN
from patchloss.training.train import fit_model
from patchloss.training.checkpoint import save_checkpoint


def null_calibration(root):
    rng = np.random.default_rng(312)
    records = []
    for count in [5, 20, 100]:
        ratios = []
        for _ in range(32):
            a = rng.normal(size=(125, count, 64)).mean(1)
            b = rng.normal(size=(125, count, 64)).mean(1)
            contrast = a - b
            ratios.extend((np.sum(contrast[:, :16] ** 2, axis=1) / np.sum(contrast**2, axis=1)).tolist())
        ratios = np.asarray(ratios)
        ks = kstest(ratios, beta_distribution(8, 24).cdf)
        records.append(
            {
                "samples_per_class": count,
                "replications": len(ratios),
                "population_mean_tar": None,
                "null_expected_empirical_tar": 0.25,
                "mean_empirical_tar": float(ratios.mean()),
                "variance_empirical_tar": float(ratios.var()),
                "theory_variance": float(beta_distribution(8, 24).var()),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
        np.save(root / f"null_tar_n{count}.npy", ratios)
    pd.DataFrame(records).to_csv(root / "null_tar.csv", index=False)
    return records


def anisotropic_control(root):
    rng = np.random.default_rng(241)
    covariance = np.diag([0.01, 10.0])
    v = np.ones(2) / np.sqrt(2)
    n = 30000
    y = 2 * rng.integers(0, 2, size=n) - 1
    x = y[:, None] * v / 2 + rng.normal(size=(n, 2)) @ np.linalg.cholesky(covariance).T
    full = float(v @ np.linalg.solve(covariance, v))
    records = []
    for j in [0, 1]:
        W = np.eye(2)[j : j + 1]
        z = x @ W.T
        signal = W @ v
        projected_cov = W @ covariance @ W.T
        energy = float(signal @ cho_solve(cho_factor(projected_cov), signal))
        prediction = np.where((z @ np.linalg.solve(projected_cov, signal)) >= 0, 1, -1)
        errors = int(np.sum(prediction != y))
        ci = binomtest(errors, n).proportion_ci(method="exact")
        records.append(
            {
                "observed_coordinate": j,
                "rank": 1,
                "euclidean_tar": 0.5,
                "covariance_retention": energy / full,
                "theory_bayes_error": float(norm.cdf(-np.sqrt(energy) / 2)),
                "mc_error": errors / n,
                "num_errors": errors,
                "num_test_samples": n,
                "ci95_low": ci.low,
                "ci95_high": ci.high,
            }
        )
    pd.DataFrame(records).to_csv(root / "anisotropic.csv", index=False)
    return records


def pretrained_audit(root, data_root):
    records = []
    assets = [
        ("deit_tiny", "model", "patch_embed.proj.weight", 16, 192),
        ("vit_b16", None, "conv_proj.weight", 16, 768),
    ]
    for name, container, key, p, d in assets:
        path = Path(data_root) / (name + ".pth")
        state = torch.load(path, map_location="cpu", weights_only=True)
        matrix = (state[container] if container else state)[key].numpy().astype(np.float64).reshape(d, -1)
        info = spectrum_info(matrix)
        # A dtype-aware tolerance is also reported; float64 SVD does not restore
        # information absent from the originally stored float32 coefficients.
        s = np.asarray(info["singular_values"])
        records.append(
            {
                "model": name,
                "patch_size": p,
                "channels": 3,
                "embedding_dim": d,
                "patch_dim": matrix.shape[1],
                "rank_rtol_1e10": info["rank"],
                "rank_float32_convention": int(
                    np.sum(s > max(matrix.shape) * np.finfo(np.float32).eps * s[0])
                ),
                "condition_nonzero": info["condition_nonzero"],
                "sigma_min": float(s[-1]),
                "sigma_max": float(s[0]),
                "stored_dtype": "float32",
                "svd_dtype": "float64",
                "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "singular_values": s.tolist(),
                "full_column_rank": info["rank"] == matrix.shape[1],
            }
        )
        np.savez_compressed(root / f"{name}_patch_matrix.npz", matrix=matrix, singular_values=s)
        del state
    return records


class RelationalMLP(torch.nn.Module):
    """Transparent raw-pixel capacity control, not a new model or Bayes oracle."""

    def __init__(self, h=16, w=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(h * w, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x - 0.5)


def relational_controls(root, seeds):
    rows = []
    history = []
    config = {
        "height": 16,
        "width": 16,
        "noise": 0.03,
        "epochs": 80,
        "patience": 12,
        "lr": 0.002,
        "batch_size": 64,
        "weight_decay": 0.001,
    }
    for seed in seeds:
        datasets = [
            xor_generate(seed, split, count, config)[:2]
            for split, count in [("train", 1024), ("validation", 256), ("test", 1024)]
        ]
        for name, cap in [(name, cap) for name in ["raw_mlp", "tiny_vit", "cnn"] for cap in [6, 80]]:
            seed_all(seed + 27)
            model = (
                RelationalMLP()
                if name == "raw_mlp"
                else SmallCNN()
                if name == "cnn"
                else TinyViT(h=16, w=16, p=4, d=32, heads=2, depth=2, trainable=True, seed=seed)
            )
            result, curve, _ = fit_model(model, *datasets, {**config, "epochs": cap}, seed + 71)
            save_checkpoint(model, root / f"xor_{name}_cap{cap}_seed{seed}.pt")
            rows.append({"seed": seed, "model": name, "epoch_cap": cap, **result})
            history.extend({"seed": seed, "model": name, "epoch_cap": cap, **item} for item in curve)
            print(
                f"XOR CONTROL {name} cap={cap} seed={seed} accuracy={result['accuracy']:.3f} epochs={result['trained_epochs']}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(root / "xor_capacity.csv", index=False)
    pd.DataFrame(history).to_csv(root / "xor_history.csv", index=False)
    return rows


def local_shapes(seed, split, n, shift=0):
    rng = np.random.default_rng(split_seed(seed, "localized_shape:" + split))
    y = np.arange(n) % 2
    rng.shuffle(y)
    x = rng.normal(0, 0.04, size=(n, 1, 16, 16)) + 0.5
    for i, label in enumerate(y):
        shape = np.zeros((16, 16))
        if label == 0:
            shape[6, 4:8] = 0.3
        else:
            shape[6:10, 4] = 0.3
        x[i, 0] += np.roll(shape, shift, axis=1)
    return x.astype(np.float32), y.astype(np.int64)


def translation_controls(root, seeds):
    from patchloss.models.probes import probe
    from patchloss.operators.linear_patch import random_projection
    from patchloss.operators.patchify import patchify_np

    rows = []
    for seed in seeds:
        x, y = local_shapes(seed, "train", 512)
        for p in [4, 8]:
            W = random_projection(8, p * p, seed + 99)
            train = (patchify_np(x, p) @ W.T).reshape(len(x), -1)
            for shift in [0, 1, 2, 3, 4, 8]:
                test, target = local_shapes(seed, "test", 512, shift)
                features = (patchify_np(test, p) @ W.T).reshape(len(test), -1)
                result = probe(train, y, features, target, seed=seed)
                rows.append(
                    {
                        "seed": seed,
                        "patch_size": p,
                        "test_shift": shift,
                        "train_shift": 0,
                        "task": "localized equal-area bar orientation",
                        **result,
                    }
                )
    pd.DataFrame(rows).to_csv(root / "heldout_translation.csv", index=False)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/selection_controls")
    parser.add_argument("--data-root", default="data/selection")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Use a new directory for an immutable control cohort")
    root.mkdir(parents=True, exist_ok=True)
    snapshot, source_hash = source_snapshot()
    (root / "source_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    started = time.perf_counter()
    with threadpool_limits(2):
        torch.set_num_threads(2)
        result = {
            "null_calibration": null_calibration(root),
            "anisotropic": anisotropic_control(root),
            "pretrained_audit": pretrained_audit(root, args.data_root),
            "translations": translation_controls(root, [0, 1, 2, 3, 4]),
            "capacity": [] if args.skip_training else relational_controls(root, [0, 1, 2, 3, 4]),
        }
    (root / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    metadata = {
        "status": "completed",
        "source_sha256": source_hash,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wallclock_sec": time.perf_counter() - started,
        "artifact_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in root.iterdir()
            if f.is_file() and f.name != "metadata.json"
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"CONTROL COHORT COMPLETE in {metadata['wallclock_sec']:.1f}s", flush=True)
