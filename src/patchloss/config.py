"""Explicit, validated YAML configs."""

import hashlib
import json
import math
from pathlib import Path
import yaml


def load_config(path):
    d = yaml.safe_load(Path(path).read_text())
    if not isinstance(d, dict):
        raise ValueError("YAML must contain a mapping")
    validate(d)
    return d


def validate(c):
    experiment = c.get("experiment")
    if experiment not in {"E00", "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10"}:
        raise ValueError(f"Unknown experiment: {experiment}")
    for key in (
        "height",
        "width",
        "channels",
        "patch_size",
        "embedding_dim",
        "heads",
        "depth",
        "train_samples",
        "val_samples",
        "test_samples",
        "epochs",
        "batch_size",
        "patience",
        "mc_samples",
        "probe_test_samples",
        "stride",
        "marker_size",
        "validation_pool_size",
    ):
        if key in c and (isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    h, w, p = int(c.get("height", 32)), int(c.get("width", 32)), int(c.get("patch_size", 8))
    if min(h, w, p) <= 0 or h % p or w % p:
        raise ValueError("H/W must be positive and divisible by patch_size")
    d = int(c.get("embedding_dim", 16))
    if d < 1:
        raise ValueError("embedding_dim must be positive")
    if experiment in {"E06", "E09", "E10"} and d % int(c.get("heads", 2)):
        raise ValueError("embedding_dim must be divisible by heads")
    if (
        min(int(c.get("train_samples", 128)), int(c.get("val_samples", 64)), int(c.get("test_samples", 128)))
        < 4
    ):
        raise ValueError("All split sizes must be at least four")
    if experiment == "E01" and d >= p * p * int(c.get("channels", 1)):
        raise ValueError("E01 requires d<m for a null space")
    if experiment == "E02" and int(c.get("mc_samples", 1000)) < 100:
        raise ValueError("E02 needs at least 100 Monte Carlo samples")
    if experiment in {"E06", "E09"} and int(c.get("epochs", 1)) < 1:
        raise ValueError("At least one training epoch required")
    if experiment != "E00" and c.get("channels", 1) != 1:
        raise ValueError("Synthetic experiment generators and CNN currently support grayscale only")
    for key in ("sigma", "lr"):
        if key in c and (not math.isfinite(float(c[key])) or float(c[key]) <= 0):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("noise", "weight_decay"):
        if key in c and (not math.isfinite(float(c[key])) or float(c[key]) < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    if experiment in {"E01", "E04", "E05", "E07", "E08", "E09"}:
        multiple = (
            4
            if experiment == "E04"
            or (experiment == "E09" and "relational" in c.get("tasks", ["frequency", "relational"]))
            else 2
        )
        if any(
            int(c.get(key, default)) % multiple
            for key, default in [("train_samples", 128), ("val_samples", 64), ("test_samples", 128)]
        ):
            raise ValueError(f"Split sizes for {experiment} must be divisible by {multiple}")
    if experiment == "E06" and (h != p or w != p or d >= p * p):
        raise ValueError("E06 needs H=W=p and a nontrivial initial null space (d<p*p)")
    if experiment == "E03" and d >= p * p:
        raise ValueError("E03 needs both row and null spaces")
    if experiment == "E02":
        ranks = c.get("ranks", [8, 16])
        alphas = c.get("alphas", [0, 0.25, 0.5, 0.75, 1])
        sigmas = c.get("sigmas", [0.25, 0.5, 1])
        if not ranks or any(not isinstance(r, int) or not 0 < r < p * p for r in ranks):
            raise ValueError("E02 ranks must be integers in [1,p*p)")
        if not alphas or any(not math.isfinite(float(a)) or not 0 <= float(a) <= 1 for a in alphas):
            raise ValueError("E02 alphas must lie in [0,1]")
        if not sigmas or any(not math.isfinite(float(s)) or float(s) <= 0 for s in sigmas):
            raise ValueError("E02 sigmas must be finite and positive")
    if experiment == "E05" and (
        not c.get("patch_sizes", [4, 8])
        or any(not isinstance(q, int) or q <= 0 or h % q or w % q for q in c.get("patch_sizes", [4, 8]))
    ):
        raise ValueError("Every sweep patch size must divide H and W")
    if experiment == "E08":
        stride = int(c.get("stride", 2))
        if not 1 <= stride <= p or (h - p) % stride or (w - p) % stride:
            raise ValueError("Overlap requires stride in [1,p] and complete border coverage")
    if experiment == "E09" and (
        not c.get("tasks", ["frequency", "relational"])
        or set(c.get("tasks", [])) - {"frequency", "relational"}
    ):
        raise ValueError("E09 tasks must be frequency or relational")
    if experiment in {"E04", "E09"} and min(h, w) < 8:
        raise ValueError("Relational tasks need images at least 8x8")
    if experiment == "E07" and (h % 2 or w % 2):
        raise ValueError("The periodic checkerboard null control requires even H/W")
    if experiment == "E10" and (h, w) != (28, 28):
        raise ValueError("MNIST is 28x28; resizing is not implemented")
    return c


def config_hash(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
