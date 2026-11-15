"""Official CIFAR partitions with separate operator, diagnostic, validation and test IDs."""

import hashlib
from pathlib import Path
import numpy as np
from ..seed import split_seed


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def stratified_parts(labels, sizes, seed):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if not classes.size or any(n <= 0 or n % len(classes) for n in sizes.values()):
        raise ValueError("Each partition size must be a positive multiple of the number of classes")
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in sizes}
    for cls in classes:
        ids = rng.permutation(np.flatnonzero(labels == cls))
        if len(ids) < sum(sizes.values()) // len(classes):
            raise ValueError("Insufficient class examples for disjoint partitions")
        offset = 0
        for name, n in sizes.items():
            count = n // len(classes)
            parts[name].extend(ids[offset : offset + count].tolist())
            offset += count
    result = {name: rng.permutation(ids).astype(np.int64) for name, ids in parts.items()}
    all_ids = np.concatenate(list(result.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError("Partition overlap")
    return result


def diagnostic_folds(labels, seed):
    n = len(labels)
    return stratified_parts(labels, {"first": n // 2, "second": n // 2}, seed)


def load_cifar(name, seed, config):
    from torchvision.datasets import CIFAR10, CIFAR100

    cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[name]
    # Download is a separate explicit preparation step; primary analysis is offline.
    train = cls(Path(config["data_root"]), train=True, download=False)
    test = cls(Path(config["data_root"]), train=False, download=False)
    train_y = np.asarray(train.targets, dtype=np.int64)
    test_y = np.asarray(test.targets, dtype=np.int64)
    indices = stratified_parts(
        train_y,
        {
            "operator": int(config["operator_samples"]),
            "diagnostic": int(config["diagnostic_samples"]),
            "validation": int(config["validation_samples"]),
        },
        split_seed(seed, name + ":official_train"),
    )
    indices["test"] = stratified_parts(
        test_y, {"test": int(config["test_samples"])}, split_seed(seed, name + ":official_test")
    )["test"]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float64)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float64)[None, :, None, None]
    data = {}
    md = {
        "dataset": name,
        "seed": seed,
        "source": "official CIFAR train/test partitions",
        "normalization": "fixed ImageNet mean/std; native 32x32 RGB; no augmentation",
        "indices": {k: v.tolist() for k, v in indices.items()},
        "partitions": {},
    }
    for part, ids in indices.items():
        images = (test.data if part == "test" else train.data)[ids]
        labels = (test_y if part == "test" else train_y)[ids]
        normalized = (images.transpose(0, 3, 1, 2).astype(np.float64) / 255.0 - mean) / std
        data[part] = (np.ascontiguousarray(normalized), labels)
        md["partitions"][part] = {
            "official_partition": "test" if part == "test" else "train",
            "raw_images_sha256": array_hash(images),
            "normalized_images_sha256": array_hash(normalized),
            "labels_sha256": array_hash(labels),
            "class_counts": np.bincount(labels).tolist(),
        }
    return data, md
