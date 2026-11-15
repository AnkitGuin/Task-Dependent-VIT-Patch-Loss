"""Content-clean training partitions and complete official test sets."""

import hashlib
import numpy as np
from .data import array_hash, stratified_parts
from ..seed import split_seed


def image_hashes(images):
    return np.array([hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in images])


def clean_training_pool(train_images, train_labels, test_images):
    train_hash = image_hashes(train_images)
    test_hash = image_hashes(test_images)
    test_set = set(test_hash)
    groups = {}
    for i, digest in enumerate(train_hash):
        groups.setdefault(digest, []).append(i)
    keep = []
    overlap = conflicting = repeated = 0
    for digest, ids in groups.items():
        if digest in test_set:
            overlap += len(ids)
        elif len(np.unique(train_labels[ids])) > 1:
            conflicting += len(ids)
        else:
            keep.append(ids[0])
            repeated += len(ids) - 1
    keep = np.array(sorted(keep), dtype=np.int64)
    return keep, train_hash, test_hash, {
        "official_train_size": len(train_images), "official_test_size": len(test_images),
        "training_rows_matching_test_content_removed": overlap,
        "training_rows_in_conflicting_label_groups_removed": conflicting,
        "repeated_training_rows_removed": repeated,
        "clean_training_pool_size": len(keep),
        "repeated_official_test_rows_retained": len(test_hash) - len(set(test_hash)),
        "policy": "test-priority content deduplication; full official test retained",
    }


def load_revision(name, seed, config):
    from torchvision.datasets import CIFAR10, CIFAR100, SVHN
    root = config["data_root"]
    if name == "svhn":
        train, test = SVHN(root, split="train"), SVHN(root, split="test")
        train_x, test_x = train.data.transpose(0, 2, 3, 1), test.data.transpose(0, 2, 3, 1)
        train_y, test_y = np.asarray(train.labels), np.asarray(test.labels)
    else:
        cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[name]
        train, test = cls(root, train=True), cls(root, train=False)
        train_x, test_x = train.data, test.data
        train_y, test_y = np.asarray(train.targets), np.asarray(test.targets)
    clean, train_hash, test_hash, audit = clean_training_pool(train_x, train_y, test_x)
    positions = stratified_parts(train_y[clean], {
        k: int(config[k + "_samples"]) for k in ["operator", "diagnostic", "validation"]
    }, split_seed(seed, name + ":revision_train"))
    indices = {k: clean[v] for k, v in positions.items()}
    indices["test"] = np.arange(len(test_y))
    mean = np.array([.485, .456, .406])[None, :, None, None]
    std = np.array([.229, .224, .225])[None, :, None, None]
    data, partitions = {}, {}
    for part, ids in indices.items():
        images = (test_x if part == "test" else train_x)[ids]
        labels = (test_y if part == "test" else train_y)[ids]
        normalized = np.ascontiguousarray((images.transpose(0, 3, 1, 2) / 255. - mean) / std)
        data[part] = normalized, labels.astype(np.int64)
        partitions[part] = dict(raw_images_sha256=array_hash(images), labels_sha256=array_hash(labels),
                                class_counts=np.bincount(labels).tolist(),
                                content_hashes=(test_hash if part == "test" else train_hash)[ids].tolist())
    sets = [set(partitions[k]["content_hashes"]) for k in partitions]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise AssertionError("Content overlap survived deduplication")
    return data, dict(dataset=name, seed=seed, indices={k: v.tolist() for k, v in indices.items()},
                     partitions=partitions, deduplication=audit)
