"""Optional, requires explicit user-invoked torchvision download flag."""

import numpy as np
from .base import metadata


def split_indices(seed, split, n_samples, train_size, test_size, validation_size=10000):
    """Partition official training IDs once; requested sample counts cannot move the boundary."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown natural-data split: {split}")
    if not 0 < validation_size < train_size:
        raise ValueError("validation_size must be between zero and official train size")
    order = np.random.default_rng(seed).permutation(train_size)
    if split == "train":
        pool = order[validation_size:]
    elif split == "validation":
        pool = order[:validation_size]
    else:
        pool = np.random.default_rng(seed + 11).permutation(test_size)
    if not 0 < n_samples <= len(pool):
        raise ValueError(f"Requested {n_samples} examples from {split} pool of {len(pool)}")
    return pool[:n_samples]


def generate(seed, split, n_samples, config):
    if not config.get("allow_download", False):
        raise RuntimeError("Natural-image data are optional. Set allow_download: true explicitly")
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown natural-data split: {split}")
    ds = MNIST(
        root=config.get("data_root", "data"), train=split != "test", download=True, transform=ToTensor()
    )
    idx = split_indices(
        seed,
        split,
        n_samples,
        60000 if split == "test" else len(ds),
        len(ds) if split == "test" else 10000,
        int(config.get("validation_pool_size", 10000)),
    )
    images = np.stack([ds[int(i)][0].numpy() for i in idx])
    labels = np.array([ds[int(i)][1] for i in idx])
    md = metadata(seed, split, n_samples, config, "natural")
    md["dataset"] = "MNIST official train/test; split from official train for validation; padding is optional"
    md["source_indices"] = idx.tolist()
    md["official_partition"] = "test" if split == "test" else "train"
    return images, labels, md
