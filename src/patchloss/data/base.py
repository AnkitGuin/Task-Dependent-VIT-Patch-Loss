"""Generator contract with independent split-specific RNG and hashed parameters."""

import hashlib
import json
import numpy as np
from ..seed import split_seed

GENERATOR_VERSION = "patchloss-generators-v2"


def rng_for(seed, split):
    if split not in {"train", "validation", "test", "analysis"}:
        raise ValueError(f"Unknown split {split}")
    return np.random.default_rng(split_seed(seed, split))


def metadata(seed, split, n, config, kind):
    return {
        "generator_version": GENERATOR_VERSION,
        "task": kind,
        "seed": int(seed),
        "split_seed": split_seed(seed, split),
        "split": split,
        "n_samples": int(n),
        "parameter_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "label_equation": {
            "nullspace": "Y=sign of fixed null-space perturbation",
            "gaussian": "X|Y=y~N(yv/2,sigma²I)",
            "frequency": "Y selects grating orientation or marker",
            "relational": "Y=bit1 XOR bit2",
            "alias_tradeoff": "Y determines low or high frequency sign",
        }.get(kind, kind),
    }


def balanced_binary(n, rng):
    arr = np.arange(n) % 2
    rng.shuffle(arr)
    return arr.astype(np.int64)


def get_generator(kind):
    from . import nullspace, gaussian, frequency, relational, alias_tradeoff, natural

    return {
        "nullspace": nullspace.generate,
        "gaussian": gaussian.generate,
        "frequency": frequency.generate,
        "relational": relational.generate,
        "alias_tradeoff": alias_tradeoff.generate,
        "natural": natural.generate,
    }[kind]
