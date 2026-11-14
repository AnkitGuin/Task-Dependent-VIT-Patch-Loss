"""Independent seeded randomness; runs do not rely on mutable global NumPy RNG."""

import random
import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def split_seed(seed: int, split: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(f"patchloss-v1:{seed}:{split}".encode()).digest()[:4], "big")
