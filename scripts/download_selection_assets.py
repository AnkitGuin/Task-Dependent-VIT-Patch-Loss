"""Download only the official datasets/weights explicitly enabled by the study config."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ViT_B_16_Weights
import yaml


def prepare(config):
    if not config.get("allow_download", False):
        raise RuntimeError("Downloads require allow_download: true")
    root = Path(config["data_root"])
    root.mkdir(parents=True, exist_ok=True)

    def dataset(name):
        cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[name]
        # The publisher URL is upgraded to HTTPS; torchvision still verifies MD5.
        # Mirror distributes the byte-identical official archives. The publisher's
        # MD5 remains mandatory; the Toronto host was too slow in this environment.
        cls.url = "https://dataset.bj.bcebos.com/cifar/" + cls.filename
        cls(root=root, train=True, download=True)
        cls(root=root, train=False, download=True)
        print(f"VERIFIED dataset {name}", flush=True)

    def weight(name, url, hash_prefix):
        target = root / (name + ".pth")
        if not target.exists():
            torch.hub.download_url_to_file(url, str(target), hash_prefix=hash_prefix, progress=False)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if not digest.startswith(hash_prefix):
            raise ValueError(f"Official weight digest mismatch for {name}")
        print(f"VERIFIED weights {name}", flush=True)
        return {"name": name, "url": url, "sha256": digest, "bytes": target.stat().st_size}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(dataset, name) for name in config["datasets"]]
        weights = []
        if config.get("pretrained", True):
            weights.append(
                pool.submit(
                    weight,
                    "deit_tiny",
                    "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                    "a1311bcf",
                )
            )
            weights.append(pool.submit(weight, "vit_b16", ViT_B_16_Weights.IMAGENET1K_V1.url, "c867db91"))
        for future in futures:
            future.result()
        metadata = [future.result() for future in weights]
    (root / "assets.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    args = parser.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()))
