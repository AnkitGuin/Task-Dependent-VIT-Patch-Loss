"""Resume byte-range downloads of byte-identical CIFAR archives; enforce official MD5."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import argparse
from pathlib import Path
import urllib.request
import yaml

ROOT = Path("data/selection")
ASSETS = [
    ("cifar-10-python.tar.gz", 170498071, "c58f30108f718f92721af3b95e74349a"),
    ("cifar-100-python.tar.gz", 169001437, "eb9058c3a382ffc7106e4002c42a8d85"),
]


def fetch(task):
    name, start, end = task
    part = ROOT / "range_parts" / name / f"{start:012d}.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    size = end - start + 1
    if part.exists() and part.stat().st_size == size:
        return
    request = urllib.request.Request(
        "https://dataset.bj.bcebos.com/cifar/" + name, headers={"Range": f"bytes={start}-{end}"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                    f"bytes {start}-{end}/"
                ):
                    raise RuntimeError("Server did not honor the exact byte range")
                with part.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
            if part.stat().st_size != size:
                raise RuntimeError("Incomplete byte range")
            return
        except Exception:
            if attempt == 2:
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if not config.get("allow_download", False):
        raise RuntimeError("Downloads require allow_download: true")
    ROOT = Path(config["data_root"])
    ROOT.mkdir(parents=True, exist_ok=True)
    chunk_size = 4 * 1024 * 1024
    tasks = [
        (name, start, min(start + chunk_size, size) - 1)
        for name, size, _ in ASSETS
        for start in range(0, size, chunk_size)
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 8 == 0 or done == len(tasks):
                print(f"RANGES {done}/{len(tasks)} verified", flush=True)
    for name, size, expected in ASSETS:
        target = ROOT / name
        with target.open("wb") as output:
            for start in range(0, size, chunk_size):
                output.write((ROOT / "range_parts" / name / f"{start:012d}.part").read_bytes())
        digest = hashlib.md5(target.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Official archive checksum mismatch: {name}")
        print(f"OFFICIAL MD5 VERIFIED {name}", flush=True)
