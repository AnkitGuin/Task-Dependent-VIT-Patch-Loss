"""Acquire official SVHN cropped train/test files, verifying publisher MD5s."""

import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    root = Path("data/selection_local")
    root.mkdir(parents=True, exist_ok=True)
    assets = []
    for split, expected in [("train", "e26dedcc434d2e4c54c9b2d4a06d8373"),
                            ("test", "eb5a983be6a315427106f1b164d9cef3")]:
        name = f"{split}_32x32.mat"
        target = root / name
        url = "https://ufldl.stanford.edu/housenumbers/" + name
        if not target.exists():
            temporary = target.with_suffix(".download")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"SVHN publisher checksum mismatch: {name}")
            temporary.replace(target)
        content = target.read_bytes()
        if hashlib.md5(content).hexdigest() != expected:
            raise ValueError(f"SVHN publisher checksum mismatch: {name}")
        assets.append(dict(file=name, url=url, md5=expected,
                           sha256=hashlib.sha256(content).hexdigest(), bytes=len(content)))
        print(f"VERIFIED {name}: {len(content)} bytes", flush=True)
    (root / "svhn_assets.json").write_text(json.dumps(assets, indent=2) + "\n")


if __name__ == "__main__":
    main()
