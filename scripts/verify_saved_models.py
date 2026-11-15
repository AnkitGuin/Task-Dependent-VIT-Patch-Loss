"""Reload E06/E09 checkpoints and reproduce recorded held-out metrics without training."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from patchloss.config import load_config
from patchloss.data import gaussian, frequency, relational
from patchloss.models.tiny_vit import TinyViT
from patchloss.models.small_cnn import SmallCNN
from patchloss.reporting.records import read_record
from patchloss.training.checkpoint import load_checkpoint
from patchloss.training.evaluate import evaluate


def digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def verify(input_dir):
    torch.set_num_threads(2)
    checked = []
    for path in sorted(Path(input_dir).glob("*/metadata.json")):
        md, data = read_record(path.parent)
        eid = md["experiment_id"]
        if eid not in {"E06", "E09"}:
            continue
        c = load_config(path.parent / "config.yaml")
        seed = md["seed"]
        p, d = c["patch_size"], c["embedding_dim"]
        records = json.loads((path.parent / "datasets.json").read_text())
        for row in data["rows"]:
            if eid == "E06":
                cfg = {**c, "height": p, "width": p, "alpha": 0.0, "sigma": float(c.get("sigma", 0.45))}
                x, y, _ = gaussian.generate(seed, "test", c["test_samples"], cfg)
                model = TinyViT(
                    p,
                    p,
                    p,
                    d=d,
                    heads=c.get("heads", 2),
                    depth=c.get("depth", 2),
                    trainable=not row["frozen"],
                    seed=seed,
                )
                checkpoint = "frozen.pt" if row["frozen"] else "learned.pt"
                task = ""
            else:
                task = row["task"]
                cfg = {**c, "mode": "frequency", "noise": float(c.get("noise", 0.025))}
                generator = frequency.generate if task == "frequency" else relational.generate
                x, y, _ = generator(seed, "test", c["test_samples"], cfg)
                model = (
                    SmallCNN()
                    if row["model"] == "cnn"
                    else TinyViT(
                        c["height"],
                        c["width"],
                        p,
                        d=d,
                        heads=c.get("heads", 2),
                        depth=c.get("depth", 2),
                        seed=seed,
                    )
                )
                checkpoint = f"{task}_{row['model']}.pt"
            source = next(item for item in records if item["split"] == "test" and item["condition"] == task)
            if digest(x) != source["images_sha256"] or digest(y) != source["labels_sha256"]:
                raise ValueError(f"Dataset hash mismatch in {path.parent.name}: {task}")
            load_checkpoint(model, path.parent / checkpoint)
            result = evaluate(model, torch.from_numpy(x).float(), torch.from_numpy(y).long())
            for key in ("accuracy", "balanced_accuracy", "negative_log_likelihood"):
                if not np.isclose(result[key], row[key], rtol=1e-6, atol=1e-7):
                    raise ValueError(f"Checkpoint metric mismatch: {path.parent.name}/{checkpoint}: {key}")
            checked.append(
                {
                    "run_id": path.parent.name,
                    "checkpoint": checkpoint,
                    "dataset_hash_match": True,
                    "metrics_match": True,
                    "accuracy": result["accuracy"],
                }
            )
    if not checked:
        raise ValueError("No E06/E09 checkpoints found")
    return {"checked_checkpoints": len(checked), "records": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Verified {result['checked_checkpoints']} checkpoints and their regenerated held-out dataset hashes"
    )
