"""Replay every saved primary head/operator and check leakage/provenance boundaries."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
from patchloss.reporting.records import read_record
from patchloss.selection.data import load_cifar
from patchloss.selection.operators import transform
from patchloss.selection.linear import predict
from patchloss.selection.statistics import freeze_selection
from patchloss.selection.study import source_snapshot
from patchloss.seed import split_seed


def verify(config, output):
    _, current_source = source_snapshot()
    checked = []
    duplicates = []
    with threadpool_limits(config.get("threads", 2)):
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                data, md = load_cifar(dataset, seed, config)
                # Index partitions are already disjoint; also disclose identical
                # pixel images in the official partitions rather than assuming none.
                hashes = {
                    name: [hashlib.sha256(image.tobytes()).hexdigest() for image in x]
                    for name, (x, _) in data.items()
                }
                for left, right in [
                    ("operator", "diagnostic"),
                    ("operator", "validation"),
                    ("diagnostic", "validation"),
                    ("operator", "test"),
                    ("diagnostic", "test"),
                    ("validation", "test"),
                ]:
                    overlap = set(hashes[left]) & set(hashes[right])
                    duplicates.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "left": left,
                            "right": right,
                            "identical_image_hashes": len(overlap),
                        }
                    )
                for budget in config["budgets"]:
                    directory = Path(config["output"]) / f"E11-{dataset}-b{budget}-s{seed}"
                    metadata, record = read_record(directory)
                    if metadata["source_sha256"] != current_source:
                        raise ValueError("Current package source differs from recorded study")
                    if json.loads((directory / "datasets.json").read_text()) != md:
                        raise ValueError("Regenerated dataset indices/hash metadata differ")
                    pretest = json.loads((directory / "pretest_selection.json").read_text())
                    recomputed = freeze_selection(
                        pretest["rows"], split_seed(seed, dataset + f":budget{budget}:ties")
                    )
                    if recomputed != pretest["selection"]:
                        raise ValueError("Score-based selection does not replay")
                    with np.load(directory / "predictions.npz") as saved:
                        np.testing.assert_array_equal(saved["labels"], data["test"][1])
                        for row in record["rows"]:
                            name = row["candidate"]
                            with np.load(directory / f"operator_{name}.npz") as stored:
                                candidate = {"matrix": stored["matrix"], "patch_size": row["patch_size"]}
                            with np.load(directory / f"head_{name}.npz") as stored:
                                model = {key: stored[key] for key in stored.files}
                            predictions = predict(model, transform(data["test"][0], candidate))
                            np.testing.assert_array_equal(predictions, saved[name])
                            accuracy = float(np.mean(predictions == saved["labels"]))
                            if accuracy != row["test_accuracy"]:
                                raise ValueError("Recorded accuracy differs from checkpoint replay")
                            checked.append(
                                {
                                    "run_id": directory.name,
                                    "candidate": name,
                                    "accuracy": accuracy,
                                    "predictions_match": True,
                                    "data_hashes_match": True,
                                }
                            )
                    print(f"REPLAY VERIFIED {directory.name}", flush=True)
    result = {
        "checked_checkpoints": len(checked),
        "source_sha256": current_source,
        "records": checked,
        "exact_duplicate_image_audit": duplicates,
        "duplicate_policy": "Official partitions retained as prospectively specified; counts disclosed, no test-driven exclusions",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(checked)} primary checkpoints and all regenerated split hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--output", type=Path, default=Path("reports/selection/checkpoint_audit.json"))
    args = parser.parse_args()
    verify(yaml.safe_load(Path(args.config).read_text()), args.output)
