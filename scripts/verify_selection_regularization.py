"""Independently replay the expanded-grid heads on regenerated official test subsets."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import yaml
from patchloss.reporting.records import read_record
from patchloss.selection.data import load_cifar
from patchloss.selection.linear import predict
from patchloss.selection.operators import transform


def verify(config, output):
    rows = []
    root = Path("runs/selection_regularization")
    with threadpool_limits(config.get("threads", 2)):
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                data, data_metadata = load_cifar(dataset, seed, config)
                images, labels = data["test"]
                for budget in config["budgets"]:
                    name = f"E11-{dataset}-b{budget}-s{seed}"
                    directory = root / name
                    original = Path(config["output"]) / name
                    _, primary = read_record(original)
                    metadata = json.loads((directory / "metadata.json").read_text())
                    assert metadata["status"] == "completed"
                    assert metadata["datasets"] == data_metadata
                    for filename, expected in metadata["artifact_sha256"].items():
                        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
                    metrics = json.loads((directory / "metrics.json").read_text())
                    patch_sizes = {r["candidate"]: r["patch_size"] for r in primary["rows"]}
                    with np.load(directory / "predictions.npz") as saved:
                        np.testing.assert_array_equal(saved["labels"], labels)
                        for row in metrics["rows"]:
                            candidate_name = row["candidate"]
                            with np.load(original / f"operator_{candidate_name}.npz") as operator:
                                candidate = {
                                    "matrix": operator["matrix"],
                                    "patch_size": patch_sizes[candidate_name],
                                }
                            with np.load(directory / f"head_{candidate_name}.npz") as head:
                                model = dict(head)
                            prediction = predict(model, transform(images, candidate))
                            np.testing.assert_array_equal(prediction, saved[candidate_name])
                            assert np.mean(prediction == labels) == row["test_accuracy"]
                            rows.append(
                                {"run_id": name, "candidate": candidate_name, "predictions_match": True}
                            )
                        with np.load(directory / "head_raw_reference.npz") as head:
                            raw = predict(dict(head), images.reshape(len(images), -1))
                        np.testing.assert_array_equal(raw, saved["raw_reference"])
                        assert np.mean(raw == labels) == metadata["raw_reference_accuracy"]
                    print(f"EXPANDED HEAD REPLAY VERIFIED {name}", flush=True)
    assert len(rows) == 280
    output.write_text(
        json.dumps(
            {
                "checked_checkpoints": len(rows),
                "checked_raw_references": 20,
                "all_split_hashes_match": True,
                "records": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/selection/expanded_checkpoint_audit.json")
    )
    args = parser.parse_args()
    verify(yaml.safe_load(Path(args.config).read_text()), args.output)
