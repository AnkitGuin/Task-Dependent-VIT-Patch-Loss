"""Run the content-clean, full-test revision cohort, offline after asset preparation."""

import argparse
from importlib.metadata import version
import platform
from pathlib import Path
import sys
import torch
import yaml
from threadpoolctl import threadpool_limits
from patchloss.selection.revision import run_revision_unit, write_json
from patchloss.selection.revision_data import load_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_revision.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config["threads"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(platform=platform.platform(), python=sys.version, processor=platform.processor(),
                       dependencies={k: version(k) for k in ["numpy", "scipy", "torch", "torchvision", "scikit-learn", "psutil"]})
    write_json(output / "environment.json", environment)
    with threadpool_limits(config["threads"]):
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                data, metadata = load_revision(dataset, seed, config)
                print(f"DATA {dataset} s{seed}: {metadata['deduplication']}", flush=True)
                for budget in config["budgets"]:
                    run_revision_unit(config, dataset, seed, budget, data, metadata)
                # Complete full-test arrays need not survive into the next seed.
                del data


if __name__ == "__main__":
    main()
