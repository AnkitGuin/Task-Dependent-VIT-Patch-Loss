"""Execute the predeclared offline tokenizer-selection cohort; downloads are separate."""

import argparse
import json
from pathlib import Path
import yaml
from patchloss.selection.study import run_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        config.update(
            {
                "datasets": ["cifar10"],
                "seeds": [0],
                "budgets": [256],
                "operator_samples": 200,
                "diagnostic_samples": 200,
                "validation_samples": 100,
                "test_samples": 200,
                "pca_patch_cap": 1000,
                "output": "runs/selection_smoke",
            }
        )
    print(json.dumps({"completed_units": run_study(config)}, indent=2))
