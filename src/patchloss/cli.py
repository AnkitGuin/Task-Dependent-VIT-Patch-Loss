"""Reproducible CLI: doctor, generate, run, suite, aggregate, figures, audit."""

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from .config import load_config
from .experiments.runner import run_experiment
from .theory.numerical_checks import verify


def doctor():
    modules = ["numpy", "scipy", "torch", "sklearn", "pandas", "matplotlib", "yaml", "pytest", "torchvision"]
    print(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "mps": torch.backends.mps.is_available(),
                "optional_mnist_requires_download": True,
                "modules": {name: bool(importlib.util.find_spec(name)) for name in modules},
            },
            indent=2,
        )
    )


def config_files(suite):
    names = [
        "smoke",
        "exact_collision",
        "gaussian_alignment",
        "task_dependence",
        "xor_task",
        "frequency_sweep",
        "learned_embedding",
        "antialiasing",
        "overlap",
        "tiny_vit",
    ]
    if suite == "smoke":
        names = ["smoke", "exact_collision", "gaussian_alignment", "xor_task", "learned_embedding"]
    elif suite == "core":
        names = names
    elif suite == "full":
        names = names + ["mnist_pilot"]
    else:
        raise ValueError(suite)
    return [Path("configs") / (n + ".yaml") for n in names]


def main(argv=None):
    p = argparse.ArgumentParser(prog="patchloss")
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ["doctor", "test-theory", "generate", "run", "run-suite", "aggregate", "figures", "audit"]:
        q = sub.add_parser(cmd)
        if cmd in ("test-theory", "generate", "run"):
            q.add_argument("--config", default="configs/smoke.yaml")
        if cmd in ("generate", "run"):
            q.add_argument("--seed", type=int, default=0)
        if cmd == "run":
            q.add_argument("--device", default="cpu")
            q.add_argument("--output", default="runs")
        if cmd == "run-suite":
            q.add_argument("--suite", choices=["smoke", "core", "full"], default="smoke")
            q.add_argument("--device", default="cpu")
            q.add_argument("--seeds", type=int, nargs="+")
            q.add_argument("--output", default="runs", help="Run cohort directory")
            q.add_argument("--report-output", default="reports", help="Reports for this cohort")
        if cmd in ("aggregate", "figures", "audit"):
            q.add_argument(
                "--input", default={"aggregate": "runs", "figures": "reports", "audit": "runs"}[cmd]
            )
            q.add_argument(
                "--output",
                default={
                    "aggregate": "reports",
                    "figures": "reports/figures",
                    "audit": "docs/results_audit.md",
                }[cmd],
            )
    a = p.parse_args(argv)
    if a.command == "doctor":
        doctor()
    elif a.command == "test-theory":
        load_config(a.config)
        print(json.dumps(verify(), indent=2))
    elif a.command == "generate":
        from .data.base import get_generator

        c = load_config(a.config)
        mapping = {
            "E01": "nullspace",
            "E02": "gaussian",
            "E03": "gaussian",
            "E04": "relational",
            "E05": "frequency",
            "E06": "gaussian",
            "E07": "alias_tradeoff",
            "E08": "frequency",
            "E09": "frequency",
            "E10": "natural",
        }
        task = mapping.get(c["experiment"])
        if task is None:
            raise ValueError("No dataset generated for E00")
        x, y, md = get_generator(task)(a.seed, "train", int(c.get("train_samples", 256)), c)
        path = Path("data") / f"{task}_seed{a.seed}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, images=x, labels=y)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "shape": list(x.shape),
                    "class_counts": np.bincount(y).tolist(),
                    "generator_version": md["generator_version"],
                    "parameters_sha256": md["parameter_hash"],
                },
                indent=2,
            )
        )
    elif a.command == "run":
        path, metrics = run_experiment(load_config(a.config), a.seed, a.output, a.device)
        print(json.dumps({"run_dir": str(path), "metrics": metrics}, indent=2))
    elif a.command == "run-suite":
        success = []
        failed = []
        for file in config_files(a.suite):
            c = load_config(file)
            seeds = (
                a.seeds
                if a.seeds is not None
                else ([0, 1, 2] if a.suite == "full" and c["experiment"] in {"E06", "E09"} else [0])
            )
            for seed in seeds:
                if c["experiment"] == "E10" and not c.get("allow_download", False):
                    print("SKIPPED E10: optional download disabled")
                    continue
                print(f"RUN {c['experiment']} seed={seed}", flush=True)
                try:
                    run_path, metrics = run_experiment(c, seed, output=a.output, device=a.device)
                    success.append(str(run_path))
                    print(f"OK {run_path} {metrics['wallclock_sec']:.2f}s", flush=True)
                except Exception as ex:
                    failed.append({"experiment": c["experiment"], "seed": seed, "error": str(ex)})
                    print(f"FAILED {c['experiment']}: {ex}", flush=True)
        from .reporting.aggregate import aggregate
        from .reporting.plot import figures
        from .reporting.tables import create_tables
        from .reporting.audit import audit

        print("AGGREGATE", aggregate(a.output, a.report_output))
        print("FIGURES", figures(a.report_output, Path(a.report_output) / "figures"))
        print("TABLES", create_tables(a.report_output, Path(a.report_output) / "tables"))
        audited = audit(a.output, Path(a.report_output) / "results_audit.md")
        print("AUDIT", audited)
        print(json.dumps({"completed": len(success), "failed": failed}, indent=2))
        if audited["failures"]:
            raise SystemExit(1)
        if failed:
            raise SystemExit(1)
    elif a.command == "aggregate":
        from .reporting.aggregate import aggregate

        print(json.dumps(aggregate(a.input, a.output), indent=2))
    elif a.command == "figures":
        from .reporting.plot import figures

        print(json.dumps(figures(a.input, a.output)))
    elif a.command == "audit":
        from .reporting.audit import audit

        result = audit(a.input, a.output)
        print(json.dumps(result, indent=2))
        if result["failures"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
