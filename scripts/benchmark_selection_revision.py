"""Fresh-process, two-thread end-to-end selector cost measurements."""

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from threadpoolctl import threadpool_limits
from patchloss.selection.revision import METHODS, score_bank, write_json
from patchloss.selection.revision_data import load_revision
from patchloss.selection.operators import dct_rows, fit_bank


def record_hardware(config, output):
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    write_json(output/"hardware.json",dict(cpu=cpu,physical_cores=psutil.cpu_count(logical=False),
               logical_processors=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,
               platform=platform.platform(),numerical_threads=config["threads"],
               benchmark_isolated_from_project_experiments=True))


def worker(args, config):
    unit = Path(config["output"]) / f"{args.dataset}-b{args.budget}-s{config['seeds'][0]}"
    pretest = json.loads((unit / "pretest.json").read_text())
    data = {p: (np.load(args.input / f"{p}.npy", mmap_mode="c"), np.load(args.input / f"{p}_labels.npy"))
            for p in ["operator", "diagnostic"]}
    bank = []
    for row in pretest["rows"]:
        if args.method == "fixed_dct" and row["candidate"] != config["fixed_candidate"]:
            continue
        bank.append({k: v for k, v in row.items() if k not in {"scores", "details"}} |
                    {"matrix": None})
    folds = {k: np.array(v) for k, v in json.loads((unit / "diagnostic_folds.json").read_text()).items()}
    torch.set_num_threads(config["threads"])
    with threadpool_limits(config["threads"]):
        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss
        peak = [baseline]
        stop = threading.Event()
        def sample():
            while not stop.wait(.01):
                peak[0] = max(peak[0], process.memory_info().rss)
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        begin = time.perf_counter()
        if args.method == "fixed_dct":
            candidate = bank[0]
            candidate["matrix"] = dct_rows(4, 3, args.budget // 64)
        else:
            bank = fit_bank(data["operator"][0], args.budget, config["patch_sizes"], config["seeds"][0], config)
        bank_sec = time.perf_counter() - begin
        score_begin = time.perf_counter()
        values, _ = score_bank(args.method, bank, data, config, folds)
        elapsed = time.perf_counter() - score_begin
        cold = time.perf_counter() - begin
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        monitor.join()
    expected = {r["candidate"]: r["scores"][args.method] for r in pretest["rows"]}
    for cid, value in values.items():
        if value is None:
            assert expected[cid] is None
        else:
            np.testing.assert_allclose(value, expected[cid], rtol=1e-9, atol=1e-10)
    result = dict(dataset=args.dataset, budget=args.budget, metric=args.method, repetition=args.repetition,
                  feature_statistics_score_sec=elapsed, shared_bank_fit_sec=bank_sec,
                  cold_selection_sec=cold, baseline_rss_mib=baseline/2**20,
                  peak_rss_mib=peak[0]/2**20, incremental_peak_rss_mib=(peak[0]-baseline)/2**20,
                  score_replay_verified=True)
    write_json(args.result, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_revision.yaml")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--method")
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.worker:
        return worker(args, config)
    root = Path("reports/selection_revision/cost")
    root.mkdir(parents=True, exist_ok=True)
    record_hardware(config,root.parent)
    for dataset in config["datasets"]:
        data, _ = load_revision(dataset, config["seeds"][0], config)
        inputs = root / dataset
        inputs.mkdir(exist_ok=True)
        for part in ["operator", "diagnostic"]:
            np.save(inputs / f"{part}.npy", data[part][0])
            np.save(inputs / f"{part}_labels.npy", data[part][1])
        del data
        for budget in config["budgets"]:
            for method in METHODS:
                for rep in range(3):
                    target = root / f"{dataset}-b{budget}-{method}-{rep}.json"
                    if target.exists():
                        continue
                    subprocess.run([sys.executable, __file__, "--worker", "--config", args.config,
                                    "--dataset", dataset, "--budget", str(budget), "--method", method,
                                    "--repetition", str(rep), "--input", str(inputs), "--result", str(target)], check=True)
                print(f"COST {dataset} b{budget} {method}", flush=True)
    records = [json.loads(p.read_text()) for p in root.glob("*.json")]
    pd.DataFrame(records).to_csv(root.parent / "cost_measurements.csv", index=False)


if __name__ == "__main__":
    main()
