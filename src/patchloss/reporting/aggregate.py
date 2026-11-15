"""Aggregate only successful, complete run records; manifest contains provenance."""

import json
from pathlib import Path
import pandas as pd
import yaml
from .records import read_record


def aggregate(input_dir="runs", output_dir="reports"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = []
    failures = []
    for d in sorted(Path(input_dir).glob("*")):
        if not d.is_dir():
            continue
        mp = d / "metadata.json"
        if not mp.exists():
            continue
        try:
            md, data = read_record(d)
        except (ValueError, OSError, KeyError, TypeError) as ex:
            failures.append({"run": d.name, "status": "excluded", "reason": str(ex)})
            continue
        header = {
            "run_id": md["run_id"],
            "experiment_id": md["experiment_id"],
            "seed": md["seed"],
            "config_hash": md["config_hash"],
            "device": md["device"],
            "run_wallclock_sec": md["wallclock_sec"],
            "source_sha256": md.get("source_sha256", "legacy"),
            "source_dir": str(d.resolve()),
        }
        if md["experiment_id"] in {"E02", "E03"}:
            config = yaml.safe_load((d / "config.yaml").read_text())
            header["num_test_samples"] = int(
                config.get("mc_samples", 5000 if md["experiment_id"] == "E02" else 4000)
            )
        if not data.get("rows"):
            records.append({**header, **data["metrics"]})
        for ix, row in enumerate(data["rows"]):
            records.append({**header, "row_index": ix, **row})
            manifest.append(
                {
                    "run_id": md["run_id"],
                    "row_index": ix,
                    "experiment_id": md["experiment_id"],
                    "source": str(d / "metrics.json"),
                    "config_hash": md["config_hash"],
                    "source_sha256": md.get("source_sha256", "legacy"),
                    "seed": md["seed"],
                    "status": "completed",
                }
            )
    pd.DataFrame(records).to_csv(out / "experiments.csv", index=False) if records else (
        out / "experiments.csv"
    ).write_text("experiment_id,seed,run_id\n")
    figure_sources = {
        "F1_operator_schematic": [],
        "F2_rank_phase_analytic": [],
        "F3_exact_collision": ["E01"],
        "F4_gaussian": ["E02"],
        "F4b_task_dependence": ["E03"],
        "F5_frequency_phase": ["E05"],
        "F5b_marker_size": ["E05"],
        "F5c_phase_shift": ["E05"],
        "F6_frozen_vs_learned": ["E06"],
        "F6b_alignment_trajectory": ["E06"],
        "F7_lowpass_tradeoff": ["E07"],
        "F8_mean_TAR_failure": ["E04"],
        "F9_accuracy_parameter_frontier": ["E09"],
    }
    panels = {
        k: [
            {"run_id": r["run_id"], "config_hash": r["config_hash"], "seed": r["seed"]}
            for r in manifest
            if r["experiment_id"] in ids
        ]
        for k, ids in figure_sources.items()
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "records": manifest,
                "figure_panels": panels,
                "analytic_figures": ["F1_operator_schematic", "F2_rank_phase_analytic"],
                "failed_or_skipped": failures,
            },
            indent=2,
        )
    )
    counts = pd.Series([r["experiment_id"] for r in records]).value_counts().to_dict() if records else {}
    summary = [
        "# Measured experiment summary",
        "",
        f"Completed source rows: {len(records)}",
        f"Excluded incomplete/failed runs: {len(failures)}",
        "",
        "Counts below reflect **record rows**, not necessarily independent runs.",
        "",
    ]
    for k in sorted(counts):
        summary.append(f"- {k}: {counts[k]} rows")
    missing = [f"E{i:02d}" for i in range(11) if f"E{i:02d}" not in counts]
    summary.append("")
    summary.append("NOT RUN: " + (", ".join(missing) if missing else "none"))
    summary.append("See `experiments.csv` and `experiment_manifest.json` for actual values.")
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    return {
        "rows": len(records),
        "runs": len({r["run_id"] for r in records}),
        "failures": failures,
        "not_run": missing,
    }
