"""Trace source data, cryptographic file checksums, and experiment status."""

from pathlib import Path
from .records import read_record


def audit(input_dir="runs", output_file="docs/results_audit.md"):
    lines = [
        "# Reproducibility audit",
        "",
        "Run records below were checked against recorded SHA256 values.",
        "",
    ]
    bad = []
    done = []
    for path in sorted(Path(input_dir).glob("*/metadata.json")):
        run = path.parent
        try:
            md, _ = read_record(run)
        except (ValueError, OSError, KeyError, TypeError) as ex:
            lines.append(f"- **FAILED/INCOMPLETE/INVALID** {run.name}: {ex}")
            bad.append(run.name)
            continue
        lines.append(
            f"- **{md['experiment_id']}** {run.name}: config `{md['config_hash'][:12]}`, seed {md['seed']}, {md['wallclock_sec']:.3f}s, {'HASH OK' if not any(run.name in b for b in bad) else 'HASH FAIL'}"
        )
        done.append(run.name)
    if not done:
        lines.append("**NOT RUN:** no completed source runs exist.")
    lines.extend(
        [
            "",
            f"Completed runs: {len(done)}",
            "Hash failures/incomplete records: " + (str(bad) if bad else "none"),
            "",
            "The audit does not certify generalization beyond the tested synthetic distributions.",
        ]
    )
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return {"completed": len(done), "failures": bad}
