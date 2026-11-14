"""Integrity checks and explicit selection of comparable, completed source runs."""

import hashlib
import json
import math
from pathlib import Path
import yaml
from ..config import config_hash


def read_record(directory):
    directory = Path(directory)
    md = json.loads((directory / "metadata.json").read_text())
    if md.get("status") != "completed":
        raise ValueError(f"Run status is {md.get('status')!r}")
    required = {"config.yaml", "metrics.json", "history.csv"}
    hashes = md.get("artifact_sha256", {})
    if required - hashes.keys():
        raise ValueError("Missing required artifact hashes")
    for name, expected in hashes.items():
        target = directory / name
        if Path(name).name != name or not target.is_file():
            raise ValueError(f"Missing or invalid artifact: {name}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    config = yaml.safe_load((directory / "config.yaml").read_text())
    if config_hash(config) != md.get("config_hash"):
        raise ValueError("Configuration fingerprint mismatch")
    data = json.loads((directory / "metrics.json").read_text())

    def check_finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite metric in completed run")
        if isinstance(value, dict):
            for item in value.values():
                check_finite(item)
        elif isinstance(value, list):
            for item in value:
                check_finite(item)

    check_finite(data)
    if md.get("run_id") != directory.name or config.get("experiment") != md.get("experiment_id"):
        raise ValueError("Run/configuration identity mismatch")
    metrics = data.get("metrics", {})
    if (
        metrics.get("status") != "completed"
        or metrics.get("experiment_id") != md["experiment_id"]
        or metrics.get("seed") != md.get("seed")
    ):
        raise ValueError("Metrics/metadata identity mismatch")
    if not isinstance(data.get("rows"), list):
        raise ValueError("Missing result rows")
    if md.get("source_sha256"):
        snapshot = json.loads((directory / "source_snapshot.json").read_text())
        found = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
        if found != md["source_sha256"]:
            raise ValueError("Source fingerprint mismatch")
    return md, data


def select_latest(df):
    """Retain one repeat per condition/seed; reject pooling incompatible conditions.

    Experiment IDs are coarse labels, not configuration identities. Figures
    currently have no separate panels for multiple configs or implementations.
    Require the caller to aggregate a deliberate cohort instead of blending it.
    """
    if df.empty or "run_id" not in df:
        return df.copy()
    for key, fallback in (("config_hash", "legacy"), ("source_sha256", "legacy")):
        if key not in df:
            df = df.assign(**{key: fallback})
        else:
            df = df.copy()
            df[key] = df[key].fillna(fallback)
    identities = ["experiment_id", "config_hash", "source_sha256"]
    cohorts = df[identities].drop_duplicates().groupby("experiment_id").size()
    if (cohorts > 1).any():
        raise ValueError(
            "Multiple configurations/source versions for "
            + ", ".join(cohorts[cohorts > 1].index)
            + "; aggregate a single explicit cohort before plotting or generating a paper"
        )
    latest = df.groupby(identities + ["seed"], dropna=False)["run_id"].max()
    return df[df.run_id.isin(latest.values)].copy()
