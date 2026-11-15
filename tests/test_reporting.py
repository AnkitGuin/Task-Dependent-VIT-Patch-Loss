import json
import pandas as pd
from patchloss.reporting.aggregate import aggregate
from patchloss.reporting.audit import audit
from patchloss.reporting.plot import figures


def test_missing_failed_runs_not_zero(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    bad = runs / "bad"
    bad.mkdir()
    (bad / "metadata.json").write_text(json.dumps({"status": "failed", "run_id": "bad"}))
    out = tmp_path / "reports"
    result = aggregate(runs, out)
    assert result["runs"] == 0
    assert "E02" in result["not_run"]
    df = pd.read_csv(out / "experiments.csv") if (out / "experiments.csv").stat().st_size else None
    assert df is None or df.empty
    assert "NOT RUN" in (out / "summary.md").read_text()
    status = audit(runs, tmp_path / "audit.md")
    assert status["completed"] == 0 and status["failures"]


def test_plot_only_analytic_for_empty_sources(tmp_path):
    (tmp_path / "experiments.csv").write_text("experiment_id,seed\n")
    out = figures(tmp_path, tmp_path / "figures")
    assert "F2_rank_phase_analytic" in out
    assert "F4_gaussian" not in out


def test_manifest_traceability_and_actual_figure(tmp_path):
    from patchloss.experiments.runner import run_experiment

    c = {
        "experiment": "E00",
        "height": 8,
        "width": 8,
        "patch_size": 2,
        "embedding_dim": 4,
        "train_samples": 8,
        "val_samples": 8,
        "test_samples": 8,
    }
    run, m = run_experiment(c, seed=5, output=tmp_path / "runs")
    aggregate(tmp_path / "runs", tmp_path / "reports")
    entry = json.loads((tmp_path / "reports/experiment_manifest.json").read_text())["records"][0]
    record = json.loads((run / "metrics.json").read_text())
    assert entry["run_id"] == run.name
    assert entry["source"] == str(run / "metrics.json")
    assert record["rows"][entry["row_index"]]["max_abs_roundtrip"] == 0
    result = figures(tmp_path / "reports", tmp_path / "reports/figures")
    assert "F2_rank_phase_analytic" in result
    assert (tmp_path / "reports/figures/F2_rank_phase_analytic.pdf").exists()
    status = audit(tmp_path / "runs", tmp_path / "audit.md")
    assert status["completed"] == 1 and not status["failures"]
