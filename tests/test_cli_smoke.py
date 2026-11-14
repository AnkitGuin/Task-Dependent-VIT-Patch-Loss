import subprocess
import sys
import json
from patchloss.experiments.runner import run_experiment


def test_cli_doctor_theory_and_run_smoke(tmp_path):
    cmd = [sys.executable, "-m", "patchloss.cli"]
    ret = subprocess.run(cmd + ["doctor"], capture_output=True, text=True)
    assert ret.returncode == 0, ret.stderr
    assert "torch" in json.loads(ret.stdout)
    ret = subprocess.run(cmd + ["test-theory"], capture_output=True, text=True)
    assert ret.returncode == 0, ret.stderr
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
    path, metrics = run_experiment(c, 0, tmp_path / "runs")
    assert metrics["lossless"]
    assert (path / "metadata.json").exists()
    assert json.loads((path / "metadata.json").read_text())["status"] == "completed"
