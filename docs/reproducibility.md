# Reproducibility instructions

Use Python 3.11+ in a clean environment with CPU PyTorch and the dependencies listed in `pyproject.toml` (an environment-specific freeze is provided separately). At repository root:

```bash
python -m pip install -e '.[dev]'
python -m patchloss.cli doctor
python -m patchloss.cli test-theory --config configs/smoke.yaml
pytest -q
python -m patchloss.cli run-suite --suite full --device cpu --output runs/review_20261114 --report-output reports/review_20261114
python -m patchloss.cli audit --input runs/review_20261114 --output reports/review_20261114/results_audit.md
python scripts/generate_paper_results.py --input runs/review_20261114 --output paper
python scripts/verify_saved_models.py --input runs/review_20261114 --output reports/review_20261114/checkpoint_audit.json
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The full suite runs three seeds for E06/E09 and only one seed for cheaper studies by default. `--seeds 0 1 2` overrides the choice, expanding every experiment. E10 is intentionally skipped until `configs/mnist_pilot.yaml` has `allow_download: true`; it then downloads MNIST and runs two baseline models. CPU tests do not download.

Each run creates a unique `runs/E..-s..-timestamp-uuid/` with config, metadata, metrics, history and any source arrays. SHA256 values are stored for artifacts; audit detects post-hoc mutation. Note: per-run timestamps and elapsed runtimes are **not** deterministic; numerical outcomes are deterministic where supported by CPU execution. CUDA/MPS may vary, and physical time depends on hardware. Random seeds for train/val/test are independent. Tests include exact round trips, leakage checks, Gaussian oracles, model gradient flow and CLI smoke.

For current evidence, compare `reports/review_20261114/experiments.csv` with its referenced `runs/review_20261114/*/metrics.json` and manifest. Root-level reports and the original 25 runs are historical; their scientific interpretation is qualified by `scientific_review_20261114.md`. Current plotting refuses mixed configs/source versions; use a separate cohort directory for each deliberate implementation/configuration set. Noiseless XOR repeats its four support points across splits, so independent RNGs do not imply unseen-template generalization for that control.

New runs save a package source snapshot and hash, dependency versions, reproduction argv, dataset hashes and selected model checkpoints. The paper generator verifies source artifacts, selects declared configs and writes `paper/results_sources.json`. Run it before recompiling. The local Windows environment can compile using `.tools/tectonic/tectonic.exe paper/main.tex --outdir paper`; its first use fetches TeX resources. Tests and synthetic experiments themselves never require network access.
