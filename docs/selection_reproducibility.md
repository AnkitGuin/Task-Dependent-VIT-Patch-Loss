# Reproducing the refined tokenizer-selection study

Use Python 3.11+, CPU PyTorch and the scientific dependencies. Install the optional torchvision dependency for this study:

```bash
python -m pip install -e '.[dev,natural]'
python scripts/download_selection_assets.py --config configs/selection_study.yaml
python -m pytest -q
ruff check .
python scripts/run_selection_study.py --config configs/selection_study.yaml --smoke
python scripts/run_selection_study.py --config configs/selection_study.yaml
python scripts/run_selection_controls.py --output runs/selection_controls --data-root data/selection_local
python scripts/verify_selection_study.py --config configs/selection_study.yaml --output reports/selection/checkpoint_audit.json
python scripts/run_selection_sensitivity.py --config configs/selection_study.yaml --output runs/selection_sensitivity
python scripts/run_logme_convergence.py --config configs/selection_study.yaml --output runs/selection_logme_profile
python scripts/run_selection_permutation.py --config configs/selection_study.yaml
python scripts/run_selection_regularization.py --config configs/selection_study.yaml
python scripts/run_selection_probe_grid.py --config configs/selection_study.yaml
python scripts/analyze_selection_study.py --config configs/selection_study.yaml --controls runs/selection_controls --output reports/selection --paper paper
python scripts/analyze_selection_sensitivity.py --config configs/selection_study.yaml --output reports/selection --paper paper
python scripts/analyze_selection_followups.py --config configs/selection_study.yaml
python scripts/verify_selection_regularization.py --config configs/selection_study.yaml
```

These commands describe a fresh reproduction. Existing completed primary units resume only if their immutable configuration/source hashes match; choose a new output cohort after changing code/configuration. Controls intentionally require a new empty output directory. Smoke results are separate and never count as primary evidence. Preserve the frozen primary protocol and label later changes exploratory.

The original preparation used a checksum-verifiable mirror because the publisher host was slow. A parallel byte-range helper is included for the archives, and all files must pass the official CIFAR MD5 checksums before extraction. A local sandbox-access issue required re-extracting the same verified archives into `data/selection_local/`; this changed cache location, not images or partitions. Public pretrained weights have recorded SHA256 fingerprints.

All candidate heads are regularized linear least-squares classifiers solved by eigendecomposition, with validation-selected penalties. Equal scalar budget is not equal full-Transformer compute. Separate within-p comparisons match dense projection dimensions; marginal diagnostic timing excludes shared feature extraction/statistics.

The primary dataset/seed/budget units save exact official indices, data hashes, package-source snapshots, operator matrices, selected heads, complete validation curves, pretest choices and test predictions. `verify_selection_study.py` reconstructs all selected official subsets and exactly replays all 280 heads. It also records exact pixel overlap without silently changing the official protocol.

The original ridge grid ended at 1,000, where 264/280 validation paths were still improving relative to 100. A separate expanded-grid cohort reaches 100,000,000; primary operators and diagnostic choices remain fixed and new penalties are selected on validation data only. A separate diagnostic-probe grid fits/tunes within the same diagnostic label pool. Their protocols and outcomes are explicitly post-primary. Expanded heads are replayed independently, including raw references. Label-permutation choices are saved before joining test outcomes. Use fresh directories for these follow-ups; completed records are immutable.

For local compilation, use the existing Tectonic executable with its workspace resource cache or a suitable LaTeX/BibTeX installation. The manuscript consumes generated `selection_results.tex`, `selection_tables.tex`, `selection_secondary_results.tex`, `selection_sensitivity.tex`, `selection_followup_results.tex` and `selection_followups.tex`; do not edit measured values by hand. The old paper and its results remain under the archive directory.

The study ran on Windows with Python 3.13.3, NumPy 2.2.5, SciPy 1.15.2, PyTorch 2.9.1+cpu, torchvision 0.24.1+cpu and scikit-learn 1.6.1. It reused existing dependencies rather than testing a fresh full dependency resolution. There is no GPU result or full-encoder natural-image fine-tuning claim.

Build the private evidence archive only after all analysis, manuscript compilation and checks finish:

```bash
python scripts/package_selection_evidence.py --output artifacts/selection_evidence_20261114.zip
```

The script never uploads. It includes an individual-file SHA256 manifest, checks ZIP CRCs and every archived file hash, and writes a sibling archive checksum/verification JSON. It refuses to overwrite an existing bundle. Dataset images, full pretrained weights, environments, caches and Git history are excluded. This is a private reproducibility bundle, not an archival public release or licensing clearance.
