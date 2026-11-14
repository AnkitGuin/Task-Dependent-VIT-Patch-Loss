# Reproducibility

This code release intentionally excludes the unpublished manuscript and generated research artifacts. Create results locally from the supplied configurations; no precomputed performance files are bundled.

Use Python 3.11+ and install a suitable PyTorch build, then run from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m patchloss.cli doctor
python -m pytest -q
ruff check .
python -m patchloss.cli test-theory
python -m patchloss.cli run-suite --suite full --device cpu --output runs/my_cohort --report-output reports/my_cohort
python -m patchloss.cli audit --input runs/my_cohort --output reports/my_cohort/results_audit.md
python scripts/verify_saved_models.py --input runs/my_cohort --output reports/my_cohort/checkpoint_audit.json
```

The local review used Python 3.13.3, NumPy 2.2.5 and PyTorch 2.9.1+cpu on Windows, with 53 passing tests. This reused preinstalled scientific dependencies, so it is not a clean dependency-resolution or cross-platform reproducibility certification.

The full suite uses seeds 0, 1 and 2 for the learned projection/model experiments E06/E09, and seed 0 for the other synthetic experiments. `--seeds 0 1 2` overrides that choice. Optional MNIST E10 is disabled by default; enabling it requires the `natural` extra and explicit download opt-in. Tests and synthetic experiments do not download datasets.

Train, validation and test generators use split-specific random seeds. Collision pairs stay together by base template; MNIST reserves a fixed disjoint validation pool. The noiseless XOR control has only four support images and deliberately repeats those support points across splits, so it does not test unseen-template generalization.

Each run saves configuration, metadata, metrics, history, source fingerprints and checksums. Dataset records contain array hashes, class counts and split metadata; selected model state dictionaries are saved for trained experiments. Aggregation and audit validate required artifacts, identities, hashes and finite metrics. The checkpoint replay helper regenerates held-out data and checks saved model metrics without training.

Create a separate cohort directory for every deliberate code/configuration set. Plotting refuses to pool incompatible source/configuration cohorts. Repeated same-seed runs are verification repeats, not independent observations. Timestamps and runtimes vary; exact numerical agreement across dependency versions or accelerators is not guaranteed.
