# Task-Dependent Information Loss in Vision Transformer Patchification

A CPU-friendly research implementation that separates reversible patch extraction, linear projection bottlenecks, task-signal alignment, filtering, overlap, and model optimization.

**Research software under development; not peer-reviewed.** This repository contains code, tests, experiment configurations and technical documentation. The unpublished manuscript, generated figures/tables, recorded runs, model weights and evidence archives are intentionally withheld. Running the experiments generates artifacts locally.

## Scientific scope

- Ordered non-overlapping patch extraction preserves every pixel and is invertible.
- A fixed rank-deficient projection discards directions; whether those directions matter depends on the task.
- For an equal-prior isotropic Gaussian location model, projected Bayes risk depends on the retained class contrast. This does not characterize arbitrary image distributions or nonlinear transformers.
- A learned projection can change its row space; a frozen-map impossibility result does not automatically apply after training.
- Probe or neural-network failure alone does not prove irrecoverable information loss.

The code includes bounded null-space collision tasks, Gaussian alignment controls, gratings and markers, relational XOR, low-pass comparisons, overlapping tokenization, tiny ViTs and a CNN baseline. These are controlled synthetic studies, not natural-image or state-of-the-art benchmarks. No novelty claim is made for standard linear algebra, Gaussian discrimination, smaller patches, overlap or convolutional stems.

## Installation and checks

Use Python 3.11 or newer with a working PyTorch installation. From the repository root:

```bash
python -m pip install -e '.[dev]'
python -m patchloss.cli doctor
python -m patchloss.cli test-theory --config configs/smoke.yaml
python -m pytest -q
ruff check .
```

The reviewed implementation passed 53 tests and Ruff locally. Tests and the core synthetic experiments do not download data. The optional MNIST pilot requires the `natural` dependency extra and an explicit `allow_download: true` setting; it remains outside the verified training results.

## Run experiments

```bash
python -m patchloss.cli run --config configs/exact_collision.yaml --seed 0 --output runs/example
python -m patchloss.cli run --config configs/gaussian_alignment.yaml --seed 0 --output runs/example
python -m patchloss.cli run-suite --suite full --device cpu --output runs/my_cohort --report-output reports/my_cohort
python -m patchloss.cli audit --input runs/my_cohort --output reports/my_cohort/results_audit.md
python scripts/verify_saved_models.py --input runs/my_cohort --output reports/my_cohort/checkpoint_audit.json
```

The full suite uses three seeds for E06/E09 and one seed for other synthetic experiments by default. Pass `--seeds 0 1 2` to expand all experiments. Use a new cohort directory after changing code or configuration: figures reject incompatible configuration/source mixtures. E10 is skipped while downloads are disabled.

Each run records its configuration, seed, environment, source and dataset fingerprints, completion status, metrics, histories and artifact hashes. Trained experiments save validation-selected checkpoints. The replay script reloads E06/E09 checkpoints and compares their held-out metrics and regenerated data hashes with the original records. Generated files are ignored by Git.

## Layout

```text
configs/                  Experiment configurations
src/patchloss/
  data/                   Synthetic generators and optional MNIST
  operators/              Patch extraction, projections, overlap, low-pass
  theory/                 Rank, null spaces, Gaussian risk and mean TAR
  models/                 Probes, tiny ViT, CNN and tokenizer stems
  training/               Training, evaluation and checkpoints
  experiments/            E00–E10 experiment runner
  reporting/              Integrity checks, aggregation, figures and tables
tests/                    Mathematical and software regression checks
scripts/                  Experiment, reporting and checkpoint replay helpers
docs/                     Methods, scope, reproducibility and prior art
```

See [mathematical notes](docs/math_notes.md), [methods](docs/methodology.md), [code-review corrections](docs/code_review.md), [reproduction instructions](docs/reproducibility.md), [limitations](docs/limitations.md) and [literature review](docs/literature_review.md). Broader empirical validation and a systematic novelty review remain necessary before publication.
