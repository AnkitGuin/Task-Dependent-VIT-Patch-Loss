# Current journal revision: 15 November 2026

See docs/REVISION_STATUS.md for final v2 execution and remaining submission requirements. The records below describe earlier work.

# Execution status â€” reproducible handoff

## Completed manuscript refinement - 2026-11-14

The revised manuscript is *Selecting Compressed Patch Tokenizers: An Audit of Task-Alignment Scores under Fixed Feature Budgets*. The prior paper is preserved in `paper/archive_preselection_20261114/`. The new manuscript, code and research records remain private/local; the earlier GitHub code snapshot has not been updated during this refinement.

| Executed work | Verified outcome |
|---|---|
| Offline unit/numerical suite | 68 passed; last full run 207.99 s during concurrent workloads |
| Ruff and Python syntax compilation | Passed after final script edits |
| Primary study | 20 units, 280 selected heads; two datasets, five seeds, two budgets, fourteen candidates |
| Primary checkpoint replay | All 280 predictions, source fingerprints and regenerated data hashes match |
| Supporting controls | Null Beta calibration, anisotropic Gaussian risk, actual pretrained matrix spectra, 30 matched-budget trained checkpoints, true held-out translations |
| Exploratory diagnostics | Label budgets 200/400/1000, adaptive H-score, profile-evidence LogME, fixed and validation-only policies, bank restrictions, exact-duplicate sensitivity |
| Validation-boundary follow-up | Separate 10-penalty grid through 1e8; 280 new heads; 2 select upper boundary, 37 plateau across largest three values |
| Expanded checkpoint replay | All 280 predictions plus 20 raw-reference predictions and regenerated data hashes match |
| Stronger diagnostic baseline | Candidate/penalty validation within the same 1,000 diagnostic labels; choices frozen before outcome joins |
| Label-shuffle negative control | Five repetitions per unit, saved pretest choices; dependent descriptive observations, not independent tasks |
| Final manuscript build | Offline cached Tectonic build; 13 pages, resolved references, zero overfull boxes; nonfatal fontconfig/bibliography-spacing notices retained in validation report |

Primary and follow-up results are preserved separately. Expanded-grid regret is 0.3950 points for covariance retention, 0.4225 for profile-evidence LogME, 0.4275 for the fixed-penalty diagnostic probe, and 0.7425 for mean retention. Paired covariance/probe and covariance/profile-evidence intervals contain zero. Matching patch size changes the leading selector. Publication acceptance, general superiority and first-in-literature novelty are not established.

Exact commands are in `docs/selection_reproducibility.md`. Current findings and submission limits are in `docs/refinement_results.md`. Both checkpoint audits, conditional uncertainty, generated figures/tables and `manuscript_validation.json` are in `reports/selection/`. The final private archive target is `artifacts/selection_evidence_20261114.zip`; its sibling JSON/SHA256 record reports CRC and individual-file SHA256 verification. Original images, full pretrained weights, virtualenv/cache and Git history are excluded.

## Current scientific review â€” 2026-11-14

The historical handoff below describes the original implementation. Current corrections and their evidence are in [`scientific_review_20261114.md`](scientific_review_20261114.md). Key fixes address the multi-base collision distribution, independent token-space Gaussian validation, actual confidence-interval coverage, scale-invariant numerical geometry/TAR, optional MNIST leakage, periodic XOR translations, border/shape validation and result provenance.

Current environment: Windows AMD64, Python 3.13.3, NumPy 2.2.5, PyTorch 2.9.1+cpu, pytest 9.2.0, Ruff 0.17.5. A local `.venv` reuses installed scientific packages; pytest/Ruff were installed into it. This was not a clean dependency-resolution test.

| Command actually executed (local `.venv` Python/Ruff) | Status |
|---|---|
| `python -m pip install -e . --no-build-isolation --no-deps` | SUCCESS |
| `python -m pytest -q` | SUCCESS, 53 passed; final run 58.21 s while the TeX resource setup was also active |
| `ruff check .` | SUCCESS |
| `python -m patchloss.cli test-theory --config configs/smoke.yaml` | SUCCESS, six finite verification categories |
| `python -m patchloss.cli run-suite --suite full --device cpu --output runs/review_20261114 --report-output reports/review_20261114` | SUCCESS, 14 runs, 92 rows, 13 figure sets, five tables, zero failures; E10 skipped |
| `python scripts/generate_paper_results.py --input runs/review_20261114 --output paper` | SUCCESS, macros/tables from 14 unique experiment/seed conditions plus source manifest |
| `python scripts/verify_saved_models.py --input runs/review_20261114 --output reports/review_20261114/checkpoint_audit.json` | SUCCESS, 24 checkpoints reproduce recorded held-out metrics and regenerated dataset hashes |
| `python -m patchloss.cli audit --input runs --output reports/review_20261114/historical_audit.md` | SUCCESS, original 25 historical runs pass artifact checks |
| `python -m compileall -q src tests scripts` | SUCCESS |
| `python -m patchloss.cli doctor` | SUCCESS, CPU environment and dependencies recorded |
| `tectonic main.tex --outdir . --keep-logs --only-cached` from `paper/`, local Tectonic 0.17.0 and workspace cache | SUCCESS, rebuilt 13-page PDF, resolved references, no overfull boxes; nonfatal Fontconfig notice and bibliography spacing warnings remain |
| PDF text/structure and package source snapshot verification | SUCCESS, zero unresolved `??` markers, expected corrected values present; all 14 run snapshots match current source bytes |
| `git diff --check` | SUCCESS |

Current run records are under `runs/review_20261114/`. Reports/figures and the corrected checksum audit are under `reports/review_20261114/`; historical root-level reports remain intact. The manuscript now uses corrected figures and results. Source snapshots distinguish the reviewed package from historical results, and paper selection rejects mixed implementation/configuration cohorts.

The corrected evidence archive is `reports/review_20261114/source_evidence.zip` (168 files, 1,681,811 bytes), with its SHA256 in `source_evidence.sha256`. It contains the raw run records, checkpoints, matching package sources, configs and replay scripts so the otherwise Git-ignored runs can be retained. ZIP CRC checks passed. PDF verification details and hash are in `reports/review_20261114/paper_validation.json`.

**Unrun/limited:** E10 training/download, GPU execution, clean dependency installation, compute-matched architectural study and systematic novelty review. The original six shell scripts had pre-existing mode changes which were preserved. Publication readiness is not asserted. The paper compiler/resources required network setup once; the final cached rebuild, tests and synthetic experiments operate offline.

## Historical handoff â€” 2026-11-07

**Date:** 2026-11-07. **Repository:** `task-dependent-vit-patch-loss`. **Execution environment:** Linux x86_64, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5. No GPU or internet-enabled dataset download was used. Source generator version: `patchloss-generators-v1`.

## Completed deliverables

- Repository, installable Python package, config validation, random-seed utilities, versioned S1â€“S5 generators, T0â€“T7 and optional T8 stem components, classical probes, 2-block tiny ViT, CNN, independent theoretical checker, E00â€“E09 runner, optional E10 code path.
- `paper/theory.tex` contains complete assumptions and proofs for raw patch isometry, shared projection rank, fixed-projection label collision, exact binary equal-prior isotropic Gaussian risk, and a data-processing discussion. `docs/math_notes.md` gives readable derivations. Finite-dimensional reference-matrix checks are implemented and tested.
- **33 mathematical/software tests passed**, including operator equivalence, exact bounded pair collisions, Gaussian Monte Carlo tests, class-mean XOR negative control, seed/split invariants, frozen vs trainable weight change, validation-only model selection and source run traceability.
- E00â€“E09 executed on CPU. Two suites were run (core/full) and one post-registered E04 extension; **25 completed run records, zero failures**. Full suite contains **3 independent seeds each** for E06 and E09. Earlier repeated seed-0 core/full runs are kept as verification and never misreported as additional independent seeds.
- `reports/experiments.csv` (169 measured rows, including duplicates), `reports/experiment_manifest.json`, source per-run JSON/histories, 13 reproducible figure sets (PNG and vector PDF), `reports/tables/`, and a source SHA256 audit.
- `paper/main.tex` and `paper/theory.tex` compiled to an **12-page** PDF with verified six primary machine-readable scholarly references plus a standard text. Numeric macros and tables are generated programmatically from actual completed run JSON.
- `README.md`, detailed methodology, literature/novelty matrix, predeclared experiment protocol, hypotheses, limitations, and decisions are included.

## Exact commands attempted and statuses

| Command | Observed status |
|---|---|
| `PYTHONPATH=src python -m patchloss.cli doctor` | SUCCESS |
| `PYTHONPATH=src python -m patchloss.cli test-theory --config configs/smoke.yaml` | SUCCESS |
| `PYTHONPATH=src pytest -q` | SUCCESS, **33 passed** in ~9 s |
| `PYTHONPATH=src python -m patchloss.cli run-suite --suite core --device cpu` | SUCCESS, 10 experiments seed 0 |
| `PYTHONPATH=src python -m patchloss.cli run-suite --suite full --device cpu` | SUCCESS, 14 source experiments (including 3-seed E06/E09); E10 explicitly SKIPPED |
| `PYTHONPATH=src python -m patchloss.cli run --config configs/xor_task.yaml --seed 0` | SUCCESS, post-registered E04 marker control |
| `PYTHONPATH=src python -m patchloss.cli aggregate` | SUCCESS, 169 source rows; E10 NOT RUN |
| `PYTHONPATH=src python -m patchloss.cli figures` | SUCCESS, 13 figure names and PDF/PNG exports |
| `PYTHONPATH=src python -m patchloss.cli audit` | SUCCESS, 25 completed runs, no mismatches |
| `python scripts/generate_paper_results.py` | SUCCESS, 14 unique `(experiment, seed)` combinations |
| `make paper` | SUCCESS, 12 A4 pages; citations resolved using `bibtex8` |
| `python -m compileall -q src tests` | SUCCESS |
| `python -m pip install -e . --no-build-isolation --no-deps` | SUCCESS, editable install in preprovisioned offline environment |
| `ruff check .` | **NOT RUN**: Ruff absent; attempt to install failed because package index DNS was inaccessible |
| E10 optional MNIST pilot | **NOT RUN**: network download explicitly disabled, never claim natural-image results |

## Selected measured numerical findings

- E00: maximal absolute reconstruction error 0.0.
- E01: paired held-out frozen-token probe 0.5000, token residual stored as zero at measured precision, Bayes token risk 0.5 versus raw-image risk 0 on the designed two-point task.
- E02: 30 independent conditions per seed, mean absolute analytic/Monte Carlo gap 0.00269; maximum 0.01168; 0 outside configured approximate 95%-tolerance cells in the displayed seed.
- E03: at fixed rank and Ïƒ=0.4, null-aligned risk 0.5 versus full-row aligned risk 0.10565.
- E04: mean-matched XOR mean TAR undefined, linear probe 0.5, nonlinear MLP 1.0; exploratory mean-shift marker variant mean TAR ~0.5868 and linear probe 1.0.
- E06 learned accuracy by seed [0.8320, 0.8320, 0.8086] versus frozen [0.5156, 0.4453, 0.5234]; means 0.8242 versus 0.4948.
- E07 checkerboard-label family raw 1.0 versus smoothed 0.4531; nuisance family both 1.0 (no evidence of improvement).
- E08 nonoverlap versus overlap operator ranks 32 versus 64 but scalar output budgets 32 versus 72 and both probe accuracies saturated; no matched-budget gain.
- E09 learned/frozen tiny ViT and small CNN were all actually trained for three seeds on frequency/XOR; detailed per-model outcomes in source JSON and compiled paper.

## Outstanding limits and exact next steps

- **Optional pilot not run:** edit `configs/mnist_pilot.yaml` to set `allow_download: true` on a network-enabled system, then run `python -m patchloss.cli run --config configs/mnist_pilot.yaml --seed 0`. Do not report MNIST findings without running it.
- **Ruff not verified:** install `ruff` in a network-enabled environment, run `ruff check .` and address any diagnostics. Syntax compilation and 33 tests passed, but lint equivalence is not asserted.
- **Not validated:** full p=2 attention sweeps, compute-matched overlap, natural-image transfer, systematic 2026 novelty search, GPU reproducibility and statistical significance over large seed groups. Consult `docs/limitations.md`.
- To rerun everything currently feasible, use `python -m patchloss.cli run-suite --suite full --device cpu` and then regenerate aggregate, figures, audit, LaTeX macros and PDF. The source `runs/` directories are included in the final ZIP but ignored by Git as intended; they were generated **before** initializing the final Git repository, so run-level `git_commit` is null rather than falsely claiming an earlier revision.

## Scientific assessment

**Strongest result:** rank-limited tokenization can exactly erase a task, with the Gaussian risk gap quantitatively explained by row-space alignment and independently checked. **Strongest negative result:** nonlinear XOR can survive tokenization even when class-mean TAR is undefined and a short-trained ViT fails to learn; thus first-moment diagnostics and optimization outcomes are not general information-theoretic certificates. Neither is a claim of mathematical novelty or performance on modern pretrained ViTs.
