# Reproduce the final revision (v2)

This guide supersedes the preliminary two-CIFAR instructions for the main paper. Historical cohorts remain readable and are never silently resumed with new source. The final configuration is `configs/selection_revision.yaml`; the locally frozen protocol and explicit pre-final-run amendment are in `docs/selection_revision_protocol.md`.

## Environment and assets

Use CPython 3.13 on Windows for the recorded environment; Python 3.11+ and other operating systems are supported by the package but are not claimed to be independently validated for these new runs. Exact dependency versions, OS and processor are recorded in `runs/selection_revision_final/environment.json`. Install the recorded CPU scientific environment with `requirements-revision-lock.txt`. The CPU PyTorch wheel index is needed for the `+cpu` versions. A clean installation of this new lock file has not been tested in a second environment.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-revision-lock.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
$env:PYTHONPATH='src'
.venv/Scripts/python.exe scripts/download_selection_assets.py --config configs/selection_study.yaml
.venv/Scripts/python.exe scripts/download_revision_svhn.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

The first preparation script obtains the official-checksum CIFAR archives and pretrained DeiT/ViT assets. The main revision uses only DeiT's patch matrix; ViT-B/16 is needed only to rerun the historical rank control. SVHN acquisition uses the official HTTPS Stanford files, checks the publisher's MD5s from torchvision's dataset definition, and stores SHA256/size/URL metadata. SVHN `extra` is unused. Raw images and full weight files are not redistributed in the release. Follow the dataset publishers' terms. Network access is required only for explicit preparation and initial environment/compiler setup. The recorded project virtual environment inherits available system packages; the lock file records the actual imported package versions, not a claim of hermetic isolation.

## Main execution, independent replay and tables

```powershell
.venv/Scripts/python.exe scripts/run_selection_revision.py
.venv/Scripts/python.exe scripts/verify_selection_revision.py
.venv/Scripts/python.exe scripts/benchmark_selection_revision.py
.venv/Scripts/python.exe scripts/analyze_selection_revision.py
```

Run the cost benchmark AFTER training, replay and the initial bootstrap analysis have finished, with no other project experiments running. It executes 198 fresh workers: 11 methods x 3 domains x 2 budgets x 3 repeats, always on seed 10. Imports, input decoding, pretest checkpoint loading, downstream head fitting and test inference are outside its timed selection stage. Required feature extraction and statistics are inside. Each worker actually refits the bank once for cold time; this prerequisite is amortized when the bank already exists. Fixed DCT constructs only the locked operator. RSS sampling includes bank construction, at 10 ms; total and baseline-relative values are both reported. Reference patch-retention implementations charge the complete shared patch-statistics routine when used alone, so timings are not claims of minimal possible implementations. See `docs/selection_revision_cost_protocol.md` for the clarified resource boundary.

`run_selection_revision.py` refuses to reuse a unit with a different package-source/config/protocol fingerprint or an incomplete status. Use a new output directory for a changed protocol. The partial first revision (`runs/selection_revision`) and overlapping exploratory cost warm-ups (`reports/selection_revision/cost_warmup_excluded`) are excluded from every final analysis and release evidence count. Main test predictions are produced only after pretest decisions and selected heads are serialized. Full test data use native class frequencies; there is no repeated test subsampling by seed.

The independent replay script reconstructs the exact content-clean official subsets, extracts patches by a NumPy reshape/transpose path independent of torch unfold, checks saved ridge coefficients with direct Cholesky solves, verifies validation-selected penalties, and requires exact equality of all full-test saved predictions. Analysis independently checks file hashes and stored test accuracies before generating manuscript numbers. Bootstrap resampling preserves shared test images across seeds and budgets, with different independent random streams for the three observed domains. The aggregate is equally weighted over domains; no bootstrap over three domains is offered as evidence for population-domain generalization.

The analysis additionally reads the historical `reports/selection/selection_by_unit.csv` and `deduplicated_test_selection.csv` to generate exact exclusion-only deltas. Historical summary CSVs are supplied in the source release; to recreate them from raw historical data and checkpoints use `docs/selection_reproducibility.md`.

## Manuscript

`paper/main.tex` includes `paper/revision_main.tex`; author: Ankit Guin. Revision tables and `revision_macros.tex` are generated by the analysis script, never hand-entered. The main paper preserves historical values in explicitly labeled appendices. With the workspace's cached Tectonic resources:

```powershell
$env:TECTONIC_CACHE_DIR=(Resolve-Path '.tools/tectonic-cache').Path
.tools/tectonic/tectonic.exe --only-cached --keep-logs paper/main.tex
```

Alternatively use a current LaTeX/BibTeX installation. Tectonic and its cached resources are local tools, not included in the code release. PDF rendering/visual inspection uses pypdfium2, with page images retained in the local validation directory.

## Release boundary

`scripts/package_selection_revision.py` creates a versioned clean source/reports/manuscript archive and a separate final-cohort checkpoint/prediction archive. Every archived file has its SHA256 and byte size in a manifest; each ZIP has an external SHA256 and is checked after writing. Historical source changes and the preliminary evidence bundle are preserved separately. No dataset images, full downloaded weights, virtual environment, Git history, credentials or partial experimental cohorts are included.

These are local, reviewable release artifacts. Upload to a public repository and immutable archive, obtain the actual persistent identifier, and replace the manuscript's availability statement before journal submission. No public upload or DOI is claimed. Affiliation, corresponding email and journal-specific declarations must also be supplied by the author; they are not invented from account metadata.
