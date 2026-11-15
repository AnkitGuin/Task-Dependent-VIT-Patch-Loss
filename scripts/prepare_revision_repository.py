"""Prepare the authorized clean GitHub checkout without raw assets or checkpoints."""

from pathlib import Path
import shutil


def main():
    source = Path.cwd().resolve()
    target = (source.parent/"github-release").resolve()
    workspace = source.parent.resolve()
    if not target.is_relative_to(workspace) or target == source or not (target/".git").exists():
        raise ValueError("Expected the existing sibling release checkout inside this workspace")
    roots = ["src","tests","scripts","configs","docs","paper","reports/selection","reports/selection_revision",".github","licenses"]
    blocked_parts = {"__pycache__",".pytest_cache","cost","cost_warmup_excluded","pdf_validation",
                     "archive_preselection_20261114","archive_prerevision_20261115"}
    blocked_suffixes = {".pyc",".aux",".log",".bbl",".blg",".out",".toc",".zip",".npy",".npz"}
    copied = 0
    for root in roots:
        for f in (source/root).rglob("*"):
            relative = f.relative_to(source)
            if not f.is_file() or set(relative.parts)&blocked_parts or f.suffix in blocked_suffixes:
                continue
            destination = target/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,destination)
            copied += 1
    for name in ["pyproject.toml","requirements-lock.txt","requirements-revision-lock.txt","LICENSE","THIRD_PARTY_NOTICES.md","Makefile","CITATION.cff",".gitattributes"]:
        shutil.copy2(source/name,target/name)
    # The original release ignore rules remain, extended only for local development artifacts.
    ignore = target/".gitignore"
    legacy_private_rules = {"/paper/","/reports/","/docs/results_audit.md","*.pdf"}
    text = "\n".join(rule for rule in ignore.read_text().splitlines() if rule not in legacy_private_rules)+"\n"
    for rule in [".venv/",".tools/","artifacts/","data/","runs/","__pycache__/",".pytest_cache/",".ruff_cache/","*.egg-info/","paper/*.aux","paper/*.log","paper/*.bbl","paper/*.blg","paper/*.out"]:
        if rule not in text.splitlines():
            text += "\n"+rule
    ignore.write_text(text.rstrip()+"\n")
    (target/"README.md").write_text(README,encoding="utf-8")
    print(f"Prepared {copied} scientific source/report/manuscript files in {target}")


README = """# Selecting Frozen Linear Patch Operators

**An Empirical Audit under Matched Feature Budgets**

Author: **Ankit Guin**. Research manuscript and reproducible code; journal acceptance is not claimed.

This revision evaluates frozen shared linear patch maps, not complete pretrained Transformers. It addresses competitive baselines, exact-content separation, full-test uncertainty, fixed-policy controls and practical selection costs.

## Versioned code and evidence

Use [release v2026.11.15-v2](https://github.com/AnkitGuin/Task-Dependent-VIT-Patch-Loss/releases/tag/v2026.11.15-v2) for the matching source/manuscript and checkpoint/prediction evidence archives. Both have per-file SHA256 manifests and external archive checksums. Raw dataset images and full downloaded pretrained weights are excluded; preparation verifies their upstream checksums.

The [paper PDF](paper/main.pdf), [LaTeX source](paper/revision_main.tex), [response to review](docs/reviewer_response_20261115.md), [final protocol](docs/selection_revision_protocol.md), [reproduction guide](docs/selection_revision_reproducibility.md) and [claim register](docs/revision_claim_register.md) document the evidence and its limits. See [machine-readable results](reports/selection_revision/).

## Final revision evidence

- CIFAR-10, CIFAR-100 and independent cropped SVHN; five fresh training-subset seeds, two scalar-feature budgets, fourteen candidates: **30 units, 420 heads**.
- All 10,000 official test images per CIFAR dataset and all 26,032 SVHN test images; exact-content filtering of training before operator/head fitting.
- Eleven main selectors, including profile-evidence LogME, adaptive H-score, diagnostic probes and a fixed DCT rule developed on historical CIFAR results and locked before SVHN evaluation.
- One validation-only ridge grid from 10^-2 through 10^10. Every choice is saved before test inference.
- Paired seed/image uncertainty, matched patch-size comparisons, complete fractional selection frequencies, predetermined shrinkage sensitivity and isolated end-to-end selection time/peak-memory measurements.
- Independent NumPy patch extraction and direct-solve verification replay **all 420 heads**, with exact full-test prediction equality. Local test suite: **72 passed**.

| Main selector | Mean regret (percentage points) |
|---|---:|
| Covariance retention | 0.221 |
| Fixed DCT, p=4 | 0.294 |
| Diagnostic probe | 0.336 |
| Tuned diagnostic probe | 0.502 |
| Adaptive H-score | 0.932 |
| Profile LogME | 0.983 |

These are conditional observations, not a universal ranking. Covariance's pooled advantage over fixed DCT has an interval spanning zero. On SVHN it selects exactly the same PCA candidate as unlabeled reconstruction, so its positive gain over fixed DCT does not establish a need for label-dependent selection. Absolute SVHN accuracy is weak, even below the empirical test-majority reference. Negative findings and the original preliminary/exploratory cohorts remain explicit.

## Reproduce

Use Python 3.11+; the recorded final environment is CPython 3.13 on Windows, CPU. Install `requirements-revision-lock.txt` with the CPU PyTorch index, then the local package. Exact commands, asset preparation, environment limitations, replay and PDF generation are in the reproduction guide.

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-revision-lock.txt
python -m pip install -e . --no-deps
python scripts/download_selection_assets.py --config configs/selection_study.yaml
python scripts/download_revision_svhn.py
python -m pytest -q
python scripts/run_selection_revision.py
python scripts/verify_selection_revision.py
python scripts/benchmark_selection_revision.py
python scripts/analyze_selection_revision.py
```

Extract the evidence archive into this checkout to inspect saved heads and regenerate results without training. Run resource measurements with no other project experiments executing. Historical results require their distinct protocols; do not mix cohorts or silently resume with a different source fingerprint. Use `CITATION.cff` when citing the research artifact. Software is MIT-licensed; upstream dataset/weight terms remain separate.
"""


if __name__ == "__main__":
    main()
