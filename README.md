# Frozen Linear Patch Operator Selection

Research code for studying fixed-budget selection among frozen linear patch operators. The repository contains implementation code, configuration files, tests, and scripts. It does not include manuscript files, reviewer correspondence, experiment reports, datasets, checkpoints, or generated result tables.

## Setup

The recorded environment uses Python 3.13 on CPU. Install the CPU PyTorch wheel index and locked dependencies:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-revision-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Study scripts are under `scripts/`; configurations are under `configs/`. Dataset preparation downloads upstream public datasets at runtime. No dataset images or pretrained checkpoints are included. See `THIRD_PARTY_NOTICES.md` and `licenses/` for upstream terms.

## License

The software is provided under the MIT License. Third-party materials retain their original licenses.
