#!/usr/bin/env bash
set -euo pipefail
python -m patchloss.cli test-theory
pytest -q
python -m patchloss.cli audit --input runs --output docs/results_audit.md
