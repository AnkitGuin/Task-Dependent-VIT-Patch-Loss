#!/usr/bin/env bash
set -euo pipefail
python -m patchloss.cli doctor
python -m patchloss.cli test-theory
python -m patchloss.cli run-suite --suite smoke --device cpu
