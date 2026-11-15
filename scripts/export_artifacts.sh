#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports
python -m patchloss.cli aggregate
python -m patchloss.cli figures
python -m patchloss.cli audit
