#!/usr/bin/env bash
set -euo pipefail
python -m patchloss.cli aggregate --input runs --output reports
python -m patchloss.cli figures --input reports --output reports/figures
