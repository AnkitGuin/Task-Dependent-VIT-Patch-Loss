#!/usr/bin/env bash
set -euo pipefail
python -m patchloss.cli run-suite --suite full --device auto
