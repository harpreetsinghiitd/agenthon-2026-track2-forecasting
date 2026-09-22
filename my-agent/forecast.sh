#!/usr/bin/env bash
set -euo pipefail

# Locate this script's directory so the .py is found regardless of cwd.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure the track repo is importable (for baselines/ and scoring/).
# Adjust REPO to wherever your clone lives inside the image.
REPO="${REPO:-/app}"

PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
  exec python3 "${HERE}/forecast.py" "$@"