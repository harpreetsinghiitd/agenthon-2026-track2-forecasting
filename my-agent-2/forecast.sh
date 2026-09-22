#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/app}"

PYTHONPATH="${HERE}:${REPO}:${PYTHONPATH:-}" \
  exec python3 "${HERE}/forecast.py" "$@"