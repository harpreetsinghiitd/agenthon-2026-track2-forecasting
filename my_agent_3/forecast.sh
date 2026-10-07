#!/usr/bin/env bash
# Executive summary: launch the my_agent_3 Python entrypoint for the forecast verb.
# Accepts either `forecast.sh forecast --...` or `forecast.sh --...`.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${HERE}/.." && pwd)}"

if [[ "${1:-}" == "forecast" ]]; then
  shift
fi

PYTHONPATH="${HERE}:${REPO}:${PYTHONPATH:-}" \
  exec python3 "${HERE}/main.py" "$@"
