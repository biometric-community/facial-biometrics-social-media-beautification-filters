#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MONO="$(cd "$ROOT/../../.." && pwd)"
cd "$ROOT"
PYTHON="${MONO}/.venv/bin/python"
"$PYTHON" -m pip install -r requirements.txt
echo "OK"
