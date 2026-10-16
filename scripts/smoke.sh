#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Prefer monorepo venv
if [[ -x "$(cd "$ROOT/../../.." && pwd)/.venv/bin/python" ]]; then
  export PATH="$(cd "$ROOT/../../.." && pwd)/.venv/bin:$PATH"
fi
python -c "import beautyfr; print('import ok', beautyfr.__version__)"
python -m beautyfr.train --help >/dev/null
python -m beautyfr.report --help >/dev/null
bash scripts/train.sh configs/smoke.yaml
bash scripts/eval.sh configs/smoke.yaml
bash scripts/report.sh configs/smoke.yaml
echo "SMOKE OK"
