#!/usr/bin/env bash
# ============================================================================
# 5% pilot: one served model, all three benchmarks, the eight main conditions.
#
#   CONFIG=configs/models/qwen3.5-9b-nothink.yaml \
#   MODEL_NAME=qwen3-8b PORT=8009 ./scripts/run_pilot.sh
#
# --limit 100 = the first 100 uids of EACH benchmark's sample index
#             = 300 items = 5.6% of the 5,324-item sample.
# The same 300 items are used for every condition and every re-invocation,
# so perturbed rows always pair against baseline rows.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:?set CONFIG to a file under configs/models/}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to the served-model-name on the server}"
PORT="${PORT:-8009}"
HOST="${HOST:-localhost}"
LIMIT="${LIMIT:-100}"
OUT="${OUT:-results/}"
CONCURRENCY="${CONCURRENCY:-32}"
DIAGNOSTICS="${DIAGNOSTICS:-0}"      # 1 = also run O1C / NQR / NQI

BASE_URL="http://${HOST}:${PORT}/v1"
COMMON=(--config "$CONFIG" --base-url "$BASE_URL" --model-name "$MODEL_NAME"
        --limit "$LIMIT" --max-concurrency "$CONCURRENCY" --out "$OUT")

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

banner "0/4  Health check — is the right model on $BASE_URL?"
python scripts/05_run.py "${COMMON[@]}" --health-check

banner "1/4  Dry run — three prompts per condition, nothing sent"
python scripts/05_run.py "${COMMON[@]}" \
  --conditions B0 N3 I1 O1 --datasets kazmmlu --dry-run --dry-run-n 1

banner "2/4  Main grid — B0 x3 + 7 perturbations on $((LIMIT * 3)) items"
python scripts/05_run.py "${COMMON[@]}" \
  --datasets kazmmlu kazculture kazbench_kk \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

if [[ "$DIAGNOSTICS" == "1" ]]; then
  banner "2b/4  Diagnostics — O1C / NQR / NQI (small at this --limit)"
  python scripts/05_run.py "${COMMON[@]}" --conditions O1C NQR NQI --force
fi

banner "3/4  Aggregate — long.parquet + paired.parquet"
python scripts/06_aggregate.py --results "$OUT"

banner "4/4  Metrics + figures"
python scripts/07_metrics.py --results "$OUT"
python scripts/08_figures.py --results "$OUT"

banner "Done"
echo "Headline numbers : ${OUT}metrics/paper_numbers.json"
echo "Self-flip floor  : ${OUT}metrics/self_flip_floor.csv"
echo "Flip vs floor    : ${OUT}metrics/flip_rates.csv"
echo "Token volume     : ${OUT}metrics/generation_health.csv"
echo "Figures          : ${OUT}figures/"
