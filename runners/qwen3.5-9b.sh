#!/usr/bin/env bash
# ==========================================================================
#  qwen3.5-9b  --  2 configuration(s), FULL sample
#
#  configurations : qwen3.5-9b-nothink, qwen3.5-9b-think
#  sample         : 5324 items (kazmmlu 2000, kazculture 1324, kazbench_kk 2000)
#  requests       : 53,240 per configuration  ->  106,480 total
#
#  STEP 1 -- serve on its own GPU (separate terminal or batch job):
#
#    MODEL_DIR=/path/to/models/qwen3.5-9b \
#    SERVED_NAME=qwen3.5-9b PORT=<PORT> CUDA_VISIBLE_DEVICES=<GPU> \
#    REASONING_PARSER=qwen3 MAX_MODEL_LEN=34816 TP=1 \
#    ./serve_model.sh
#
#  STEP 2 -- run this file:   ./runners/qwen3.5-9b.sh
#
#  Runs evaluation ONLY. Aggregation is a separate step so that several
#  models running in parallel do not race on results/aggregate/:
#      ./runners/_analysis.sh          # once, after everything finishes
# ==========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- EDIT THESE TWO LINES ----------------------------------------------
PORT=8000
SERVED=qwen3.5-9b          # must equal the server's --served-model-name
# ------------------------------------------------------------------------

HOST="${HOST:-localhost}"
OUT="${OUT:-results/}"
BASE_URL="http://${HOST}:${PORT}/v1"
COMMON=(--base-url "$BASE_URL" --model-name "$SERVED" --out "$OUT"
        --datasets kazmmlu kazculture kazbench_kk)

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

banner "qwen3.5-9b  @  $BASE_URL  (served name: $SERVED)"

# --------------------------------------------------------------------------
banner "qwen3.5-9b-nothink  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/qwen3.5-9b-nothink.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

# Diagnostics: qwen3.5-9b-nothink is listed in defaults.o1c_configs and defaults.nq_probe_configs
banner "qwen3.5-9b-nothink  --  diagnostics O1C NQR NQI"
python scripts/05_run.py --config configs/models/qwen3.5-9b-nothink.yaml \
  "${COMMON[@]}" --conditions O1C NQR NQI

# --------------------------------------------------------------------------
banner "qwen3.5-9b-think  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/qwen3.5-9b-think.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

# P1 floor-transfer check: qwen3.5-9b-think is one of defaults.p1_floor_configs
banner "qwen3.5-9b-think  --  P1 x3 (floor transfer)"
python scripts/05_run.py --config configs/models/qwen3.5-9b-think.yaml \
  "${COMMON[@]}" --conditions P1 --repeat P1=3

# Diagnostics: qwen3.5-9b-think is listed in defaults.o1c_configs
banner "qwen3.5-9b-think  --  diagnostics O1C"
python scripts/05_run.py --config configs/models/qwen3.5-9b-think.yaml \
  "${COMMON[@]}" --conditions O1C

banner "done: qwen3.5-9b"
echo "  ${OUT}qwen3.5-9b-nothink/"
echo "  ${OUT}qwen3.5-9b-think/"
echo
echo "When every model has finished:  ./runners/_analysis.sh"
