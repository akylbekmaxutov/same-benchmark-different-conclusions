#!/usr/bin/env bash
# ==========================================================================
#  gemma-4-26b-a4b-it  --  2 configuration(s), FULL sample
#
#  configurations : gemma-4-26b-a4b-it-nothink, gemma-4-26b-a4b-it-think
#  sample         : 5324 items (kazmmlu 2000, kazculture 1324, kazbench_kk 2000)
#  requests       : 53,240 per configuration  ->  106,480 total
#
#  STEP 1 -- serve on its own GPU (separate terminal or batch job):
#
#    MODEL_DIR=/path/to/models/gemma-4-26b-a4b-it \
#    SERVED_NAME=gemma-4-26b-a4b-it PORT=<PORT> CUDA_VISIBLE_DEVICES=<GPU> \
#    REASONING_PARSER=gemma4 MAX_MODEL_LEN=34816 TP=2 \
#    ./serve_model.sh
#
#  STEP 2 -- run this file:   ./runners/gemma-4-26b-a4b-it.sh
#
#  Runs evaluation ONLY. Aggregation is a separate step so that several
#  models running in parallel do not race on results/aggregate/:
#      ./runners/_analysis.sh          # once, after everything finishes
# ==========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- EDIT THESE TWO LINES ----------------------------------------------
PORT=8005
SERVED=gemma-4-26b-a4b-it          # must equal the server's --served-model-name
# ------------------------------------------------------------------------

HOST="${HOST:-localhost}"
OUT="${OUT:-results/}"
BASE_URL="http://${HOST}:${PORT}/v1"
COMMON=(--base-url "$BASE_URL" --model-name "$SERVED" --out "$OUT"
        --datasets kazmmlu kazculture kazbench_kk)

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

banner "gemma-4-26b-a4b-it  @  $BASE_URL  (served name: $SERVED)"

# --------------------------------------------------------------------------
banner "gemma-4-26b-a4b-it-nothink  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/gemma-4-26b-a4b-it-nothink.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

# --------------------------------------------------------------------------
banner "gemma-4-26b-a4b-it-think  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/gemma-4-26b-a4b-it-think.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

banner "done: gemma-4-26b-a4b-it"
echo "  ${OUT}gemma-4-26b-a4b-it-nothink/"
echo "  ${OUT}gemma-4-26b-a4b-it-think/"
echo
echo "When every model has finished:  ./runners/_analysis.sh"
