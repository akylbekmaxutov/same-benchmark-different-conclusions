#!/usr/bin/env bash
# ==========================================================================
#  qolda-avl-34b  --  1 configuration(s), FULL sample
#
#  configurations : qolda-avl-34b
#  sample         : 5324 items (kazmmlu 2000, kazculture 1324, kazbench_kk 2000)
#  requests       : 53,240 per configuration  ->  53,240 total
#
#  STEP 1 -- serve on its own GPU (separate terminal or batch job):
#
#    MODEL_DIR=/path/to/models/qolda-avl-34b \
#    SERVED_NAME=qolda-avl-34b PORT=<PORT> CUDA_VISIBLE_DEVICES=<GPU> \
#    REASONING_PARSER=qwen3 MAX_MODEL_LEN=34816 TP=2 \
#    VLLM_EXTRA_ARGS='--trust-remote-code' \
#    ./serve_model.sh
#
#  STEP 2 -- run this file:   ./runners/qolda-avl-34b.sh
#
#  Runs evaluation ONLY. Aggregation is a separate step so that several
#  models running in parallel do not race on results/aggregate/:
#      ./runners/_analysis.sh          # once, after everything finishes
# ==========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- EDIT THESE TWO LINES ----------------------------------------------
PORT=8009
SERVED=qolda-avl-34b          # must equal the server's --served-model-name
# ------------------------------------------------------------------------

HOST="${HOST:-localhost}"
OUT="${OUT:-results/}"
BASE_URL="http://${HOST}:${PORT}/v1"
COMMON=(--base-url "$BASE_URL" --model-name "$SERVED" --out "$OUT"
        --datasets kazmmlu kazculture kazbench_kk)

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

banner "qolda-avl-34b  @  $BASE_URL  (served name: $SERVED)"

# --------------------------------------------------------------------------
banner "qolda-avl-34b  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/qolda-avl-34b.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

banner "done: qolda-avl-34b"
echo "  ${OUT}qolda-avl-34b/"
echo
echo "When every model has finished:  ./runners/_analysis.sh"
