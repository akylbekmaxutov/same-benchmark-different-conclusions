#!/usr/bin/env bash
# ==========================================================================
#  muse-glimmer-30b  --  2 configuration(s), FULL sample
#
#  configurations : muse-glimmer-30b-low, muse-glimmer-30b-high
#  sample         : 5324 items (kazmmlu 2000, kazculture 1324, kazbench_kk 2000)
#  requests       : 53,240 per configuration  ->  106,480 total
#
#  STEP 1 -- serve on its own GPU (separate terminal or batch job):
#
#    MODEL_DIR=/path/to/models/muse-glimmer-30b \
#    SERVED_NAME=muse-glimmer-30b PORT=<PORT> CUDA_VISIBLE_DEVICES=<GPU> \
#    REASONING_PARSER=muse-glimmer MAX_MODEL_LEN=34816 TP=2 \
#    ./serve_model.sh
#
#  STEP 2 -- run this file:   ./runners/muse-glimmer-30b.sh
#
#  Runs evaluation ONLY. Aggregation is a separate step so that several
#  models running in parallel do not race on results/aggregate/:
#      ./runners/_analysis.sh          # once, after everything finishes
# ==========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- EDIT THESE TWO LINES ----------------------------------------------
PORT=8007
SERVED=muse-glimmer          # must equal the server's --served-model-name
# ------------------------------------------------------------------------

HOST="${HOST:-localhost}"
OUT="${OUT:-results/}"
BASE_URL="http://${HOST}:${PORT}/v1"
COMMON=(--base-url "$BASE_URL" --model-name "$SERVED" --out "$OUT"
        --datasets kazmmlu kazculture kazbench_kk)

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

banner "muse-glimmer-30b  @  $BASE_URL  (served name: $SERVED)"

# --------------------------------------------------------------------------
banner "muse-glimmer-30b-low  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/muse-glimmer-30b-low.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

# --------------------------------------------------------------------------
banner "muse-glimmer-30b-high  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/muse-glimmer-30b-high.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

# Diagnostics: muse-glimmer-30b-high is listed in defaults.o1c_configs
banner "muse-glimmer-30b-high  --  diagnostics O1C"
python scripts/05_run.py --config configs/models/muse-glimmer-30b-high.yaml \
  "${COMMON[@]}" --conditions O1C

banner "done: muse-glimmer-30b"
echo "  ${OUT}muse-glimmer-30b-low/"
echo "  ${OUT}muse-glimmer-30b-high/"
echo
echo "When every model has finished:  ./runners/_analysis.sh"
