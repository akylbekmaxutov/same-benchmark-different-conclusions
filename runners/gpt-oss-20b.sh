#!/usr/bin/env bash
# ==========================================================================
#  gpt-oss-20b  --  2 configuration(s), FULL sample
#
#  configurations : gpt-oss-20b-low, gpt-oss-20b-high
#  sample         : 5324 items (kazmmlu 2000, kazculture 1324, kazbench_kk 2000)
#  requests       : 53,240 per configuration  ->  106,480 total
#
#  STEP 1 -- serve on its own GPU (separate terminal or batch job):
#
#    MODEL_DIR=/path/to/models/gpt-oss-20b \
#    SERVED_NAME=gpt-oss-20b PORT=<PORT> CUDA_VISIBLE_DEVICES=<GPU> \
#    REASONING_PARSER=GptOss MAX_MODEL_LEN=34816 TP=1 \
#    ./serve_model.sh
#
#  STEP 2 -- run this file:   ./runners/gpt-oss-20b.sh
#
#  Runs evaluation ONLY. Aggregation is a separate step so that several
#  models running in parallel do not race on results/aggregate/:
#      ./runners/_analysis.sh          # once, after everything finishes
# ==========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- EDIT THESE TWO LINES ----------------------------------------------
PORT=8006
SERVED=gpt-oss-20b          # must equal the server's --served-model-name
# ------------------------------------------------------------------------

HOST="${HOST:-localhost}"
OUT="${OUT:-results/}"
BASE_URL="http://${HOST}:${PORT}/v1"
COMMON=(--base-url "$BASE_URL" --model-name "$SERVED" --out "$OUT"
        --datasets kazmmlu kazculture kazbench_kk)

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

banner "gpt-oss-20b  @  $BASE_URL  (served name: $SERVED)"

# --------------------------------------------------------------------------
banner "gpt-oss-20b-low  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/gpt-oss-20b-low.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

# --------------------------------------------------------------------------
banner "gpt-oss-20b-high  --  main grid, B0 x3 + 7 perturbations"
python scripts/05_run.py --config configs/models/gpt-oss-20b-high.yaml \
  "${COMMON[@]}" \
  --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
  --baseline-runs 3

# Diagnostics: gpt-oss-20b-high is listed in defaults.o1c_configs and defaults.nq_probe_configs
banner "gpt-oss-20b-high  --  diagnostics O1C NQR NQI"
python scripts/05_run.py --config configs/models/gpt-oss-20b-high.yaml \
  "${COMMON[@]}" --conditions O1C NQR NQI

banner "done: gpt-oss-20b"
echo "  ${OUT}gpt-oss-20b-low/"
echo "  ${OUT}gpt-oss-20b-high/"
echo
echo "When every model has finished:  ./runners/_analysis.sh"
