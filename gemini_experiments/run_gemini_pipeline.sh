#!/usr/bin/env bash
# ============================================================================
# Stage 1: build the unified corpus and generate all perturbation material
# with the Gemini Batch API. Run this ONCE, from the project root, on a machine
# with internet access. No GPU required.
#
#   ./gemini_experiments/run_gemini_pipeline.sh                 # full pipeline
#   ./gemini_experiments/run_gemini_pipeline.sh --step prepare  # single step
#   MAX_PER_DATASET=2000 ./gemini_experiments/run_gemini_pipeline.sh
#
# Steps: prepare -> smoke -> build -> submit -> fetch -> merge -> validate
# The regenerate step reruns build/submit/fetch/merge/validate for rejected ids.
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-python3}"
STEP="all"
MAX_PER_DATASET="${MAX_PER_DATASET:-0}"     # 0 = full evaluation sets
SEED="${SEED:-20250101}"
ROUND="${ROUND:-1}"
DATASETS_ARG=()
SKIP_SMOKE="${SKIP_SMOKE:-0}"
GEN_DATE="$(date +%Y-%m-%d)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)       STEP="$2"; shift 2 ;;
    --round)      ROUND="$2"; shift 2 ;;
    --max-per-dataset) MAX_PER_DATASET="$2"; shift 2 ;;
    --datasets)   shift; while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS_ARG+=("$1"); shift; done ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

DS_FLAG=()
if [[ ${#DATASETS_ARG[@]} -gt 0 ]]; then DS_FLAG=(--datasets "${DATASETS_ARG[@]}"); fi

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

step_prepare() {
  banner "Phase 1-2: download + normalize the three benchmarks"
  local extra=()
  [[ "$MAX_PER_DATASET" != "0" ]] && extra=(--max-per-dataset "$MAX_PER_DATASET" --seed "$SEED")
  $PY -m gemini_experiments.prepare_datasets "${DS_FLAG[@]}" "${extra[@]}"
}

step_smoke() {
  [[ "$SKIP_SMOKE" == "1" ]] && { echo "smoke test skipped"; return 0; }
  banner "Pre-flight: live Gemini smoke test (a few requests)"
  $PY -m gemini_experiments.smoke_test "${DS_FLAG[@]}" --n 2
}

step_build()  { banner "Phase 3: build batch request files (round $ROUND)"
                $PY -m gemini_experiments.batch_build  "${DS_FLAG[@]}" --round "$ROUND"; }
step_submit() { banner "Phase 3: submit batch jobs (round $ROUND)"
                $PY -m gemini_experiments.batch_submit --round "$ROUND"; }
step_fetch()  { banner "Phase 3: wait for + fetch batch results (round $ROUND)"
                $PY -m gemini_experiments.batch_fetch  --round "$ROUND" --wait; }
step_merge()  { banner "Phase 7: merge into release JSONL files"
                $PY -m gemini_experiments.merge_release "${DS_FLAG[@]}" --generation-date "$GEN_DATE"; }
step_validate() { banner "Phase 8: validate generated perturbations"
                $PY -m gemini_experiments.validate "${DS_FLAG[@]}"; }
step_finalize() { banner "Phase 8: write the usable-only release"
                $PY -m gemini_experiments.finalize_release "${DS_FLAG[@]}"
                $PY -m gemini_experiments.paper_report; }

step_regenerate() {
  local inv="gemini_experiments/reports/invalid_ids.json"
  [[ -f "$inv" ]] || { echo "No $inv - run validate first."; return 1; }
  local n
  n=$($PY -c "import json,sys;d=json.load(open('$inv'));print(sum(len(v) for v in d.values()))")
  if [[ "$n" == "0" ]]; then echo "Nothing to regenerate."; return 0; fi
  banner "Regenerating $n rejected item(s) as round $ROUND"
  $PY -m gemini_experiments.batch_build  "${DS_FLAG[@]}" --round "$ROUND" --only-ids "$inv"
  $PY -m gemini_experiments.batch_submit --round "$ROUND"
  $PY -m gemini_experiments.batch_fetch  --round "$ROUND" --wait
  step_merge
  step_validate
  step_finalize
}

case "$STEP" in
  prepare)    step_prepare ;;
  smoke)      step_smoke ;;
  build)      step_build ;;
  submit)     step_submit ;;
  fetch)      step_fetch ;;
  merge)      step_merge ;;
  validate)   step_validate ;;
  finalize)   step_finalize ;;
  regenerate) step_regenerate ;;
  all)
    step_prepare
    step_smoke
    step_build
    step_submit
    step_fetch
    step_merge
    step_validate
    step_finalize
    banner "Done"
    echo "Release artifacts:  data/perturbations/*_perturbations.jsonl"
    echo "Reports:            gemini_experiments/reports/"
    echo
    echo "If validate rejected items, regenerate them with:"
    echo "  ROUND=2 ./gemini_experiments/run_gemini_pipeline.sh --step regenerate"
    ;;
  *) echo "Unknown step: $STEP" >&2; exit 1 ;;
esac
