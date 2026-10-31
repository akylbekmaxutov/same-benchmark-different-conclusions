#!/usr/bin/env bash
# ============================================================================
# Run EVERY mode of one served model against one live endpoint.
#
#   ./scripts/run_model.sh qwen3.5-9b              # nothink, then think
#   ./scripts/run_model.sh gpt-oss-20b             # low, then high
#   LIMIT=100 ./scripts/run_model.sh qwen3.5-9b    # 5% pilot
#   ./scripts/run_model.sh --list
#
# The modes of a model differ only in chat_template_kwargs or the system
# prompt, so a single deployment serves all of them -- no restart between them.
#
# Env: PORT (8000) HOST (localhost) LIMIT (0 = full sample) OUT (results/)
#      CONDITIONS ("B0 N1 N2 N3 N4 I1 O1 P1")  BASELINE_RUNS (3)
#      DRY_RUN=1  SKIP_ANALYSIS=1
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GROUPS_FILE=configs/model_groups.yaml

PORT="${PORT:-8000}"
HOST="${HOST:-localhost}"
LIMIT="${LIMIT:-0}"
OUT="${OUT:-results/}"
CONDITIONS="${CONDITIONS:-B0 N1 N2 N3 N4 I1 O1 P1}"
BASELINE_RUNS="${BASELINE_RUNS:-3}"

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
die()    { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == "--list" || $# -eq 0 ]]; then
  python - "$GROUPS_FILE" <<'PY'
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1]))
print(f"models_root: {spec['models_root']}\n")
print(f"{'group':22s} {'served name':22s} {'parser':16s} configurations")
for name, g in spec["groups"].items():
    print(f"{name:22s} {g['served_name']:22s} {str(g.get('reasoning_parser')):16s} "
          + ", ".join(g["configs"]))
PY
  exit 0
fi

GROUP="$1"; shift
eval "$(python - "$GROUPS_FILE" "$GROUP" <<'PY'
import sys, yaml, shlex
spec = yaml.safe_load(open(sys.argv[1]))
g = spec["groups"].get(sys.argv[2])
if g is None:
    print(f'die "unknown model group {shlex.quote(sys.argv[2])}; try --list"'); sys.exit(0)
print(f'SERVED={shlex.quote(g["served_name"])}')
print(f'MIN_MML={spec.get("min_max_model_len", 32768)}')
print(f'EXTRA_SERVE={shlex.quote(g.get("extra_serve_args") or "")}')
print(f'PARSER={shlex.quote(str(g.get("reasoning_parser") or ""))}')
print(f'MODEL_DIR={shlex.quote(spec["models_root"] + "/" + sys.argv[2])}')
print("CONFIGS=(" + " ".join(shlex.quote(c) for c in g["configs"]) + ")")
PY
)"

BASE_URL="http://${HOST}:${PORT}/v1"
LIMIT_ARG=(); [[ "$LIMIT" != "0" ]] && LIMIT_ARG=(--limit "$LIMIT")

banner "$GROUP  ->  ${#CONFIGS[@]} configuration(s): ${CONFIGS[*]}"
echo "  endpoint    $BASE_URL   (served name must be '$SERVED')"
echo "  model dir   $MODEL_DIR"
echo
echo "  Serve it with:"
echo "    MODEL_DIR=$MODEL_DIR \\"
echo "    SERVED_NAME=$SERVED PORT=$PORT \\"
echo "    REASONING_PARSER=$PARSER MAX_MODEL_LEN=$MIN_MML \\"
[[ -n "$EXTRA_SERVE" ]] && echo "    VLLM_EXTRA_ARGS='$EXTRA_SERVE' \\"
echo "    ./serve_model.sh"
echo
[[ "$LIMIT" != "0" ]] && echo "  pilot       first $LIMIT items per benchmark"

for CFG in "${CONFIGS[@]}"; do
  banner "$CFG"
  ARGS=(--config "configs/models/${CFG}.yaml" --base-url "$BASE_URL"
        --model-name "$SERVED" --out "$OUT"
        --datasets kazmmlu kazculture kazbench_kk
        --conditions $CONDITIONS --baseline-runs "$BASELINE_RUNS"
        "${LIMIT_ARG[@]}" "$@")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    python scripts/05_run.py "${ARGS[@]}" --dry-run --dry-run-n 1
  else
    python scripts/05_run.py "${ARGS[@]}"
  fi
done

if [[ "${SKIP_ANALYSIS:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  banner "Aggregate + metrics + figures"
  python scripts/06_aggregate.py --results "$OUT" || \
    echo "(aggregation reported gaps -- see ${OUT}aggregate/coverage.json)"
  python scripts/07_metrics.py --results "$OUT"
  python scripts/08_figures.py --results "$OUT"
fi

banner "Done: $GROUP"
for CFG in "${CONFIGS[@]}"; do echo "  ${OUT}${CFG}/"; done
