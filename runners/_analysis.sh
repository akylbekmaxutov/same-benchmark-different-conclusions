#!/usr/bin/env bash
# ==========================================================================
#  Aggregation + metrics + figures.  Run ONCE, after every model has
#  finished. Running it while a model is still writing would aggregate a
#  partial grid, and two copies at once would race on results/aggregate/.
# ==========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${OUT:-results/}"

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

banner "Aggregate"
python scripts/06_aggregate.py --results "$OUT" || {
  echo
  echo "Aggregation refused or found gaps. Details:"
  echo "  ${OUT}aggregate/coverage.json"
  echo "Re-run with --allow-shortfall once you have accounted for them:"
  echo "  python scripts/06_aggregate.py --results $OUT --allow-shortfall"
  exit 1
}

banner "Metrics"
python scripts/07_metrics.py --results "$OUT"

banner "Figures"
python scripts/08_figures.py --results "$OUT"

banner "Done"
echo "  ${OUT}metrics/paper_numbers.json   <- every number quoted in the paper"
echo "  ${OUT}metrics/                     <- tidy CSVs"
echo "  ${OUT}figures/                     <- PDF/PNG + appendix_tables.tex"
