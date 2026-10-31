"""Final step: split the complete generation record into the usable release.

    python -m gemini_experiments.finalize_release

Reads   data/perturbations/{dataset}_perturbations_full.jsonl   (everything)
Writes  data/perturbations/{dataset}_perturbations_full.jsonl   (re-stamped)
        data/perturbations/{dataset}_perturbations.jsonl        (USABLE ONLY)

The verdict is recomputed here rather than read from invalid_ids.json, so the
files can never drift out of sync with the current validator. Every row in the
_full file carries:

    validation_passed : bool   - usable
    validation_errors : [str]  - why it was excluded (empty if passed)
    validation_warnings: [str] - kept, but flagged for manual inspection

The plain file is what `stability_eval` loads, so the evaluation set is exactly
the accepted items and all eight conditions cover an identical item set.
"""
from __future__ import annotations

import argparse
import collections
import json
from typing import Any, Dict

from .config import DATASETS, REPORTS_DIR, normalized_path, release_full_path, release_path
from .jsonl import read_jsonl, write_jsonl
from .validate import validate_item


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the usable-only release files.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--keep-warnings", action="store_true", default=True,
                    help="Keep items that only produced warnings (default).")
    ap.add_argument("--drop-warnings", dest="keep_warnings", action="store_false",
                    help="Also exclude items with any warning (much stricter).")
    args = ap.parse_args()

    report: Dict[str, Any] = {"kept_warned_items": args.keep_warnings, "datasets": {}}

    for dataset in args.datasets:
        full_path = release_full_path(dataset)
        if not full_path.exists():
            print(f"[{dataset}] SKIP - {full_path} missing (run merge_release first)")
            continue

        sources = {r["id"]: r for r in read_jsonl(normalized_path(dataset))}
        rows = read_jsonl(full_path)

        stamped, usable = [], []
        reasons: collections.Counter = collections.Counter()
        for row in rows:
            source = sources.get(row["id"])
            if source is None:
                errors, warnings = ["item id not present in the normalized corpus"], []
            else:
                errors, warnings = validate_item(row, source)

            row["validation_passed"] = not errors
            row["validation_errors"] = errors
            row["validation_warnings"] = warnings
            stamped.append(row)

            if errors:
                for e in errors:
                    reasons[e.split(":")[0]] += 1
                continue
            if warnings and not args.keep_warnings:
                continue
            usable.append({k: v for k, v in row.items()
                           if k not in ("validation_errors", "validation_warnings")})

        write_jsonl(full_path, stamped)
        write_jsonl(release_path(dataset), usable)

        n = len(stamped)
        report["datasets"][dataset] = {
            "n_generated": n,
            "n_usable": len(usable),
            "n_excluded": n - len(usable),
            "usable_rate": round(len(usable) / n, 4) if n else 0.0,
            "exclusion_reasons": dict(reasons.most_common()),
        }
        print(f"[{dataset}] usable {len(usable)}/{n} ({len(usable)/n:.1%})")
        print(f"           usable   -> {release_path(dataset)}")
        print(f"           complete -> {full_path}")

    totals = report["datasets"].values()
    if totals:
        n = sum(d["n_generated"] for d in totals)
        u = sum(d["n_usable"] for d in totals)
        report["overall"] = {"n_generated": n, "n_usable": u, "n_excluded": n - u,
                             "usable_rate": round(u / n, 4)}
        print(f"\nOVERALL usable {u}/{n} ({u/n:.1%}), excluded {n - u}")

    path = REPORTS_DIR / "release_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Release report -> {path}")


if __name__ == "__main__":
    main()
