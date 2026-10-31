#!/usr/bin/env python3
"""results/*/raw/*.jsonl -> results/aggregate/{long,paired}.parquet

Everything in §12 reads these two tables and groups.  Nothing recomputes a
pairing.  Run this before any metric.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from stability.aggregate import (AggregationRefused, build_paired, coverage_report,
                                 discover_configs, load_long, load_manifests)
from stability.config import load_defaults
from stability.io import write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--allow-shortfall", action="store_true",
                    help="report cell shortfalls instead of exiting non-zero")
    args = ap.parse_args()

    d = load_defaults()
    results = Path(args.results or d["results_dir"])
    outdir = results / "aggregate"

    try:
        long = load_long(results)
        manifests = load_manifests(discover_configs(results))
    except AggregationRefused as exc:
        print(f"[refused] {exc}", file=sys.stderr)
        return 2

    cov = coverage_report(long, manifests, d["work_dir"], tuple(d["datasets"]))
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "coverage.json", cov)

    long.to_parquet(outdir / "long.parquet", index=False)
    paired = build_paired(long)
    paired.to_parquet(outdir / "paired.parquet", index=False)

    print(f"long.parquet   {len(long):8d} rows, {cov['n_configs']} configurations")
    print(f"paired.parquet {len(paired):8d} rows")
    if cov["configs_missing_datasets"]:
        print("[warn] configurations missing a dataset:", cov["configs_missing_datasets"])
        print("       KazBench-KK is the one that falls out of loops; any aggregate "
              "must see all three.")
        if not args.allow_shortfall:
            return 4
    if cov["shortfalls"]:
        print(f"[warn] {len(cov['shortfalls'])} cells short of the expected row count; "
              f"see {outdir / 'coverage.json'}")
        for s in cov["shortfalls"][:10]:
            print("   ", s)
        if not args.allow_shortfall:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
