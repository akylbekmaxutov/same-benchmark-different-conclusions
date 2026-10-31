#!/usr/bin/env python3
"""raw perturbation JSONL -> unified `Item` JSONL in work/normalized/."""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from stability.config import load_defaults
from stability.io import write_json, write_jsonl
from stability.normalize import DATASETS, load_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None, choices=["released", "full"])
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    d = load_defaults()
    variant = args.variant or d.get("perturbation_variant", "released")
    datasets = args.datasets or d.get("datasets", list(DATASETS))
    work = Path(d["work_dir"])

    items = load_all(d["raw_data_dir"], datasets, variant)
    summary = {"variant": variant, "raw_data_dir": d["raw_data_dir"], "datasets": {}}
    for ds, rows in items.items():
        out = work / "normalized" / f"{ds}.jsonl"
        n = write_jsonl(out, [i.to_dict() for i in rows])
        with_e = sum(1 for i in rows if i.has_option_e)
        missing = {f: sum(1 for i in rows if getattr(i, f) is None)
                   for f in ("relevant_noise", "irrelevant_noise", "question_paraphrase")}
        summary["datasets"][ds] = {
            "path": str(out), "n_items": n, "n_with_option_e": with_e,
            "n_four_option": n - with_e, "missing_generated_fields": missing,
            "n_categories": len({i.category for i in rows}),
            "n_subcategories": len({i.subcategory for i in rows if i.subcategory}),
        }
        print(f"{ds:14s} {n:6d} items -> {out}")
    write_json(work / "normalized" / "_summary.json", summary)
    # every aggregation must assert it saw all three
    assert set(summary["datasets"]) == set(DATASETS), \
        f"expected all of {DATASETS}, got {sorted(summary['datasets'])}"


if __name__ == "__main__":
    main()
