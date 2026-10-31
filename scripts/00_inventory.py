#!/usr/bin/env python3
"""Inventory the read-only Gemini corpus.  Writes reports/inventory.{json,md}.

Discovery, not assumption: every count below is measured from the files.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import _bootstrap  # noqa: F401
from stability.config import load_defaults
from stability.io import read_jsonl, write_json, write_text
from stability.normalize import DATASETS, source_path

FIELDS = ("relevant_noise", "irrelevant_noise", "question_paraphrase")


def inspect(path: Path) -> dict:
    rows = list(read_jsonl(path))
    keys = collections.Counter()
    for r in rows:
        keys.update(r.keys())
    ids = [r.get("id") for r in rows]
    cats = collections.Counter(r.get("category") for r in rows)
    subcats = collections.Counter((r.get("meta") or {}).get("subcategory") for r in rows)
    cellsizes = collections.Counter(
        (r.get("category"), (r.get("meta") or {}).get("subcategory")) for r in rows)
    meta_keys = collections.Counter()
    for r in rows:
        meta_keys.update((r.get("meta") or {}).keys())

    out = {
        "path": str(path),
        "n_rows": len(rows),
        "row_keys": dict(keys),
        "meta_keys": dict(meta_keys),
        "unique_ids_within_file": len(set(ids)) == len(ids),
        "n_unique_ids": len(set(ids)),
        "n_with_option_e": sum(1 for r in rows if r.get("E") not in (None, "")),
        "n_generation_ok": sum(1 for r in rows if r.get("generation_ok")),
        "n_validation_passed": sum(1 for r in rows if r.get("validation_passed")),
        "empty_generated_field": {
            f: sum(1 for r in rows if not str(r.get(f) or "").strip()) for f in FIELDS},
        "answer_letter_distribution": dict(collections.Counter(r.get("answer") for r in rows)),
        "gold_not_among_options": [r["id"] for r in rows
                                   if not str(r.get(r.get("answer")) or "").strip()],
        "empty_option_text": [r["id"] for r in rows
                              if any(not str(r.get(L) or "").strip() for L in "ABCD")],
        "n_categories": len(cats),
        "category_counts": dict(cats.most_common()),
        "subcategory_populated": sum(1 for r in rows
                                     if (r.get("meta") or {}).get("subcategory")),
        "n_subcategories": len([k for k in subcats if k]),
        "n_category_x_subcategory_cells": len([k for k in cellsizes if k[1]]),
        "min_category_x_subcategory_cell": (min(v for k, v in cellsizes.items() if k[1])
                                            if any(k[1] for k in cellsizes) else None),
        "o1c_eligible_four_option": sum(1 for r in rows if r.get("E") in (None, "")),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to defaults.yaml")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    defaults = load_defaults(args.config) if args.config else load_defaults()
    raw = Path(defaults["raw_data_dir"])

    report: dict = {"raw_data_dir": str(raw), "files": {}}
    all_ids: collections.Counter = collections.Counter()
    for ds in DATASETS:
        for variant in ("released", "full"):
            p = source_path(raw, ds, variant)
            if not p.exists():
                report["files"][f"{ds}:{variant}"] = {"missing": str(p)}
                continue
            info = inspect(p)
            info["dataset"] = ds
            info["variant"] = variant
            report["files"][f"{ds}:{variant}"] = info
            if variant == "released":
                all_ids.update(r["id"] for r in read_jsonl(p))
    report["cross_file_id_collisions"] = [k for k, v in all_ids.items() if v > 1]
    report["n_ids_total_released"] = len(all_ids)

    outdir = Path(args.out)
    write_json(outdir / "inventory.json", report)

    lines = ["# Corpus inventory", "", f"`raw_data_dir` = `{raw}` (read-only)", ""]
    lines += ["| file | rows | uniq ids | with E | gen_ok | validated | "
              "empty rel/irr/para | cats | subcats | 4-option |",
              "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for key, info in report["files"].items():
        if "missing" in info:
            lines.append(f"| {key} | MISSING `{info['missing']}` | | | | | | | | |")
            continue
        e = info["empty_generated_field"]
        lines.append(
            f"| `{Path(info['path']).name}` | {info['n_rows']} | {info['n_unique_ids']} | "
            f"{info['n_with_option_e']} | {info['n_generation_ok']} | "
            f"{info['n_validation_passed']} | "
            f"{e['relevant_noise']}/{e['irrelevant_noise']}/{e['question_paraphrase']} | "
            f"{info['n_categories']} | {info['n_subcategories']} | "
            f"{info['o1c_eligible_four_option']} |")
    lines += ["", f"Cross-file id collisions: "
                  f"{report['cross_file_id_collisions'] or 'none'}",
              f"Total released ids: {report['n_ids_total_released']}", ""]
    write_text(outdir / "inventory.md", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
