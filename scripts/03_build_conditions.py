#!/usr/bin/env python3
"""Materialize every condition once.  The grid must see identical perturbed
items across all 20 configurations -- that identity is what makes the
cross-configuration consensus check valid.

Also runs the build-time prompt-length guard: tokenize every rendered prompt
and report max / p99 per dataset x condition, so a `--max-model-len` problem
surfaces here rather than as scattered 400s during the grid.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from stability.conditions import ALL_CONDITIONS, MAIN_GRID, DIAGNOSTICS, build_many
from stability.config import load_defaults
from stability.io import read_json, write_json, write_jsonl
from stability.normalize import DATASETS, iter_normalized
from stability.prompts import load_prompts


def condition_path(work: Path, dataset: str, condition: str) -> Path:
    return work / "conditions" / f"{dataset}__{condition}.jsonl"


def _tokenizer(name: str | None):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(f"[warn] length_guard.tokenizer={name!r} but `transformers` is not "
              "installed; falling back to a character-based estimate.")
        return None
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=list(MAIN_GRID) + list(DIAGNOSTICS))
    ap.add_argument("--tokenizer", default=None, help="override length_guard.tokenizer")
    args = ap.parse_args()

    for c in args.conditions:
        if c not in ALL_CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; valid: {ALL_CONDITIONS}")

    d = load_defaults()
    work = Path(d["work_dir"])
    book = load_prompts()
    shuffle_seed = d["shuffle_seed"]

    items_by_ds = {ds: {i.uid: i for i in iter_normalized(work / "normalized" / f"{ds}.jsonl")}
                   for ds in d.get("datasets", DATASETS)}
    assert set(items_by_ds) == set(DATASETS)

    sample = {ds: read_json(work / "sample_index" / f"{ds}_sample.json")["uids"]
              for ds in items_by_ds}
    o1c_pool = set(read_json(work / "sample_index" / "o1c_pool.json")["uids"])
    nq_pool = set(read_json(work / "sample_index" / "nq_pool.json")["uids"])

    guard = dict(d.get("length_guard") or {})
    tok = _tokenizer(args.tokenizer or guard.get("tokenizer"))
    max_len = int(guard.get("max_model_len", 32768))
    warn_at = max_len * float(guard.get("warn_fraction", 0.8))

    report: dict = {"prompt_version": book.version, "prompt_sha256": book.sha256,
                    "shuffle_seed": shuffle_seed,
                    "tokenizer": args.tokenizer or guard.get("tokenizer"),
                    "max_model_len": max_len, "cells": {}}
    warnings: list[dict] = []
    skipped_report: dict = {}

    for ds in sorted(items_by_ds):
        pool_uids = sample[ds]
        for cond in args.conditions:
            uids = pool_uids
            if cond == "O1C":
                uids = [u for u in pool_uids if u in o1c_pool]
            elif cond in ("NQR", "NQI"):
                uids = [u for u in pool_uids if u in nq_pool]
            items = [items_by_ds[ds][u] for u in uids]

            cis, skipped = build_many(items, cond, shuffle_seed, skip_missing=True)
            if skipped:
                skipped_report[f"{ds}__{cond}"] = skipped

            lens = []
            for ci in cis:
                text = book.render(ci)
                n = len(tok(text)["input_ids"]) if tok else max(1, len(text) // 3)
                lens.append(n)
                if n >= warn_at:
                    warnings.append({"uid": ci.uid, "dataset": ds, "condition": cond,
                                     "tokens": n, "max_model_len": max_len,
                                     "estimated": tok is None})

            out = condition_path(work, ds, cond)
            write_jsonl(out, [ci.to_dict() for ci in cis])
            cell = {
                "path": str(out), "n_items": len(items), "n_rows": len(cis),
                "n_skipped_missing_field": len(skipped),
                "prompt_tokens_max": max(lens) if lens else 0,
                "prompt_tokens_p99": (int(statistics.quantiles(lens, n=100)[98])
                                      if len(lens) >= 100 else (max(lens) if lens else 0)),
                "prompt_tokens_median": int(statistics.median(lens)) if lens else 0,
                "token_counts_estimated": tok is None,
            }
            report["cells"][f"{ds}__{cond}"] = cell
            print(f"{ds:14s} {cond:4s} rows={len(cis):6d} skipped={len(skipped):3d} "
                  f"tok_max={cell['prompt_tokens_max']:6d} "
                  f"p99={cell['prompt_tokens_p99']:6d}"
                  f"{'  (estimated)' if tok is None else ''}")

    report["n_length_warnings"] = len(warnings)
    report["skipped_missing_field"] = {k: len(v) for k, v in skipped_report.items()}
    write_json(work / "conditions" / "length_warnings.json", warnings)
    write_json(work / "conditions" / "_build_report.json", report)
    write_json(work / "conditions" / "_skipped_missing_field.json", skipped_report)
    print(f"\nlength warnings: {len(warnings)} (>= {warn_at:.0f} tokens) -> "
          f"{work / 'conditions' / 'length_warnings.json'}")
    if tok is None:
        print("NOTE: token counts are a char/3 estimate. Set length_guard.tokenizer "
              "in configs/defaults.yaml (and pip install transformers) before the grid.")


if __name__ == "__main__":
    main()
