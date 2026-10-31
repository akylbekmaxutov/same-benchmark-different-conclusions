"""Phase 1-2 CLI: download the three benchmarks and write the unified corpus to
`data/normalized/{dataset}.jsonl`.

    python -m gemini_experiments.prepare_datasets
    python -m gemini_experiments.prepare_datasets --datasets kazculture
    python -m gemini_experiments.prepare_datasets --max-per-dataset 2000

Sub-sampling is category-stratified and seeded, so the released corpus is
reproducible. Default is the full evaluation set (no sub-sampling).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from typing import Any, Dict, List

from .config import DATASETS, HF_SOURCES, HF_SPLITS, REPORTS_DIR, normalized_path
from .jsonl import write_jsonl
from .normalize import LOADERS, present_letters


def stratified_sample(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    """Take ~k rows keeping the category distribution, preserving input order."""
    if k <= 0 or k >= len(rows):
        return rows

    by_cat: Dict[str, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_cat[r["category"]].append(i)

    rng = random.Random(seed)
    keep: set[int] = set()
    # Proportional allocation with at least one item per category.
    total = len(rows)
    for cat, idxs in sorted(by_cat.items()):
        quota = max(1, round(k * len(idxs) / total))
        quota = min(quota, len(idxs))
        keep.update(rng.sample(idxs, quota))

    # Trim / top up to land on exactly k.
    remaining = [i for i in range(total) if i not in keep]
    rng.shuffle(remaining)
    while len(keep) < k and remaining:
        keep.add(remaining.pop())
    if len(keep) > k:
        droppable = sorted(keep)
        rng.shuffle(droppable)
        while len(keep) > k:
            keep.discard(droppable.pop())

    return [rows[i] for i in sorted(keep)]


def summarize(dataset: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cats = collections.Counter(r["category"] for r in rows)
    golds = collections.Counter(r["answer"] for r in rows)
    n_opts = collections.Counter(len(present_letters(r)) for r in rows)
    return {
        "dataset": dataset,
        "source_repo": HF_SOURCES[dataset],
        "source_split": HF_SPLITS[dataset],
        "n_items": len(rows),
        "n_categories": len(cats),
        "gold_label_distribution": dict(sorted(golds.items())),
        "option_count_distribution": dict(sorted(n_opts.items())),
        "categories": dict(cats.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize the three Kazakh MCQ benchmarks.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--kazmmlu-configs", nargs="+", default=None,
                    help="Override the auto-detected Kazakh KazMMLU subsets.")
    ap.add_argument("--max-per-dataset", type=int, default=0,
                    help="Category-stratified sub-sample size (0 = keep everything).")
    ap.add_argument("--seed", type=int, default=20250101)
    args = ap.parse_args()

    report: Dict[str, Any] = {"sampling": {"max_per_dataset": args.max_per_dataset,
                                           "seed": args.seed}, "datasets": {}}

    for dataset in args.datasets:
        print(f"\n=== {dataset} ({HF_SOURCES[dataset]}, split={HF_SPLITS[dataset]}) ===")
        loader = LOADERS[dataset]
        rows = list(loader(args.kazmmlu_configs) if dataset == "kazmmlu" else loader())
        print(f"[{dataset}] {len(rows)} valid normalized items")

        if args.max_per_dataset:
            rows = stratified_sample(rows, args.max_per_dataset, args.seed)
            print(f"[{dataset}] sub-sampled to {len(rows)}")
            # Re-key ids so they stay dense and stable for this release.
            for i, r in enumerate(rows):
                r["meta"]["full_corpus_id"] = r["id"]
                r["id"] = f"{dataset}-{i:06d}"

        out = normalized_path(dataset)
        write_jsonl(out, rows)
        print(f"[{dataset}] wrote {out}")

        info = summarize(dataset, rows)
        report["datasets"][dataset] = info
        print(f"[{dataset}] categories={info['n_categories']} "
              f"gold_dist={info['gold_label_distribution']}")

    path = REPORTS_DIR / "normalized_corpus_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCorpus report -> {path}")


if __name__ == "__main__":
    main()
