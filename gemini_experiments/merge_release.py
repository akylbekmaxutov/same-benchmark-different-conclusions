"""Phase 7: merge the normalized corpus with the Gemini generations into the
release-ready `data/perturbations/{dataset}_perturbations.jsonl` files.

    python -m gemini_experiments.merge_release

Generations from later rounds override earlier ones, so re-running after a
regeneration round simply refreshes the release.
"""
from __future__ import annotations

import argparse
import collections
import json
from typing import Any, Dict

from .config import (BATCH_INPUT_DIR, DATASETS, GEMINI_MODEL, GENERATIONS_DIR,
                     GEN_PARAMS, OPTION_LETTERS, PROMPT_VERSION, REPORTS_DIR,
                     normalized_path, release_full_path)
from .jsonl import read_jsonl, write_jsonl


def collect_generations(dataset: str) -> Dict[str, Dict[str, Any]]:
    """id -> best generation, preferring the highest round that succeeded."""
    best: Dict[str, Dict[str, Any]] = {}
    round_dirs = sorted(
        (p for p in GENERATIONS_DIR.glob("r*") if p.is_dir()),
        key=lambda p: int(p.name[1:]),
    )
    for rd in round_dirs:
        path = rd / f"{dataset}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            prev = best.get(row["id"])
            # Later round wins, unless it failed and an earlier one succeeded.
            if prev is None or (row.get("error") is None or prev.get("error") is not None):
                best[row["id"]] = row
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge generations into release JSONL files.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--require-complete", action="store_true",
                    help="Drop items whose three perturbation fields are not all present.")
    ap.add_argument("--generation-date", default=None,
                    help="Recorded in the release provenance block (YYYY-MM-DD).")
    args = ap.parse_args()

    # The release is generated over several rounds and the prompt may be revised
    # between them, so record what each round actually used rather than only the
    # current value of the constants.
    per_round = []
    for jobs_file in sorted(BATCH_INPUT_DIR.glob("r*/jobs.json"),
                            key=lambda p: int(p.parent.name[1:])):
        data = json.loads(jobs_file.read_text(encoding="utf-8"))
        per_round.append({
            "round": data.get("round"),
            "gemini_model": data.get("model"),
            "prompt_version": data.get("prompt_version"),
            "n_requests": sum(j["n_requests"] for j in data.get("jobs", {}).values()),
        })

    provenance = {
        "gemini_model": GEMINI_MODEL,
        "current_prompt_version": PROMPT_VERSION,
        "current_generation_params": GEN_PARAMS,
        "generation_date": args.generation_date,
        "rounds": per_round,
        "note": "Per-item provenance is in the `generation_round` field of every "
                "released record; see reports/generation_provenance.json.",
    }

    report: Dict[str, Any] = {"provenance": provenance, "datasets": {}}

    for dataset in args.datasets:
        src = normalized_path(dataset)
        if not src.exists():
            print(f"[{dataset}] SKIP - {src} missing")
            continue

        items = read_jsonl(src)
        gens = collect_generations(dataset)
        stats = collections.Counter()
        out_rows = []

        for item in items:
            gen = gens.get(item["id"])
            fields = {
                "relevant_noise": gen.get("relevant_noise") if gen else None,
                "irrelevant_noise": gen.get("irrelevant_noise") if gen else None,
                "question_paraphrase": gen.get("question_paraphrase") if gen else None,
            }
            complete = all(fields.values())
            stats["total"] += 1
            if gen is None:
                stats["missing_generation"] += 1
            elif not complete:
                stats["incomplete"] += 1
            else:
                stats["complete"] += 1

            if args.require_complete and not complete:
                continue

            row = {
                "id": item["id"],
                "dataset": item["dataset"],
                "category": item["category"],
                "question": item["question"],
                **{L: item.get(L) for L in OPTION_LETTERS},
                "answer": item["answer"],
                **fields,
                "generation_ok": complete,
                "generation_round": gen.get("round") if gen else None,
                "generation_error": gen.get("error") if gen else "no generation",
                "meta": item.get("meta", {}),
            }
            out_rows.append(row)

        out = release_full_path(dataset)
        write_jsonl(out, out_rows)
        report["datasets"][dataset] = dict(stats)
        print(f"[{dataset}] {len(out_rows)} released "
              f"(complete={stats['complete']}, incomplete={stats['incomplete']}, "
              f"missing={stats['missing_generation']}) -> {out}")

    path = REPORTS_DIR / "merge_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMerge report -> {path}")


if __name__ == "__main__":
    main()
