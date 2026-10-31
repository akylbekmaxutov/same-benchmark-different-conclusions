#!/usr/bin/env python3
"""Render every condition for one chosen item per benchmark (GUIDELINE §16.3).

Not part of the pipeline -- a reviewable dump of exactly what the server sees.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from stability.conditions import ALL_CONDITIONS, MAIN_GRID, build, o1c_eligible
from stability.config import load_defaults
from stability.io import read_json, write_text
from stability.normalize import iter_normalized
from stability.prompts import load_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", action="append", default=[],
                    help="uid to render; repeat. Default: auto-pick one per benchmark")
    ap.add_argument("--conditions", nargs="*", default=list(MAIN_GRID))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load_defaults()
    work = Path(d["work_dir"])
    book = load_prompts()
    seed = d["shuffle_seed"]

    chosen: list = []
    if args.uid:
        wanted = set(args.uid)
        for ds in d["datasets"]:
            chosen += [i for i in iter_normalized(work / "normalized" / f"{ds}.jsonl")
                       if i.uid in wanted]
    else:
        for ds in d["datasets"]:
            sample = set(read_json(work / "sample_index" / f"{ds}_sample.json")["uids"])
            for it in iter_normalized(work / "normalized" / f"{ds}.jsonl"):
                if it.uid not in sample:
                    continue
                if ds == "kazmmlu" and not it.has_option_e:
                    continue          # show a five-option item
                if ds == "kazbench_kk" and not it.subcategory:
                    continue          # show both levels populated
                chosen.append(it)
                break

    out: list[str] = []
    for it in chosen:
        out += ["#" * 78,
                f"# {it.uid}   dataset={it.dataset}",
                f"# category={it.category!r}  subcategory={it.subcategory!r}",
                f"# gold={it.answer}  n_options={it.n_options}  has_E={it.has_option_e}",
                "#" * 78, ""]
        conds = list(args.conditions)
        if o1c_eligible(it) and "O1C" not in conds:
            conds.append("O1C")
        for code in conds:
            if code not in ALL_CONDITIONS:
                continue
            try:
                units = build(it, code, seed)
            except Exception as exc:                    # noqa: BLE001
                out += [f"--- {code}: SKIPPED ({exc})", ""]
                continue
            for ci in units:
                tag = code if ci.rotation_index is None else f"{code} rot{ci.rotation_index}"
                out += ["=" * 74,
                        f"--- {tag}   gold={ci.gold_answer} (orig {ci.gold_answer_original})  "
                        f"perm={ci.option_permutation}  slot={ci.gold_position_index}",
                        "=" * 74,
                        book.render(ci), ""]
    text = "\n".join(out)
    print(text)
    if args.out:
        write_text(args.out, text)


if __name__ == "__main__":
    main()
