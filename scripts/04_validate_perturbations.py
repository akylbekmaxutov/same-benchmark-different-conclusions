#!/usr/bin/env python3
"""Offline perturbation validation -> results/validation/*.json + report.md."""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from stability.conditions import MAIN_GRID, DIAGNOSTICS
from stability.config import load_defaults
from stability.io import read_json, read_jsonl, write_json, write_text
from stability.normalize import DATASETS, iter_normalized
from stability.schema import ConditionItem
from stability.validate import (embedding_screen, id_alignment, leakage,
                                missing_fields, noise_lengths, option_byte_identity,
                                paraphrase_leaves_options)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-model", default=None,
                    help="e.g. sentence-transformers/LaBSE; off by default")
    args = ap.parse_args()

    d = load_defaults()
    work = Path(d["work_dir"])
    outdir = Path(d["results_dir"]) / "validation"

    normalized = {ds: list(iter_normalized(work / "normalized" / f"{ds}.jsonl"))
                  for ds in d.get("datasets", DATASETS)}
    assert set(normalized) == set(DATASETS), "a benchmark fell out of the loop"

    report: dict = {"id_alignment": id_alignment(normalized), "datasets": {}}

    for ds, items in normalized.items():
        by_uid = {i.uid: i for i in items}
        cond_items: list[ConditionItem] = []
        for cond in list(MAIN_GRID) + list(DIAGNOSTICS):
            p = work / "conditions" / f"{ds}__{cond}.jsonl"
            if p.exists():
                cond_items += [ConditionItem.from_dict(r) for r in read_jsonl(p)]

        entry = {
            "n_items": len(items),
            "missing_generated_fields": missing_fields(items),
            "gold_leakage": leakage(items),
            "noise_lengths": noise_lengths(items),
            "option_integrity": option_byte_identity(by_uid, cond_items),
            "p1_options_mutated": paraphrase_leaves_options(by_uid, cond_items),
            "n_with_option_e": sum(1 for i in items if i.has_option_e),
        }
        if ds == "kazbench_kk":
            entry["split_note"] = (
                "repository names the split `train`; the accompanying paper describes "
                "it as the test set. Treated as the evaluation set "
                "(meta.split_source='train', meta.split_role='test_per_paper'). "
                "No contamination analysis is performed on this basis.")
        if args.embedding_model:
            entry["embedding_screen"] = embedding_screen(items, args.embedding_model)
        report["datasets"][ds] = entry

    for ds in ("kazmmlu", "kazculture", "kazbench_kk"):
        idx = read_json(work / "sample_index" / f"{ds}_sample.json")
        report["datasets"][ds]["sample_index"] = {
            k: idx[k] for k in ("n_available", "n_target", "n_sampled",
                                "sampling_applied", "stratify_key", "seed")}

    write_json(outdir / "validation.json", report)

    L = ["# Perturbation validation", ""]
    ali = report["id_alignment"]["_global"]
    L += [f"IDs unique across all three files: **{ali['unique_across_files']}** "
          f"({ali['n']} total).", ""]
    L += ["| benchmark | n | missing rel/irr/para | opt mutated | bad perm | "
          "P1 opts mutated | with E | sampled |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for ds, e in report["datasets"].items():
        m = e["missing_generated_fields"]
        oi = e["option_integrity"]
        L.append(f"| {ds} | {e['n_items']} | "
                 f"{m['relevant_noise']}/{m['irrelevant_noise']}/{m['question_paraphrase']} | "
                 f"{oi['n_option_text_mutated']} | {oi['n_bad_permutation']} | "
                 f"{e['p1_options_mutated']} | {e['n_with_option_e']} | "
                 f"{e['sample_index']['n_sampled']} |")
    L += ["", "## Gold-answer leakage (normalized substring of the gold option)", "",
          "`distractor_hit_rate` is the base rate: a gold rate at or below it is "
          "topical overlap, not leakage.", "",
          "| benchmark | field | n | gold hit rate | distractor base rate |",
          "|---|---|---:|---:|---:|"]
    for ds, e in report["datasets"].items():
        for field, s in e["gold_leakage"]["per_field"].items():
            L.append(f"| {ds} | {field} | {s['n_tested']} | {s['gold_hit_rate']:.4f} | "
                     f"{s['distractor_hit_rate']:.4f} |")
    L += ["", "## Noise length comparability (characters)", "",
          "| benchmark | field | n | mean | median | p05 | p95 |", "|---|---|---:|---:|---:|---:|---:|"]
    for ds, e in report["datasets"].items():
        for field, s in e["noise_lengths"].items():
            if s.get("n"):
                L.append(f"| {ds} | {field} | {s['n']} | {s['chars_mean']} | "
                         f"{s['chars_median']} | {s['chars_p05']} | {s['chars_p95']} |")
    L += ["", "## Notes", ""]
    for ds, e in report["datasets"].items():
        if "split_note" in e:
            L.append(f"- **{ds}**: {e['split_note']}")
        si = e["sample_index"]
        if not si["sampling_applied"]:
            L.append(f"- **{ds}**: only {si['n_available']} items available against a target "
                     f"of {si['n_target']}; the small-benchmark rule applied — all items used, "
                     f"no upsampling, the other benchmarks were not shrunk to match.")
    write_text(outdir / "report.md", "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
