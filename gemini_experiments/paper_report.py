"""Assemble every number and artifact the paper needs about dataset generation
(Phase 8 bookkeeping + Phase 32 generation materials) into one place.

    python -m gemini_experiments.paper_report

Writes:
  reports/generation_provenance.json   machine-readable record of the whole run
  reports/paper_appendix.md            appendix-ready tables
  reports/prompts/r{n}_system.txt      the EXACT system instruction sent in round n
  reports/prompts/r{n}_user_example.txt  a fully rendered user prompt from round n

The per-round prompt text is recovered from the batch input files that were
actually uploaded, not from the current prompts.py - so it stays correct even
after the prompt template is edited between rounds.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (BATCH_INPUT_DIR, DATASETS, GENERATIONS_DIR, GEN_PARAMS,
                     HF_SOURCES, HF_SPLITS, REPORTS_DIR, normalized_path)
from .jsonl import read_jsonl
from .validate import error_key, validate_item

OPTIONAL_FIELDS = ("relevant_noise", "irrelevant_noise", "question_paraphrase")


def rounds_present() -> List[int]:
    return sorted(int(p.name[1:]) for p in GENERATIONS_DIR.glob("r*")
                  if p.is_dir() and p.name[1:].isdigit())


def generations_upto(dataset: str, max_round: int) -> Dict[str, Dict[str, Any]]:
    """The generation state as it stood after round `max_round`."""
    best: Dict[str, Dict[str, Any]] = {}
    for rnd in rounds_present():
        if rnd > max_round:
            break
        path = GENERATIONS_DIR / f"r{rnd}" / f"{dataset}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            prev = best.get(row["id"])
            if prev is None or (row.get("error") is None or prev.get("error") is not None):
                best[row["id"]] = row
    return best


def evaluate_state(dataset: str, max_round: int) -> Dict[str, Any]:
    """Re-run the acceptance checks on the state after a given round."""
    sources = {r["id"]: r for r in read_jsonl(normalized_path(dataset))}
    gens = generations_upto(dataset, max_round)

    reasons: collections.Counter = collections.Counter()
    n_rejected = n_warned = 0
    for item_id, source in sources.items():
        gen = gens.get(item_id)
        merged = {
            **source,
            **{f: (gen.get(f) if gen else None) for f in OPTIONAL_FIELDS},
            "generation_ok": bool(gen and all(gen.get(f) for f in OPTIONAL_FIELDS)),
            "generation_error": gen.get("error") if gen else "no generation",
        }
        errs, warns = validate_item(merged, source)
        if errs:
            n_rejected += 1
            for e in errs:
                reasons[error_key(e)] += 1
        if warns:
            n_warned += 1

    n = len(sources)
    return {
        "n_items": n,
        "n_rejected": n_rejected,
        "n_accepted": n - n_rejected,
        "n_with_warnings": n_warned,
        "acceptance_rate": round((n - n_rejected) / n, 4) if n else 0.0,
        "rejection_reasons": dict(reasons.most_common()),
    }


def round_metadata(rnd: int) -> Dict[str, Any]:
    jobs_file = BATCH_INPUT_DIR / f"r{rnd}" / "jobs.json"
    if not jobs_file.exists():
        return {}
    data = json.loads(jobs_file.read_text(encoding="utf-8"))
    per_dataset = {}
    for job in data.get("jobs", {}).values():
        per_dataset[job["dataset"]] = {
            "n_requests": job["n_requests"],
            "job_state": job.get("state"),
        }
    return {
        "round": rnd,
        "gemini_model": data.get("model"),
        "prompt_version": data.get("prompt_version"),
        "n_batch_jobs": len(data.get("jobs", {})),
        "requests_per_dataset": per_dataset,
        "total_requests": sum(j["n_requests"] for j in per_dataset.values()),
    }


def extract_prompts(rnd: int) -> Optional[Dict[str, str]]:
    """Recover the exact prompt text sent in a round from its batch input file."""
    in_dir = BATCH_INPUT_DIR / f"r{rnd}"
    files = sorted(p for p in in_dir.glob("*.jsonl") if p.name != "jobs.json")
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            line = fh.readline().strip()
        if not line:
            continue
        request = json.loads(line)["request"]
        system = request["system_instruction"]["parts"][0]["text"]
        user = request["contents"][0]["parts"][0]["text"]
        out_dir = REPORTS_DIR / "prompts"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"r{rnd}_system.txt").write_text(system, encoding="utf-8")
        (out_dir / f"r{rnd}_user_example.txt").write_text(user, encoding="utf-8")
        return {"system": system, "user_example": user,
                "generation_config": request.get("generation_config", {}),
                "safety_settings": request.get("safety_settings", [])}
    return None


def build() -> Dict[str, Any]:
    rnds = rounds_present()
    provenance: Dict[str, Any] = {
        "corpus": {ds: {"source_repo": HF_SOURCES[ds], "split_used": HF_SPLITS[ds]}
                   for ds in DATASETS},
        "rounds": [],
        "acceptance_by_round": {},
        "final": {},
    }

    for rnd in rnds:
        meta = round_metadata(rnd)
        prompts = extract_prompts(rnd)
        if prompts:
            meta["generation_config"] = prompts["generation_config"]
            meta["safety_settings"] = prompts["safety_settings"]
            meta["system_instruction_file"] = f"reports/prompts/r{rnd}_system.txt"
            meta["user_prompt_example_file"] = f"reports/prompts/r{rnd}_user_example.txt"
        provenance["rounds"].append(meta)

    for rnd in rnds:
        provenance["acceptance_by_round"][f"r{rnd}"] = {
            ds: evaluate_state(ds, rnd) for ds in DATASETS
            if normalized_path(ds).exists()
        }

    if rnds:
        final = provenance["acceptance_by_round"][f"r{rnds[-1]}"]
        totals = {
            "n_items": sum(v["n_items"] for v in final.values()),
            "n_accepted": sum(v["n_accepted"] for v in final.values()),
            "n_rejected": sum(v["n_rejected"] for v in final.values()),
        }
        totals["acceptance_rate"] = round(totals["n_accepted"] / totals["n_items"], 4)
        provenance["final"] = {"per_dataset": final, "overall": totals}
    return provenance


def render_markdown(p: Dict[str, Any]) -> str:
    L: List[str] = ["# Dataset generation — provenance and validation\n",
                    "\nAuto-generated by `python -m gemini_experiments.paper_report`.\n"]

    L.append("\n## Sources\n\n| dataset | repository | split used |\n|---|---|---|\n")
    for ds, c in p["corpus"].items():
        note = " (labelled `train` upstream; used as the evaluation set)" \
            if ds == "kazbench_kk" else ""
        L.append(f"| {ds} | `{c['source_repo']}` | `{c['split_used']}`{note} |\n")

    L.append("\n## Generation rounds\n\n"
             "| round | model | prompt | batch jobs | requests | max_output_tokens | temp |\n"
             "|---|---|---|---|---|---|---|\n")
    for r in p["rounds"]:
        gc = r.get("generation_config", {})
        L.append(f"| {r.get('round')} | `{r.get('gemini_model')}` | {r.get('prompt_version')} "
                 f"| {r.get('n_batch_jobs')} | {r.get('total_requests')} "
                 f"| {gc.get('max_output_tokens', '')} | {gc.get('temperature', '')} |\n")
    L.append("\nRequests per dataset per round:\n\n| round | " +
             " | ".join(DATASETS) + " |\n|---|" + "---|" * len(DATASETS) + "\n")
    for r in p["rounds"]:
        cells = [str(r.get("requests_per_dataset", {}).get(ds, {}).get("n_requests", 0))
                 for ds in DATASETS]
        L.append(f"| {r.get('round')} | " + " | ".join(cells) + " |\n")

    L.append("\n## Acceptance rate after each round\n\n"
             "All rounds are scored with the *final* validator, so the numbers are "
             "directly comparable.\n\n"
             "| round | dataset | items | accepted | rejected | acceptance |\n"
             "|---|---|---|---|---|---|\n")
    for rkey, per_ds in p["acceptance_by_round"].items():
        for ds, s in per_ds.items():
            L.append(f"| {rkey} | {ds} | {s['n_items']} | {s['n_accepted']} "
                     f"| {s['n_rejected']} | {s['acceptance_rate']:.1%} |\n")

    if p.get("final"):
        o = p["final"]["overall"]
        L.append(f"\n**Final: {o['n_accepted']} / {o['n_items']} accepted "
                 f"({o['acceptance_rate']:.1%}), {o['n_rejected']} rejected.**\n")

        L.append("\n## Final rejection reasons\n\n| dataset | reason | n |\n|---|---|---|\n")
        for ds, s in p["final"]["per_dataset"].items():
            for reason, n in s["rejection_reasons"].items():
                L.append(f"| {ds} | {reason} | {n} |\n")

    L.append("\n## Released generation materials\n\n"
             "- Exact prompts per round: `reports/prompts/r*_system.txt`, "
             "`reports/prompts/r*_user_example.txt`\n"
             "- Batch requests actually sent: `work/batch_input/r*/`\n"
             "- Raw Gemini responses: `work/batch_output/r*/`\n"
             "- Parsed generations per round: `work/generations/r*/`\n"
             "- Released datasets: `data/perturbations/*_perturbations.jsonl`\n"
             "- Manual inspection sheets: `reports/manual_inspection/*.md`\n")
    return "".join(L)


def main() -> None:
    provenance = build()
    out_json = REPORTS_DIR / "generation_provenance.json"
    out_json.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = REPORTS_DIR / "paper_appendix.md"
    out_md.write_text(render_markdown(provenance), encoding="utf-8")

    if provenance.get("final"):
        o = provenance["final"]["overall"]
        print(f"Final acceptance: {o['n_accepted']}/{o['n_items']} "
              f"({o['acceptance_rate']:.1%})")
    print(f"Provenance -> {out_json}")
    print(f"Appendix   -> {out_md}")


if __name__ == "__main__":
    main()
