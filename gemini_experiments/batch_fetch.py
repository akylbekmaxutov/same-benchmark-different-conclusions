"""Phase 3 (fetch): poll the Gemini batch jobs, download results, parse them.

    python -m gemini_experiments.batch_fetch --round 1 --wait

Produces gemini_experiments/work/generations/r{round}/{dataset}.jsonl with one
record per source item:
    {"id", "round", "raw_text", "relevant_noise", "irrelevant_noise",
     "question_paraphrase", "error"}
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import (BATCH_OUTPUT_DIR, BATCH_POLL_SECONDS, GENERATIONS_DIR,
                     REPORTS_DIR)
from .batch_submit import jobs_path, load_jobs, save_jobs
from .gemini_client import (TERMINAL_STATES, download_result_file, get_batch_job,
                            get_client, iter_batch_results, job_error, job_state,
                            result_file_name)
from .jsonl import write_jsonl

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
FIELDS = ("relevant_noise", "irrelevant_noise", "question_paraphrase")


def parse_generation(text: Optional[str]) -> Tuple[Dict[str, Optional[str]], Optional[str]]:
    """Return ({field: value}, error). Tolerates fences and trailing prose."""
    empty = {f: None for f in FIELDS}
    if not text or not text.strip():
        return empty, "empty response"

    cleaned = FENCE_RE.sub("", text.strip())
    obj: Any = None
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = OBJ_RE.search(cleaned)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                return empty, f"invalid JSON: {exc}"
        else:
            return empty, "no JSON object found"

    if not isinstance(obj, dict):
        return empty, f"expected a JSON object, got {type(obj).__name__}"

    out: Dict[str, Optional[str]] = {}
    missing = []
    for field in FIELDS:
        value = obj.get(field)
        if isinstance(value, str) and value.strip():
            out[field] = value.strip()
        else:
            out[field] = None
            missing.append(field)
    return out, (f"missing/empty fields: {','.join(missing)}" if missing else None)


MAX_POLL_FAILURES = 10


def poll(client, state: Dict[str, Any], round_no: int, wait: bool, timeout_s: int) -> Dict[str, Any]:
    deadline = time.time() + timeout_s if timeout_s else None
    failures: collections.Counter = collections.Counter()
    while True:
        pending = []
        for key, job in state["jobs"].items():
            if job.get("state") in TERMINAL_STATES:
                continue
            try:
                live = get_batch_job(client, job["job_name"])
            except Exception as exc:
                failures[key] += 1
                print(f"[{key}] poll failed ({failures[key]}/{MAX_POLL_FAILURES}): {exc}")
                if failures[key] >= MAX_POLL_FAILURES:
                    job["state"] = "JOB_STATE_FAILED"
                    job["error"] = f"unreadable after {MAX_POLL_FAILURES} polls: {exc}"
                else:
                    pending.append(key)
                continue
            failures[key] = 0
            job["state"] = job_state(live)
            job["result_file"] = result_file_name(live)
            job["error"] = job_error(live)
            if job["state"] not in TERMINAL_STATES:
                pending.append(key)
        save_jobs(round_no, state)

        done = len(state["jobs"]) - len(pending)
        print(f"  {done}/{len(state['jobs'])} job(s) terminal; pending: {pending or 'none'}")
        if not pending or not wait:
            return state
        if deadline and time.time() > deadline:
            print("  poll timeout reached; leaving remaining jobs pending")
            return state
        time.sleep(BATCH_POLL_SECONDS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch and parse Gemini batch results.")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--wait", action="store_true", help="Block until every job is terminal.")
    ap.add_argument("--timeout", type=int, default=0, help="Max seconds to wait (0 = forever).")
    args = ap.parse_args()

    if not jobs_path(args.round).exists():
        raise SystemExit(f"{jobs_path(args.round)} missing - run batch_submit first.")

    client = get_client()
    state = load_jobs(args.round)
    print(f"Polling {len(state['jobs'])} job(s) for round {args.round} ...")
    state = poll(client, state, args.round, args.wait, args.timeout)

    out_dir = GENERATIONS_DIR / f"r{args.round}"
    raw_dir = BATCH_OUTPUT_DIR / f"r{args.round}"
    per_dataset: Dict[str, list] = collections.defaultdict(list)
    stats = collections.Counter()

    for key, job in sorted(state["jobs"].items()):
        st = job.get("state")
        if st != "JOB_STATE_SUCCEEDED":
            print(f"[{key}] state={st} error={job.get('error')} - no results")
            stats["jobs_not_succeeded"] += 1
            continue

        file_name = job.get("result_file")
        if not file_name:
            print(f"[{key}] succeeded but no result file reported")
            stats["jobs_without_result_file"] += 1
            continue

        local = raw_dir / f"{Path(key).stem}_results.jsonl"
        if not local.exists():
            print(f"[{key}] downloading results ...")
            download_result_file(client, file_name, local)
        else:
            print(f"[{key}] using cached {local.name}")

        for res in iter_batch_results(local):
            item_id = res["key"]
            if not item_id:
                stats["results_without_key"] += 1
                continue
            fields, parse_err = parse_generation(res["text"])
            error = res["error"] or parse_err
            per_dataset[job["dataset"]].append({
                "id": item_id,
                "round": args.round,
                "raw_text": res["text"],
                **fields,
                "error": error,
            })
            stats["ok" if error is None else "failed"] += 1

    summary = {"round": args.round, "counts": dict(stats), "per_dataset": {}}
    for dataset, rows in sorted(per_dataset.items()):
        rows.sort(key=lambda r: r["id"])
        path = out_dir / f"{dataset}.jsonl"
        write_jsonl(path, rows)
        n_bad = sum(1 for r in rows if r["error"])
        summary["per_dataset"][dataset] = {"n": len(rows), "n_failed": n_bad}
        print(f"[{dataset}] {len(rows)} generations ({n_bad} failed) -> {path}")

    report = REPORTS_DIR / f"fetch_report_r{args.round}.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFetch report -> {report}")


if __name__ == "__main__":
    main()
