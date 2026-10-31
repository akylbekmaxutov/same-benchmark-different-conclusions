"""Phase 3 (submit): upload the batch request files and create Gemini batch jobs.

    python -m gemini_experiments.batch_submit --round 1

Writes gemini_experiments/work/batch_input/r{round}/jobs.json, which
`batch_fetch` polls. Re-running is safe: files that already have a live job are
skipped unless --resubmit is passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import BATCH_INPUT_DIR, GEMINI_MODEL, PROMPT_VERSION
from .gemini_client import (create_batch_job, get_batch_job, get_client,
                            job_state, upload_batch_file)


def jobs_path(round_no: int) -> Path:
    return BATCH_INPUT_DIR / f"r{round_no}" / "jobs.json"


def load_jobs(round_no: int) -> Dict[str, Any]:
    path = jobs_path(round_no)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"round": round_no, "model": GEMINI_MODEL,
            "prompt_version": PROMPT_VERSION, "jobs": {}}


def save_jobs(round_no: int, state: Dict[str, Any]) -> None:
    path = jobs_path(round_no)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit Gemini batch jobs.")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--model", default=GEMINI_MODEL)
    ap.add_argument("--resubmit", action="store_true",
                    help="Recreate jobs even if a job already exists for a file.")
    args = ap.parse_args()

    build_manifest = BATCH_INPUT_DIR / f"r{args.round}" / "build_manifest.json"
    if not build_manifest.exists():
        raise SystemExit(f"{build_manifest} missing - run batch_build --round {args.round} first.")
    manifest = json.loads(build_manifest.read_text(encoding="utf-8"))

    client = get_client()
    state = load_jobs(args.round)
    state["model"] = args.model

    for entry in manifest["files"]:
        path = Path(entry["abs_path"])
        key = path.name

        existing = state["jobs"].get(key)
        if existing and not args.resubmit:
            try:
                live = get_batch_job(client, existing["job_name"])
                print(f"[{key}] already submitted ({job_state(live)}) - skipping")
                continue
            except Exception as exc:                       # job vanished / expired
                print(f"[{key}] existing job unreadable ({exc}); resubmitting")

        display = f"stability-{args.round}-{path.stem}"
        print(f"[{key}] uploading {entry['n_requests']} requests ...")
        file_name = upload_batch_file(client, path, display)
        job = create_batch_job(client, file_name, display, model=args.model)
        job_name = str(getattr(job, "name", job))

        state["jobs"][key] = {
            "dataset": entry["dataset"],
            "chunk": entry["chunk"],
            "input_path": str(path),
            "n_requests": entry["n_requests"],
            "uploaded_file": file_name,
            "job_name": job_name,
            "display_name": display,
            "state": job_state(job),
        }
        save_jobs(args.round, state)
        print(f"[{key}] job {job_name} ({job_state(job)})")

    save_jobs(args.round, state)
    print(f"\n{len(state['jobs'])} job(s) recorded -> {jobs_path(args.round)}")
    print(f"Next: python -m gemini_experiments.batch_fetch --round {args.round} --wait")


if __name__ == "__main__":
    main()
