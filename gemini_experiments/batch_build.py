"""Phase 3 (build): turn the normalized corpus into Gemini Batch API request files.

    python -m gemini_experiments.batch_build
    python -m gemini_experiments.batch_build --round 2 --only-ids reports/invalid_ids.json

Output: gemini_experiments/work/batch_input/r{round}/{dataset}_{chunk:03d}.jsonl
Each line: {"key": <item id>, "request": {...GenerateContentRequest...}}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .config import (BATCH_CHUNK_SIZE, BATCH_INPUT_DIR, DATASETS, GEN_PARAMS,
                     PROJECT_ROOT, PROMPT_VERSION, SAFETY_SETTINGS,
                     normalized_path)
from .jsonl import read_jsonl, write_jsonl
from .prompts import RESPONSE_SCHEMA, SYSTEM_INSTRUCTION, build_user_prompt


def build_request(item: Dict[str, Any]) -> Dict[str, Any]:
    generation_config = dict(GEN_PARAMS)
    generation_config["response_schema"] = RESPONSE_SCHEMA
    return {
        "key": item["id"],
        "request": {
            "contents": [{"role": "user", "parts": [{"text": build_user_prompt(item)}]}],
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "generation_config": generation_config,
            "safety_settings": SAFETY_SETTINGS,
        },
    }


def chunked(rows: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    """size <= 0 means: the whole dataset in a single file (a single batch job)."""
    if size <= 0:
        yield rows
        return
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def load_id_filter(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict):                       # {"kazmmlu": [...], ...}
        ids: Set[str] = set()
        for value in data.values():
            ids.update(value)
        return ids
    return set(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Gemini batch request files.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--round", type=int, default=1, help="Generation round (2+ = regeneration).")
    ap.add_argument("--only-ids", default=None,
                    help="JSON list or {dataset: [ids]} of item ids to (re)generate.")
    ap.add_argument("--chunk-size", type=int, default=BATCH_CHUNK_SIZE,
                    help="Requests per batch job. 0 = one job per dataset.")
    ap.add_argument("--limit", type=int, default=0, help="Debug: cap items per dataset.")
    args = ap.parse_args()

    id_filter = load_id_filter(args.only_ids)
    out_root = BATCH_INPUT_DIR / f"r{args.round}"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"round": args.round, "prompt_version": PROMPT_VERSION, "files": []}
    total = 0

    for dataset in args.datasets:
        src = normalized_path(dataset)
        if not src.exists():
            print(f"[{dataset}] SKIP - {src} missing (run prepare_datasets first)")
            continue

        items = read_jsonl(src)
        if id_filter is not None:
            items = [it for it in items if it["id"] in id_filter]
        if args.limit:
            items = items[:args.limit]
        if not items:
            print(f"[{dataset}] nothing to generate")
            continue

        requests = [build_request(it) for it in items]
        for chunk_i, chunk in enumerate(chunked(requests, args.chunk_size)):
            path = out_root / f"{dataset}_{chunk_i:03d}.jsonl"
            write_jsonl(path, chunk)
            manifest["files"].append({
                "dataset": dataset,
                "chunk": chunk_i,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "abs_path": str(path),
                "n_requests": len(chunk),
            })
            print(f"[{dataset}] {path.name}: {len(chunk)} requests")
        total += len(requests)

    manifest_path = out_root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{total} requests across {len(manifest['files'])} files")
    print(f"Build manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
