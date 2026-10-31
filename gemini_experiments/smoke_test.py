"""Cheap pre-flight check: send a handful of live (non-batch) Gemini requests so
prompt, model id, schema and parser are verified before a full batch is paid for.

    python -m gemini_experiments.smoke_test --n 3
"""
from __future__ import annotations

import argparse
import json

from .config import (DATASETS, GEMINI_MODEL, GEN_PARAMS, SAFETY_SETTINGS,
                     normalized_path)
from .batch_fetch import parse_generation
from .gemini_client import extract_text, get_client
from .jsonl import read_jsonl
from .prompts import RESPONSE_SCHEMA, SYSTEM_INSTRUCTION, build_user_prompt
from .validate import validate_item


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the Gemini generation prompt.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=2, help="Items per dataset.")
    ap.add_argument("--model", default=GEMINI_MODEL)
    args = ap.parse_args()

    client = get_client()
    config = dict(GEN_PARAMS)
    config["system_instruction"] = SYSTEM_INSTRUCTION
    config["response_schema"] = RESPONSE_SCHEMA
    config["safety_settings"] = SAFETY_SETTINGS

    failures = 0
    for dataset in args.datasets:
        path = normalized_path(dataset)
        if not path.exists():
            print(f"[{dataset}] SKIP - {path} missing")
            continue
        items = read_jsonl(path)[:args.n]
        for item in items:
            print(f"\n=== {item['id']} ({item['category']}) ===")
            resp = client.models.generate_content(
                model=args.model,
                contents=build_user_prompt(item),
                config=config,
            )
            text = extract_text(resp)
            fields, err = parse_generation(text)
            print(json.dumps(fields, ensure_ascii=False, indent=2))
            if err:
                print(f"  PARSE ERROR: {err}")
                failures += 1
                continue
            merged = {**item, **fields, "generation_ok": True, "generation_error": None}
            errs, warns = validate_item(merged, item)
            print(f"  validation errors:   {errs or 'none'}")
            print(f"  validation warnings: {warns or 'none'}")
            failures += bool(errs)

    print(f"\nSmoke test finished with {failures} problematic item(s).")
    print(f"Model: {args.model}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
