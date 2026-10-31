#!/usr/bin/env python3
"""THE main entry point.  One deployed model per invocation.

    python scripts/05_run.py --config configs/models/gpt-oss-20b-high.yaml \
      --port 8003 --datasets kazmmlu kazculture kazbench_kk \
      --conditions B0 N1 N2 N3 N4 I1 O1 P1 --baseline-runs 3 --out results/
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from stability.client import StabilityClient, check_context_budget
from stability.conditions import ALL_CONDITIONS, MAIN_GRID
from stability.config import load_defaults, load_model_config
from stability.io import read_json, read_jsonl, write_json
from stability.manifest import build_manifest, sha256_of_files
from stability.normalize import DATASETS
from stability.prompts import load_prompts
from stability.runner import RunAborted, Runner
from stability.schema import ConditionItem


def _bool(x: str) -> bool:
    if str(x).lower() in ("true", "1", "yes", "on"):
        return True
    if str(x).lower() in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {x!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="results dir (default: defaults.yaml)")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--conditions", nargs="*", default=list(MAIN_GRID))
    p.add_argument("--baseline-runs", type=int, default=None,
                   help="shorthand for --repeat B0=<n>")
    p.add_argument("--repeat", action="append", default=[], metavar="COND=N")
    p.add_argument("--limit", type=int, default=None, help="first N items per cell (pilot)")

    # endpoint
    p.add_argument("--port", type=int, default=None, help="override the port in base_url")
    p.add_argument("--base-url", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--endpoint-kind", choices=["vllm", "hosted"], default=None)

    # sampling -- every one is optional; unset means the key is not sent at all
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument("--frequency-penalty", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)

    p.add_argument("--system-prompt", default=None)
    p.add_argument("--enable-thinking", type=_bool, default=None)
    p.add_argument("--chat-template-kwarg", action="append", default=[],
                   metavar="KEY=VALUE", help="generic chat_template_kwargs passthrough")
    p.add_argument("--seed-base", type=int, default=None)
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--timeout-s", type=int, default=None)
    p.add_argument("--capture-logprobs", action="store_true", default=None)
    p.add_argument("--structured-output", action="store_true", default=None,
                   help="OFF by default; forcing JSON destroys the compliance metric")

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-run-n", type=int, default=3)
    p.add_argument("--health-check", action="store_true",
                   help="probe /v1/models and exit")
    p.add_argument("--no-health-check", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force-resume", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="allow O1C on a config outside defaults.o1c_configs")
    p.add_argument("--prompts", default=None)
    return p.parse_args(argv)


def _overrides(a) -> dict:
    o: dict = {}
    for k in ("base_url", "model_name", "api_key", "endpoint_kind", "system_prompt",
              "seed_base", "max_concurrency", "timeout_s", "capture_logprobs",
              "structured_output"):
        v = getattr(a, k)
        if v is not None:
            o[k] = v
    for cli, key in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                     ("max_tokens", "max_tokens"), ("repetition_penalty", "repetition_penalty"),
                     ("presence_penalty", "presence_penalty"),
                     ("frequency_penalty", "frequency_penalty"), ("min_p", "min_p")):
        v = getattr(a, cli)
        if v is not None:
            o[f"sampling.{key}"] = v
    ctk: dict = {}
    if a.enable_thinking is not None:
        ctk["enable_thinking"] = a.enable_thinking
    for kv in a.chat_template_kwarg:
        k, _, v = kv.partition("=")
        if v.lower() in ("true", "false"):
            ctk[k] = v.lower() == "true"
        elif re.fullmatch(r"-?\d+", v):
            ctk[k] = int(v)
        else:
            ctk[k] = v
    if ctk:
        o["chat_template_kwargs"] = ctk
    return o


def _repeats(a, defaults) -> dict:
    reps = dict(defaults.get("repeats") or {"default": 1})
    reps.setdefault("default", 1)
    if a.baseline_runs is not None:
        reps["B0"] = a.baseline_runs
    for spec in a.repeat:
        k, _, v = spec.partition("=")
        if k not in ALL_CONDITIONS:
            raise SystemExit(f"--repeat: unknown condition {k!r}")
        reps[k] = int(v)
    return reps


def main(argv=None) -> int:
    a = parse_args(argv)
    defaults = load_defaults()
    cfg = load_model_config(a.config, defaults, _overrides(a))

    if a.port is not None:
        cfg.base_url = re.sub(r":\d+(/|$)", f":{a.port}\\1", cfg.base_url)
    results_dir = Path(a.out or defaults["results_dir"])
    work = Path(defaults["work_dir"])
    book = load_prompts(a.prompts) if a.prompts else load_prompts()

    conditions = list(dict.fromkeys(a.conditions))
    for c in conditions:
        if c not in ALL_CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; valid: {ALL_CONDITIONS}")
    datasets = a.datasets or defaults.get("datasets", list(DATASETS))
    repeats = _repeats(a, defaults)

    # O1-C runs only on the declared configurations, so the diagnostic does not
    # end up on whichever models happened to be up.
    if "O1C" in conditions:
        allowed = defaults.get("o1c_configs") or []
        if cfg.config_id not in allowed and not a.force:
            raise SystemExit(
                f"O1C is declared for {allowed}; {cfg.config_id!r} is not on the list. "
                "Add it to defaults.o1c_configs or pass --force.")

    # ---------------------------------------------------------- load items
    cond_items: dict[tuple[str, str], list[ConditionItem]] = {}
    item_counts: dict[str, int] = {}
    # --limit defines the pilot item set as the first N uids of the dataset's
    # SAMPLE INDEX, identically for every condition and every invocation.  Taking
    # the first N of each condition file instead would give the diagnostics a
    # different item set from B0, and their rows would silently fail to pair.
    pilot: dict[str, set[str]] = {}
    if a.limit:
        for ds in datasets:
            idx = work / "sample_index" / f"{ds}_sample.json"
            pilot[ds] = set(read_json(idx)["uids"][:a.limit])
    for ds in datasets:
        for cond in conditions:
            path = work / "conditions" / f"{ds}__{cond}.jsonl"
            if not path.exists():
                raise SystemExit(f"missing materialized condition file: {path}\n"
                                 "Run scripts/03_build_conditions.py first.")
            rows = [ConditionItem.from_dict(r) for r in read_jsonl(path)]
            rows.sort(key=lambda c: c.uid)
            n_full = len({r.parent_uid for r in rows})
            if a.limit:
                rows = [r for r in rows if r.parent_uid in pilot[ds]]
                n_kept = len({r.parent_uid for r in rows})
                if cond in ("O1C", "NQR", "NQI") and n_kept < min(a.limit, n_full) * 0.5:
                    print(f"[warn] --limit leaves {ds}/{cond} with {n_kept} items "
                          f"(the diagnostic pool is a subsample of the benchmark, so a "
                          f"small pilot barely intersects it). Raise --limit or drop "
                          f"{cond} from the pilot.")
            cond_items[(ds, cond)] = rows
            item_counts[f"{ds}__{cond}"] = len(rows)
    for ds in datasets:
        assert any(k[0] == ds for k in cond_items), f"{ds} fell out of the loop"

    # ---------------------------------------------------------- dry run
    if a.dry_run:
        for (ds, cond) in sorted(cond_items):
            rows = cond_items[(ds, cond)]
            print("=" * 78)
            print(f"{ds} / {cond}  ({len(rows)} rows)")
            for ci in rows[:a.dry_run_n]:
                print("-" * 78)
                print(f"uid={ci.uid} gold={ci.gold_answer} "
                      f"(orig {ci.gold_answer_original}) perm={ci.option_permutation} "
                      f"rot={ci.rotation_index}")
                if cfg.system_prompt:
                    print(f"[system] {cfg.system_prompt}")
                print("[user]")
                print(book.render(ci))
        return 0

    # longest rendered prompt across the cells this invocation will run
    max_prompt_tokens = None
    build_report = work / "conditions" / "_build_report.json"
    if build_report.exists():
        cells = (read_json(build_report) or {}).get("cells", {})
        lens = [c.get("prompt_tokens_max", 0) for key, c in cells.items()
                if key.split("__")[0] in datasets and key.split("__")[1] in conditions]
        if lens:
            max_prompt_tokens = max(lens)

    # ---------------------------------------------------------- manifest
    sample_files = sorted((work / "sample_index").glob("*.json"))
    manifest_stub = dict(conditions=conditions, datasets=datasets, repeats=repeats,
                         item_counts=item_counts,
                         sample_index_sha256=sha256_of_files(sample_files))

    runner = Runner(cfg, defaults, book, results_dir=results_dir,
                    overwrite=a.overwrite, force_resume=a.force_resume)
    client = StabilityClient(cfg, on_warning=runner.warn)
    runner.client = client

    async def go() -> int:
        notes: list[str] = []
        server_models = None
        if not a.no_health_check:
            try:
                server_models = await client.health_check()
            except Exception as exc:                     # noqa: BLE001
                raise RunAborted(
                    f"health check failed against {cfg.base_url}\n"
                    f"  {type(exc).__name__}: {exc}\n"
                    "  Is a server listening on that port, is it the RIGHT one, and is "
                    "it healthy?\n"
                    "  Check with:  curl -s "
                    f"{cfg.base_url.rstrip('/')}/models\n"
                    "  Two servers up and requests going to the wrong one is exactly "
                    "what this check exists to catch.") from None
            print(f"[health] {cfg.base_url} serves "
                  f"{[m.get('id') for m in server_models.get('data', [])]}")

            # prompt + max_tokens must fit in --max-model-len, or every request
            # 400s deterministically.  Catch it here, not six hours in.
            ok, msg = check_context_budget(
                server_models, cfg.sampling.get("max_tokens"), max_prompt_tokens)
            print(f"[health] {msg}")
            if not ok:
                raise RunAborted(msg)
        if a.health_check:
            write_json(results_dir / cfg.config_id / "health_check.json", server_models)
            print(f"[health] written to "
                  f"{results_dir / cfg.config_id / 'health_check.json'}")
            return 0
        if cfg.endpoint_kind == "hosted":
            probe = await client.probe_parameters()
            print(f"[probe] {probe}")
            if client.seed_honored is False:
                notes.append("endpoint did not honour `seed`: the three baseline runs "
                             "measure a floor over an UNSEEDED sampler and are not "
                             "byte-reproducible. Say so in the reproducibility section.")

        manifest = build_manifest(
            cfg, defaults, book, dropped_params=client.dropped_params,
            probe_report=client.probe_report, server_models=server_models,
            seed_honored=client.seed_honored, notes=notes, **manifest_stub)
        runner.check_resume_manifest(manifest)

        done = runner.completed_keys()
        tasks = runner.build_tasks(cond_items, repeats, done)
        total = sum(len(v) * repeats.get(k[1], repeats["default"])
                    for k, v in cond_items.items())
        # count only the completed keys this invocation is responsible for --
        # a previous batch of OTHER conditions is not part of "total"
        scope = {k[1] for k in cond_items}
        done_here = sum(1 for (_uid, cond, _run) in done if cond in scope)
        print(f"[plan] {len(tasks)} requests to issue "
              f"({done_here} of {total} already complete for these conditions"
              + (f"; {len(done) - done_here} more on disk from earlier batches)"
                 if len(done) > done_here else ")"))
        if not tasks:
            print("[plan] nothing to do")
            mpath = results_dir / cfg.config_id / "manifest.json"
            if mpath.exists() and not a.overwrite:
                from stability.manifest import merge_manifest
                manifest = merge_manifest(read_json(mpath), manifest)
            write_json(mpath, manifest)
            return 0

        stats = await runner.run(tasks, manifest)
        print(f"[done] {runner.status_line()}")
        write_json(results_dir / cfg.config_id / "logs" / "run_stats.json",
                   {"stats": stats, "warnings": runner.warnings})
        return 0

    try:
        return asyncio.run(go())
    except RunAborted as exc:
        print(f"[ABORT] {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            asyncio.run(client.aclose())
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
