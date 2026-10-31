"""Run manifest: every non-target variable, frozen and logged.

`manifest_hash` covers exactly the fields that must be identical for a resume
to be legitimate.  Timestamps, server responses and progress counters are
deliberately outside it.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from .parse import PARSER_VERSION
from .prompts import PromptBook
from .sampling import request_template

SRC_ROOT = Path(__file__).resolve().parent


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha(repo: Path | None = None) -> str | None:
    repo = repo or SRC_ROOT
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


#: Modules whose content can change what a request or a parsed answer IS.
#: The metrics / aggregate / validate layers are deliberately excluded: editing
#: a metric must not block resuming a half-finished grid.
GENERATION_MODULES = ("schema.py", "normalize.py", "conditions.py", "prompts.py",
                      "sampling.py", "client.py", "parse.py", "runner.py",
                      "config.py", "io.py")


def code_sha256(modules=GENERATION_MODULES) -> str:
    """Hash of the generation-affecting harness source.  Stands in for the git
    SHA when the working tree is not a repository, and is strictly more honest
    than one when it is dirty."""
    h = hashlib.sha256()
    for name in sorted(modules):
        p = SRC_ROOT / name
        if not p.exists():
            continue
        h.update(name.encode())
        h.update(b"\x00")
        h.update(p.read_bytes())
        h.update(b"\x01")
    return h.hexdigest()


def server_version(models_payload: Any) -> str | None:
    try:
        for m in (models_payload or {}).get("data", []):
            for k in ("vllm_version", "version"):
                if m.get(k):
                    return str(m[k])
    except Exception:
        pass
    return None


#: Config keys that change what is SENT or what is WRITTEN onto a result row.
#: Resuming with any of these altered is silent corruption.  Everything else in
#: the config -- max_concurrency, timeout_s, max_attempts, api_key, base_url --
#: is operational: it affects throughput or routing, never the content of a
#: request or a row, and changing it between batches is ordinary (a model gets
#: redeployed on a different port; you dial concurrency up overnight).
RESUME_RELEVANT_CONFIG_KEYS = (
    "config_id", "model_name", "sampling", "system_prompt", "chat_template_kwargs",
    "seed_base", "structured_output", "capture_logprobs", "endpoint_kind",
    # analysis metadata is denormalized onto every result row, so a mid-run
    # change would leave the shard internally inconsistent
    "family", "params_total_b", "params_active_b", "is_moe", "kazakh_specialized",
    "reasoning_regime",
)

#: Fields that MUST be identical for a resume to be legitimate.  A crashed run
#: resumed with different sampling params is silent corruption; a second batch
#: covering different CONDITIONS is ordinary additive work, so the condition
#: list, repeat spec and item counts live in `run_scope` and are merged instead.
HASHED_FIELDS = ("config_behavioural", "request_body_template", "prompt_version",
                 "prompt_sha256", "system_prompt_sha256", "parser_version",
                 "seed_base", "shuffle_seed", "sample_index_sha256",
                 "harness_code_sha256")


def behavioural_config(cfg_dict: dict) -> dict:
    return {k: cfg_dict.get(k) for k in RESUME_RELEVANT_CONFIG_KEYS}


def build_manifest(cfg, defaults: dict, book: PromptBook, *, conditions: list[str],
                   datasets: list[str], repeats: dict[str, int],
                   item_counts: dict[str, int], sample_index_sha256: str,
                   dropped_params: set[str] | None = None,
                   probe_report: dict | None = None,
                   server_models: Any = None,
                   seed_honored: bool | None = None,
                   notes: list[str] | None = None) -> dict:
    m: dict[str, Any] = {
        "created_utc": utcnow(),
        "config": cfg.to_dict(),
        "config_behavioural": behavioural_config(cfg.to_dict()),
        "request_body_template": request_template(
            cfg, seed=cfg.seed_base, capture_logprobs=cfg.capture_logprobs,
            structured_output=cfg.structured_output, drop=dropped_params or set()),
        "prompt_version": book.version,
        "prompt_sha256": book.sha256,
        "system_prompt": cfg.system_prompt,
        "system_prompt_sha256": cfg.system_prompt_sha256,
        "parser_version": PARSER_VERSION,
        "conditions": sorted(conditions),
        "datasets": sorted(datasets),
        "repeats": {k: repeats[k] for k in sorted(repeats)},
        "item_counts": {k: item_counts[k] for k in sorted(item_counts)},
        "run_scope": [{"utc": utcnow(), "conditions": sorted(conditions),
                       "datasets": sorted(datasets),
                       "repeats": {k: repeats[k] for k in sorted(repeats)},
                       "item_counts": {k: item_counts[k] for k in sorted(item_counts)}}],
        "seed_base": cfg.seed_base,
        "shuffle_seed": defaults.get("shuffle_seed"),
        "sample_index_sha256": sample_index_sha256,
        "harness_git_sha": git_sha(),
        "harness_code_sha256": code_sha256(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "endpoint_kind": cfg.endpoint_kind,
        "dropped_params": sorted(dropped_params or []),
        "endpoint_probe": probe_report or {},
        "seed_honored": seed_honored,
        "server_models": server_models,
        "vllm_version": server_version(server_models),
        "structured_output": cfg.structured_output,
        "capture_logprobs": cfg.capture_logprobs,
        "notes": notes or [],
        "max_tokens_exceptions": [],
    }
    m["manifest_hash"] = manifest_hash(m)
    return m


def manifest_hash(m: dict) -> str:
    payload = {k: m.get(k) for k in HASHED_FIELDS}
    return hashlib.sha256(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)
    ).hexdigest()


def sha256_of_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        h.update(Path(p).name.encode())
        h.update(b"\x00")
        h.update(Path(p).read_bytes())
        h.update(b"\x01")
    return h.hexdigest()


def merge_manifest(old: dict, new: dict) -> dict:
    """Fold a compatible follow-up invocation into the stored manifest.

    The invariants (HASHED_FIELDS) are identical by construction -- the resume
    check refused otherwise -- so only the run scope accumulates.  Without this,
    a diagnostics batch would erase the main grid's expected row counts and
    06_aggregate would report the grid as an unexplained shortfall.
    """
    out = dict(old)
    out["updated_utc"] = new.get("created_utc")
    out["conditions"] = sorted(set(old.get("conditions", [])) | set(new.get("conditions", [])))
    out["datasets"] = sorted(set(old.get("datasets", [])) | set(new.get("datasets", [])))
    out["repeats"] = {**(old.get("repeats") or {}), **(new.get("repeats") or {})}
    out["item_counts"] = {**(old.get("item_counts") or {}), **(new.get("item_counts") or {})}
    out["run_scope"] = (old.get("run_scope") or []) + (new.get("run_scope") or [])
    for k in ("server_models", "vllm_version", "endpoint_probe", "dropped_params",
              "seed_honored"):
        if new.get(k) is not None:
            out[k] = new[k]
    out["notes"] = sorted(set(old.get("notes") or []) | set(new.get("notes") or []))
    out["manifest_hash"] = new["manifest_hash"]
    return out
