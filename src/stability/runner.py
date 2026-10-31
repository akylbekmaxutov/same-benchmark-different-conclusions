"""Run orchestration: resume, retries, ordering, guards, shards, manifest."""
from __future__ import annotations

import asyncio
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from .client import Completion, StabilityClient
from .conditions import CONDITION_GROUP
from .io import JsonlAppender, read_json, read_jsonl, write_json
from .manifest import build_manifest, manifest_hash, merge_manifest, utcnow
from .parse import extract_answer_logprobs, contains_think_tags, parse_answer
from .prompts import PromptBook
from .schema import ConditionItem, blank_result_row

REASONING_LOG_LIMIT = 20_000


class RunAborted(RuntimeError):
    pass


def shard_path(results_dir, config_id: str, dataset: str, condition: str, run_index: int) -> Path:
    return (Path(results_dir) / config_id / "raw"
            / f"{dataset}__{condition}__run{run_index}.jsonl")


def parse_shard_name(path: Path) -> tuple[str, str, int] | None:
    stem = path.stem
    parts = stem.split("__")
    if len(parts) != 3 or not parts[2].startswith("run"):
        return None
    try:
        return parts[0], parts[1], int(parts[2][3:])
    except ValueError:
        return None


@dataclass
class Task:
    ci: ConditionItem
    run_index: int
    seed: int


class Runner:
    def __init__(self, cfg, defaults: dict, book: PromptBook, *,
                 results_dir: str | Path, dry_run: bool = False,
                 overwrite: bool = False, force_resume: bool = False,
                 progress_every: int = 1000):
        self.cfg = cfg
        self.defaults = defaults
        self.book = book
        self.results_dir = Path(results_dir)
        self.out_dir = self.results_dir / cfg.config_id
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.force_resume = force_resume
        self.progress_every = progress_every
        self.guards = dict(defaults.get("guards") or {})
        self.warnings: list[str] = []
        self.client: StabilityClient | None = None

        self.stat = Counter()
        self.n_done = 0
        self._first_n_seen = 0
        self._first_n_parse_fail = 0
        self._appenders: dict[tuple, JsonlAppender] = {}

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[warn] {msg}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------- resume
    def completed_keys(self) -> set[tuple[str, str, int]]:
        done: set[tuple[str, str, int]] = set()
        raw = self.out_dir / "raw"
        if self.overwrite or not raw.exists():
            return done
        for path in sorted(raw.glob("*.jsonl")):
            if parse_shard_name(path) is None:
                continue
            for row in read_jsonl(path):
                done.add((row.get("uid"), row.get("condition"), row.get("run_index")))
        return done

    def check_resume_manifest(self, manifest: dict) -> None:
        path = self.out_dir / "manifest.json"
        if not path.exists() or self.overwrite:
            return
        old = read_json(path)
        old_hash = old.get("manifest_hash") or manifest_hash(old)
        if old_hash != manifest["manifest_hash"]:
            from .manifest import HASHED_FIELDS
            differing = []
            for k in HASHED_FIELDS:
                if old.get(k) == manifest.get(k):
                    continue
                if k == "config_behavioural":
                    a, b = old.get(k) or {}, manifest.get(k) or {}
                    differing += [f"config.{ck}: {a.get(ck)!r} -> {b.get(ck)!r}"
                                  for ck in sorted(set(a) | set(b))
                                  if a.get(ck) != b.get(ck)]
                else:
                    differing.append(k)
            msg = (f"manifest mismatch on resume ({path}); differing fields: {differing}. "
                   "Resuming into a mismatched manifest is silent corruption.")
            if not self.force_resume:
                raise RunAborted(msg + " Use --force-resume to override.")
            self.warn(msg + " --force-resume given; proceeding.")

    # ------------------------------------------------------------- tasks
    def build_tasks(self, cond_items: dict[tuple[str, str], list[ConditionItem]],
                    repeats: dict[str, int], done: set) -> list[Task]:
        """Grouped by (dataset, condition), stable uid order within a group:
        keeps the shared instruction prefix hot in vLLM's prefix cache and makes
        a partially written shard a contiguous prefix of the intended work."""
        tasks: list[Task] = []
        for (dataset, condition) in sorted(cond_items):
            items = sorted(cond_items[(dataset, condition)], key=lambda c: c.uid)
            n_rep = repeats.get(condition, repeats.get("default", 1))
            for run_index in range(n_rep):
                seed = self.cfg.seed_base + run_index
                for ci in items:
                    if (ci.uid, condition, run_index) in done:
                        continue
                    tasks.append(Task(ci=ci, run_index=run_index, seed=seed))
        return tasks

    # ------------------------------------------------------------- render
    def messages_for(self, ci: ConditionItem) -> list[dict]:
        msgs: list[dict] = []
        if self.cfg.system_prompt:
            msgs.append({"role": "system", "content": self.cfg.system_prompt})
        msgs.append({"role": "user", "content": self.book.render(ci)})
        return msgs

    # ------------------------------------------------------------- rows
    def make_row(self, task: Task, comp: Completion) -> dict:
        ci, cfg = task.ci, self.cfg
        row = blank_result_row()
        content = comp.content
        pr = parse_answer(content, ci.options.keys())

        reasoning = comp.reasoning_content
        truncated_log = False
        if reasoning is not None and len(reasoning) > REASONING_LOG_LIMIT:
            reasoning = reasoning[:REASONING_LOG_LIMIT]
            truncated_log = True

        logprobs = None
        if cfg.capture_logprobs:
            logprobs = extract_answer_logprobs(comp.logprobs, content, pr.parsed_answer)

        row.update({
            "uid": ci.uid, "parent_uid": ci.parent_uid,
            "dataset": ci.dataset, "category": ci.category, "subcategory": ci.subcategory,
            "condition": ci.condition, "condition_group": CONDITION_GROUP[ci.condition],
            "run_index": task.run_index, "seed": task.seed,

            "config_id": cfg.config_id, "family": cfg.family, "model_name": cfg.model_name,
            "reasoning_regime": cfg.reasoning_regime,
            "params_total_b": cfg.params_total_b, "params_active_b": cfg.params_active_b,
            "is_moe": cfg.is_moe, "kazakh_specialized": cfg.kazakh_specialized,
            "prompt_version": self.book.version, "prompt_sha256": self.book.sha256,
            "system_prompt_sha256": cfg.system_prompt_sha256,

            "n_options": ci.n_options, "has_option_e": ci.has_option_e,
            "gold_answer": ci.gold_answer, "gold_answer_original": ci.gold_answer_original,
            "option_permutation": ci.option_permutation,
            "gold_position_index": ci.gold_position_index,
            "rotation_index": ci.rotation_index,
            "noise_type": ci.noise_type, "noise_position": ci.noise_position,
            "noise_char_len": ci.noise_char_len, "instruction_lang": ci.instruction_lang,

            "raw_response": content,
            "reasoning_content": reasoning,
            "reasoning_truncated_in_log": truncated_log,
            "parsed_answer": pr.parsed_answer, "parse_tier": pr.parse_tier,
            "format_valid": pr.format_valid, "parse_error": pr.parse_error,
            "homoglyph_normalized": pr.homoglyph_normalized,
            "is_correct": (pr.parsed_answer == ci.gold_answer) if pr.parsed_answer else False,

            "finish_reason": comp.finish_reason,
            "truncated": comp.finish_reason == "length",
            "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens,
            "reasoning_tokens": comp.reasoning_tokens,
            "answer_logprobs": logprobs,
            "latency_s": comp.latency_s,

            "error": comp.error, "error_kind": comp.error_kind,
            "attempt_count": comp.attempt_count,
            "endpoint_kind": cfg.endpoint_kind,
            "seed_honored": self.client.seed_honored if self.client else None,
            "timestamp_utc": utcnow(),
            "harness_git_sha": self._git_sha,
            "harness_code_sha256": self._code_sha,
            "vllm_version": self._vllm_version,
        })
        if comp.error:
            row["parsed_answer"] = None
            row["is_correct"] = False
        return row

    # ------------------------------------------------------------- guards
    def _update_stats(self, row: dict) -> None:
        self.n_done += 1
        self.stat["n"] += 1
        if row["error"]:
            self.stat["errors"] += 1
            self.stat[f"error_{row['error_kind']}"] += 1
        if row["truncated"]:
            self.stat["truncated"] += 1
        if row["parse_tier"] == "failed":
            self.stat["parse_failed"] += 1
        if row["format_valid"]:
            self.stat["format_valid"] += 1
        if row["is_correct"]:
            self.stat["correct"] += 1
        if row["homoglyph_normalized"]:
            self.stat["homoglyph"] += 1
        if row["completion_tokens"]:
            self.stat["completion_tokens"] += row["completion_tokens"]

        first_n = int(self.guards.get("first_n", 50))
        if self._first_n_seen < first_n:
            self._first_n_seen += 1
            if row["parse_tier"] == "failed" and not row["error"]:
                self._first_n_parse_fail += 1
            if self._first_n_seen == first_n:
                rate = self._first_n_parse_fail / first_n
                if rate > float(self.guards.get("max_parse_failure_rate", 0.5)):
                    raise RunAborted(
                        f"fail-fast: {rate:.0%} parse failures in the first {first_n} "
                        "responses. Diagnose the prompt or the parser before spending a grid.")

        if self.guards.get("abort_on_think_tags_in_content", True) \
                and row["reasoning_content"] in (None, "") \
                and contains_think_tags(row["raw_response"]):
            raise RunAborted(
                "fail-fast: <think> tags in message.content with no reasoning_content. "
                "The server was started without a reasoning parser; the run must be "
                "aborted rather than analysed.")

        n = self.stat["n"]
        if n >= 200:
            tr = self.stat["truncated"] / n
            if tr > float(self.guards.get("max_truncation_rate", 0.25)):
                raise RunAborted(f"fail-fast: truncation rate {tr:.0%} exceeds guard. "
                                 "Raise max_tokens for this configuration.")
            cl = self.stat["error_context_length"] / n
            if cl > float(self.guards.get("max_context_length_error_rate", 0.005)):
                raise RunAborted(
                    f"fail-fast: context_length errors at {cl:.2%}. That is a systematic "
                    "prompt-length problem to fix in 03_build_conditions.py, not noise.")

    def status_line(self) -> str:
        n = max(1, self.stat["n"])
        return (f"n={self.stat['n']} acc={self.stat['correct']/n:.3f} "
                f"fmt={self.stat['format_valid']/n:.3f} "
                f"parsefail={self.stat['parse_failed']/n:.3f} "
                f"trunc={self.stat['truncated']/n:.3f} "
                f"mean_completion_tokens={self.stat['completion_tokens']/n:.0f} "
                f"errors={self.stat['errors']}")

    # ------------------------------------------------------------- execute
    def _appender(self, ci: ConditionItem, run_index: int) -> JsonlAppender:
        key = (ci.dataset, ci.condition, run_index)
        if key not in self._appenders:
            self._appenders[key] = JsonlAppender(
                shard_path(self.results_dir, self.cfg.config_id, *key))
        return self._appenders[key]

    async def run(self, tasks: list[Task], manifest: dict) -> dict:
        from .manifest import code_sha256, git_sha
        self._git_sha = manifest.get("harness_git_sha")
        self._code_sha = manifest.get("harness_code_sha256")
        self._vllm_version = manifest.get("vllm_version")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "logs").mkdir(exist_ok=True)
        path = self.out_dir / "manifest.json"
        if path.exists() and not self.overwrite:
            manifest = merge_manifest(read_json(path), manifest)
        write_json(path, manifest)

        assert self.client is not None
        queue: asyncio.Queue = asyncio.Queue()
        for t in tasks:
            queue.put_nowait(t)

        bar = tqdm(total=len(tasks), desc=self.cfg.config_id, unit="req")
        abort: list[BaseException] = []
        lock = asyncio.Lock()

        async def worker():
            while True:
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    if abort:
                        return
                    comp = await self.client.complete(self.messages_for(task.ci), task.seed)
                    row = self.make_row(task, comp)
                    async with lock:
                        self._appender(task.ci, task.run_index).write(row)
                        try:
                            self._update_stats(row)
                        except RunAborted as exc:
                            abort.append(exc)
                        bar.update(1)
                        if self.stat["n"] % self.progress_every == 0:
                            bar.write(f"[status] {self.status_line()}")
                finally:
                    queue.task_done()

        n_workers = max(1, min(self.cfg.max_concurrency, len(tasks) or 1))
        await asyncio.gather(*[worker() for _ in range(n_workers)])
        bar.close()
        for ap in self._appenders.values():
            ap.close()
        if abort:
            raise abort[0]

        # The plan caps max_tokens at 2,048 with truncation logged per
        # configuration.  Above ~5% the cap must be raised FOR THIS
        # CONFIGURATION and the exception recorded in the manifest -- an
        # uncapped thinking model on a distracting condition is the single
        # largest schedule risk in this project.
        n = max(1, self.stat["n"])
        trunc = self.stat["truncated"] / n
        if trunc > 0.05:
            note = (f"truncation {trunc:.1%} exceeds 5% at "
                    f"max_tokens={self.cfg.sampling.get('max_tokens')}; raise the cap "
                    f"for {self.cfg.config_id} and keep this exception in the manifest")
            self.warn(note)
            m = read_json(path)
            m.setdefault("max_tokens_exceptions", []).append(
                {"utc": utcnow(), "truncation_rate": trunc,
                 "max_tokens": self.cfg.sampling.get("max_tokens"), "note": note})
            write_json(path, m)
        return dict(self.stat)
