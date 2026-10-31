"""Interrupting and resuming produces exactly the same row set as an
uninterrupted run, and a mismatched manifest is refused."""
import asyncio
from pathlib import Path

import pytest

from conftest import mk_item
from stability.client import Completion
from stability.conditions import build
from stability.config import ModelConfig
from stability.io import read_jsonl
from stability.manifest import build_manifest
from stability.prompts import load_prompts
from stability.runner import RunAborted, Runner


class FakeClient:
    """Deterministic stand-in: the answer is a function of (uid, seed)."""
    seed_honored = True

    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.n = 0

    async def complete(self, messages, seed):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise KeyboardInterrupt("simulated crash")
        text = messages[-1]["content"]
        letter = "ABCD"[(hash(text) + (seed or 0)) % 4]
        return Completion(content='{"answer": "%s"}' % letter, finish_reason="stop",
                          prompt_tokens=10, completion_tokens=5, latency_s=0.01,
                          attempt_count=1)


def _cfg():
    return ModelConfig(config_id="tc", model_name="m", base_url="http://x/v1",
                       sampling={"temperature": 0.7}, seed_base=100, max_concurrency=1)


def _setup(tmp_path, cfg, defaults):
    book = load_prompts()
    items = [mk_item(uid=f"kazmmlu::{i:06d}") for i in range(6)]
    cond_items = {("kazmmlu", c): [ci for it in items for ci in build(it, c, 1)]
                  for c in ("B0", "N1")}
    manifest = build_manifest(cfg, defaults, book, conditions=["B0", "N1"],
                              datasets=["kazmmlu"], repeats={"B0": 2, "default": 1},
                              item_counts={"kazmmlu__B0": 6, "kazmmlu__N1": 6},
                              sample_index_sha256="s")
    return book, cond_items, manifest


def _rows(root: Path):
    out = []
    for p in sorted((root / "tc" / "raw").glob("*.jsonl")):
        out += list(read_jsonl(p))
    return out


def _key(r):
    return (r["uid"], r["condition"], r["run_index"])


DEFAULTS = {"guards": {"first_n": 1000}, "shuffle_seed": 1}
REPEATS = {"B0": 2, "default": 1}


def _run(root, client, **kw):
    cfg = _cfg()
    book, cond_items, manifest = _setup(root, cfg, DEFAULTS)
    r = Runner(cfg, DEFAULTS, book, results_dir=root, **kw)
    r.client = client
    r.check_resume_manifest(manifest)
    done = r.completed_keys()
    tasks = r.build_tasks(cond_items, REPEATS, done)
    try:
        asyncio.run(r.run(tasks, manifest))
    except KeyboardInterrupt:
        for ap in r._appenders.values():
            ap.close()
    return r, len(tasks)


def test_uninterrupted_run_writes_every_key(tmp_path):
    _run(tmp_path, FakeClient())
    rows = _rows(tmp_path)
    assert len(rows) == 6 * 2 + 6 * 1                 # B0 x2 runs + N1 x1
    assert len({_key(r) for r in rows}) == len(rows)


def test_resume_matches_an_uninterrupted_run(tmp_path):
    whole = tmp_path / "whole"
    _run(whole, FakeClient())
    expected = sorted(_key(r) for r in _rows(whole))

    part = tmp_path / "part"
    _run(part, FakeClient(fail_after=7))
    partial = _rows(part)
    assert 0 < len(partial) < len(expected)

    runner, n_remaining = _run(part, FakeClient())
    got = _rows(part)
    assert sorted(_key(r) for r in got) == expected
    assert len(got) == len(expected)                   # no duplicates on resume
    assert n_remaining == len(expected) - len(partial)


def test_resume_refuses_a_mismatched_manifest(tmp_path):
    _run(tmp_path, FakeClient())
    cfg = _cfg()
    cfg.sampling = {"temperature": 0.99}               # a different run, silently
    book, cond_items, manifest = _setup(tmp_path, cfg, DEFAULTS)
    r = Runner(cfg, DEFAULTS, book, results_dir=tmp_path)
    with pytest.raises(RunAborted, match="manifest mismatch"):
        r.check_resume_manifest(manifest)


def test_force_resume_overrides_with_a_warning(tmp_path):
    _run(tmp_path, FakeClient())
    cfg = _cfg()
    cfg.sampling = {"temperature": 0.99}
    book, cond_items, manifest = _setup(tmp_path, cfg, DEFAULTS)
    r = Runner(cfg, DEFAULTS, book, results_dir=tmp_path, force_resume=True)
    r.check_resume_manifest(manifest)
    assert any("manifest mismatch" in w for w in r.warnings)


def test_overwrite_ignores_existing_rows(tmp_path):
    _run(tmp_path, FakeClient())
    r, n = _run(tmp_path, FakeClient(), overwrite=True)
    assert n == 6 * 2 + 6 * 1


def test_shard_layout(tmp_path):
    _run(tmp_path, FakeClient())
    names = sorted(p.name for p in (tmp_path / "tc" / "raw").glob("*.jsonl"))
    assert names == ["kazmmlu__B0__run0.jsonl", "kazmmlu__B0__run1.jsonl",
                     "kazmmlu__N1__run0.jsonl"]


def test_think_tags_in_content_abort_the_run(tmp_path):
    class ThinkClient(FakeClient):
        async def complete(self, messages, seed):
            return Completion(content='<think>…</think>{"answer": "A"}',
                              finish_reason="stop", attempt_count=1, latency_s=0.01)
    with pytest.raises(RunAborted, match="reasoning parser"):
        _run(tmp_path, ThinkClient())


# ---------------------------------------------------------- resume guard scope
def _manifest(tmp_path, **cfg_kw):
    cfg = _cfg()
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    book, cond_items, manifest = _setup(tmp_path, cfg, DEFAULTS)
    return cfg, book, manifest


def _resume_ok(tmp_path, **cfg_kw):
    """-> True if a resume with these config changes is allowed."""
    cfg, book, manifest = _manifest(tmp_path, **cfg_kw)
    r = Runner(cfg, DEFAULTS, book, results_dir=tmp_path)
    try:
        r.check_resume_manifest(manifest)
        return True
    except RunAborted:
        return False


@pytest.mark.parametrize("field,value", [
    ("max_concurrency", 8),      # throughput knob
    ("timeout_s", 900),          # patience
    ("max_attempts", 3),         # retry budget
    ("api_key", "other"),        # credential
    ("base_url", "http://elsewhere:9999/v1"),   # model redeployed on a new port
])
def test_operational_changes_do_not_block_resume(tmp_path, field, value):
    """These affect throughput or routing, never the content of a request or a
    result row.  Blocking on them would make the guard cry wolf."""
    _run(tmp_path, FakeClient())
    assert _resume_ok(tmp_path, **{field: value}), f"{field} should not block resume"


@pytest.mark.parametrize("field,value", [
    ("sampling", {"temperature": 0.99}),
    ("model_name", "a-different-model"),
    ("system_prompt", "Reasoning: high"),
    ("chat_template_kwargs", {"enable_thinking": True}),
    ("seed_base", 999),
    ("structured_output", True),
    ("reasoning_regime", "high"),      # denormalized onto every row
    ("kazakh_specialized", True),      # ditto
])
def test_behavioural_changes_block_resume(tmp_path, field, value):
    _run(tmp_path, FakeClient())
    assert not _resume_ok(tmp_path, **{field: value}), f"{field} must block resume"


def test_the_refusal_names_the_offending_config_key(tmp_path):
    _run(tmp_path, FakeClient())
    cfg, book, manifest = _manifest(tmp_path, sampling={"temperature": 0.99})
    r = Runner(cfg, DEFAULTS, book, results_dir=tmp_path)
    with pytest.raises(RunAborted, match=r"config\.sampling"):
        r.check_resume_manifest(manifest)


# ------------------------------------------------------- context-budget guard
@pytest.mark.parametrize("mml,max_tokens,prompt,ok", [
    (16384, 32768,  800, False),   # qolda card default -- every request would 400
    (40960, 32768,  800, True),
    (32768, 32768,  800, False),   # exactly max_model_len leaves no room for a prompt
    (32768,  2048,  800, True),
    (None,  32768,  800, True),    # server did not report it; cannot check
    (16384,  None,  800, True),    # max_tokens omitted; server decides
])
def test_context_budget_guard(mml, max_tokens, prompt, ok):
    from stability.client import check_context_budget
    payload = {"data": [{"id": "m", **({"max_model_len": mml} if mml else {})}]}
    got, msg = check_context_budget(payload, max_tokens, prompt)
    assert got is ok, msg
    if not ok:
        assert "max-model-len" in msg and "lower max_tokens" in msg
