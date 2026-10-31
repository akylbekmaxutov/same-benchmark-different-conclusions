"""Async OpenAI-compatible client for a single vLLM (or hosted) endpoint.

One model per process.  No model loading, no `vllm serve` subprocess, no
model-swapping logic.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai

from .parse import contains_think_tags
from .sampling import build_request

TRANSIENT = ("connection", "timeout", "5xx", "rate_limit")


@dataclass
class Completion:
    content: str | None = None
    reasoning_content: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    logprobs: Any = None
    latency_s: float | None = None
    attempt_count: int = 0
    error: str | None = None
    error_kind: str | None = None
    raw_choice: dict | None = field(default=None, repr=False)


def classify_error(exc: BaseException) -> tuple[str, str]:
    """-> (error_kind, message).  Never lump these together: only the transient
    classes are retried."""
    msg = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, openai.APITimeoutError) or isinstance(exc, asyncio.TimeoutError):
        return "timeout", msg
    if isinstance(exc, openai.APIConnectionError) or isinstance(exc, httpx.ConnectError):
        return "connection", msg
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit", msg
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        body = str(getattr(exc, "message", "") or exc).lower()
        if status == 400 and any(s in body for s in (
                "maximum context length", "max_model_len", "longer than the maximum",
                "context length", "too long", "reduce the length")):
            return "context_length", msg
        if status == 429:
            return "rate_limit", msg
        if status is not None and 500 <= int(status) < 600:
            return "5xx", msg
        return "bad_request", msg
    return "unknown", msg


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def server_max_model_len(models_payload) -> int | None:
    """vLLM reports `max_model_len` on the model object in /v1/models.  Absent
    on some builds and on hosted endpoints, in which case we cannot check."""
    try:
        for m in (models_payload or {}).get("data", []):
            for k in ("max_model_len", "max_context_length", "context_length"):
                if m.get(k):
                    return int(m[k])
    except Exception:
        pass
    return None


def check_context_budget(models_payload, max_tokens: int | None,
                         max_prompt_tokens: int | None) -> tuple[bool, str]:
    """prompt + max_tokens must fit inside the server's --max-model-len.

    A shortfall is a deterministic 400 on EVERY request, so it belongs in a
    pre-flight check, not in a scattered context_length error rate discovered
    six hours in.  -> (ok, message)
    """
    mml = server_max_model_len(models_payload)
    if mml is None:
        return True, ("server did not report max_model_len; context budget "
                      "unchecked -- confirm --max-model-len by hand")
    if not max_tokens:
        return True, f"server max_model_len={mml}; max_tokens omitted, nothing to check"
    prompt = max_prompt_tokens or 0
    need = prompt + max_tokens
    if need <= mml:
        return True, (f"context budget OK: max prompt {prompt} + max_tokens "
                      f"{max_tokens} = {need} <= max_model_len {mml}")
    return False, (
        f"context budget too small: longest prompt {prompt} + max_tokens {max_tokens} "
        f"= {need} tokens, but the server reports max_model_len={mml}.\n"
        f"  Every request would fail with a deterministic 400.\n"
        f"  Either restart the server with --max-model-len {need + 1024} or lower "
        f"max_tokens to at most {mml - prompt}.")


class StabilityClient:
    def __init__(self, cfg, on_warning=None):
        self.cfg = cfg
        self._client = openai.AsyncOpenAI(
            base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY",
            timeout=cfg.timeout_s, max_retries=0,   # retries are ours, and typed
        )
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self.dropped_params: set[str] = set()
        self.seed_honored: bool | None = None if cfg.endpoint_kind == "hosted" else True
        self.probe_report: dict[str, Any] = {}
        self.server_models: Any = None
        self._warn = on_warning or (lambda m: None)

    # ------------------------------------------------------------- startup
    async def health_check(self) -> dict:
        """Hit /v1/models and confirm the served model name matches.  Catching a
        wrong-port collision after a six-hour run is not acceptable."""
        resp = await self._client.models.list()
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        self.server_models = data
        names = [m.get("id") for m in data.get("data", [])]
        if self.cfg.model_name not in names:
            raise RuntimeError(
                f"health check failed: {self.cfg.base_url} serves {names}, "
                f"config expects {self.cfg.model_name!r}. Wrong port?")
        return data

    async def probe_parameters(self, drop_candidates=("seed", "top_k", "repetition_penalty",
                                                      "min_p", "chat_template_kwargs", "logprobs")) -> dict:
        """Hosted endpoints may silently ignore `seed`, reject vLLM extras, or
        refuse logprobs.  Probe once with throwaway requests and record exactly
        what was accepted -- never silently proceed as if `seed` worked."""
        report: dict[str, Any] = {}
        msgs = [{"role": "user", "content": "Reply with the single character: 1"}]

        async def _try(drop: set[str]) -> tuple[bool, str | None]:
            body = build_request(msgs, self.cfg, seed=self.cfg.seed_base,
                                 capture_logprobs=self.cfg.capture_logprobs,
                                 drop=drop | self.dropped_params)
            body["max_tokens"] = min(body.get("max_tokens") or 8, 8)
            try:
                await self._client.chat.completions.create(**body)
                return True, None
            except Exception as exc:                     # noqa: BLE001
                return False, classify_error(exc)[1]

        ok_all, err_all = await _try(set())
        report["full_body_accepted"] = ok_all
        report["full_body_error"] = err_all
        if not ok_all:
            for p in drop_candidates:
                ok, err = await _try({p})
                report[f"accepted_without_{p}"] = ok
                if ok:
                    self.dropped_params.add(p)
                    self._warn(f"endpoint rejected {p!r}; dropping it for this run")
                    break

        # seed determinism: same body twice, compare
        if "seed" not in self.dropped_params:
            body = build_request(
                [{"role": "user", "content": "Write one random Kazakh noun."}],
                self.cfg, seed=self.cfg.seed_base, drop=self.dropped_params)
            body["max_tokens"] = 16
            try:
                a = await self._client.chat.completions.create(**body)
                b = await self._client.chat.completions.create(**body)
                same = (a.choices[0].message.content == b.choices[0].message.content)
                self.seed_honored = bool(same)
                report["seed_repeat_identical"] = bool(same)
                if not same:
                    self._warn("endpoint did not honour `seed`: two identical requests "
                               "differed. The self-flip floor is over an unseeded "
                               "sampler; the manifest records this.")
            except Exception as exc:                     # noqa: BLE001
                report["seed_probe_error"] = classify_error(exc)[1]
                self.seed_honored = None
        else:
            self.seed_honored = False
        self.probe_report = report
        return report

    # ------------------------------------------------------------- request
    async def complete(self, messages: list[dict], seed: int | None) -> Completion:
        cfg = self.cfg
        body = build_request(messages, cfg, seed=seed,
                             capture_logprobs=cfg.capture_logprobs,
                             structured_output=cfg.structured_output,
                             drop=self.dropped_params)
        delay = 1.0
        last_kind = last_msg = None
        async with self._sem:
            for attempt in range(1, cfg.max_attempts + 1):
                t0 = time.monotonic()
                try:
                    resp = await self._client.chat.completions.create(**body)
                    out = self._extract(resp)
                    out.latency_s = round(time.monotonic() - t0, 4)
                    out.attempt_count = attempt
                    return out
                except Exception as exc:                 # noqa: BLE001
                    kind, msg = classify_error(exc)
                    last_kind, last_msg = kind, msg
                    if kind not in TRANSIENT or attempt == cfg.max_attempts:
                        break
                    await asyncio.sleep(delay * (1.0 + random.random() * 0.5))
                    delay = min(delay * 2, 30.0)
        return Completion(error=last_msg, error_kind=last_kind,
                          attempt_count=attempt, latency_s=None)

    def _extract(self, resp) -> Completion:
        choice = resp.choices[0]
        msg = choice.message
        content = _get(msg, "content")
        reasoning = _get(msg, "reasoning_content")
        if reasoning is None:
            reasoning = _get(msg, "reasoning")
        usage = _get(resp, "usage")
        details = _get(usage, "completion_tokens_details")

        if reasoning is None and contains_think_tags(content):
            self._warn("<think> tags found in message.content and no reasoning_content: "
                       "the server was started without a reasoning parser")

        lp = _get(choice, "logprobs")
        try:
            lp = lp.model_dump() if hasattr(lp, "model_dump") else lp
        except Exception:                                 # pragma: no cover
            lp = None

        return Completion(
            content=content,
            reasoning_content=reasoning,
            finish_reason=_get(choice, "finish_reason"),
            prompt_tokens=_get(usage, "prompt_tokens"),
            completion_tokens=_get(usage, "completion_tokens"),
            reasoning_tokens=_get(details, "reasoning_tokens"),
            logprobs=lp,
            raw_choice=None,
        )

    async def aclose(self) -> None:
        await self._client.close()
