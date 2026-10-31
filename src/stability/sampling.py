"""Optional-parameter request assembly.

The single most error-prone part of the harness.  Written once, tested, never
inlined.  A sampling key the user did not set is ABSENT from the body -- not
None, not a hardcoded default -- so the server applies the model's own
generation_config.
"""
from __future__ import annotations

from typing import Any, Iterable

OPENAI_NATIVE = frozenset({"temperature", "top_p", "max_tokens", "presence_penalty",
                           "frequency_penalty", "stop", "seed", "n"})
VLLM_EXTRA = frozenset({"top_k", "repetition_penalty", "min_p", "min_tokens"})

#: Every key the CLI / YAML layers may set.  Anything else is a typo, not a
#: passthrough -- catch it at config load, not with a 400 mid-grid.
KNOWN_SAMPLING_KEYS = OPENAI_NATIVE | VLLM_EXTRA


class UnknownSamplingKey(ValueError):
    pass


def check_sampling_keys(sampling: dict) -> None:
    bad = sorted(set(sampling) - KNOWN_SAMPLING_KEYS)
    if bad:
        raise UnknownSamplingKey(
            f"unknown sampling keys {bad}; known: {sorted(KNOWN_SAMPLING_KEYS)}")


def build_request(messages: list[dict], cfg, seed: int | None = None,
                  capture_logprobs: bool = False,
                  structured_output: bool = False,
                  drop: Iterable[str] = ()) -> dict[str, Any]:
    """`cfg` needs .model_name, .sampling (dict), .chat_template_kwargs (dict|None).

    `drop` lets the hosted-endpoint probe remove parameters the endpoint
    rejected, without mutating the configuration.
    """
    drop = set(drop)
    body: dict[str, Any] = {"model": cfg.model_name, "messages": messages}
    extra: dict[str, Any] = {}

    for k, v in (cfg.sampling or {}).items():
        if v is None:           # explicitly unset -> do not send
            continue
        if k in drop:
            continue
        (body if k in OPENAI_NATIVE else extra)[k] = v

    if seed is not None and "seed" not in drop:
        body["seed"] = seed

    ctk = getattr(cfg, "chat_template_kwargs", None)
    if ctk and "chat_template_kwargs" not in drop:
        extra["chat_template_kwargs"] = dict(ctk)

    if capture_logprobs and "logprobs" not in drop:
        body["logprobs"] = True
        body["top_logprobs"] = 20

    if structured_output and "response_format" not in drop:
        # OFF by default.  Format compliance is a measured outcome; forcing it
        # makes the metric trivially 100%.  If ever used it must be constant
        # across every condition of a configuration.
        extra["guided_json"] = {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]}},
            "required": ["answer"],
        }

    if extra:
        body["extra_body"] = extra
    return body


def request_template(cfg, seed: int | None, capture_logprobs: bool = False,
                     structured_output: bool = False, drop: Iterable[str] = ()) -> dict:
    """The assembled body minus `messages`, for the run manifest."""
    body = build_request([], cfg, seed=seed, capture_logprobs=capture_logprobs,
                         structured_output=structured_output, drop=drop)
    body.pop("messages", None)
    return body
