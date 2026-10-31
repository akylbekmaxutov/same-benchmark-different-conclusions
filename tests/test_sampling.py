"""Omitted keys are absent from the body; vLLM extras land in extra_body;
explicit None never serializes."""
import json
import pytest

from stability.config import ModelConfig
from stability.sampling import UnknownSamplingKey, build_request, request_template


def cfg(**kw):
    base = dict(config_id="c", model_name="m", base_url="http://x/v1")
    base.update(kw)
    return ModelConfig(**base)


def test_omitted_keys_absent():
    body = build_request([{"role": "user", "content": "hi"}], cfg(sampling={}), seed=None)
    assert set(body) == {"model", "messages"}
    for k in ("temperature", "top_p", "top_k", "max_tokens",
              "repetition_penalty", "presence_penalty", "seed"):
        assert k not in body
        assert k not in body.get("extra_body", {})


def test_explicit_none_never_serializes():
    c = cfg(sampling={"temperature": None, "top_p": 0.9})
    assert "temperature" not in c.sampling
    body = build_request([], c, seed=None)
    assert body["top_p"] == 0.9
    assert "temperature" not in body
    assert "null" not in json.dumps(body)


def test_vllm_extras_go_to_extra_body():
    body = build_request([], cfg(sampling={"top_k": 20, "repetition_penalty": 1.05,
                                           "min_p": 0.01, "temperature": 0.7}), seed=7)
    assert body["temperature"] == 0.7
    assert body["seed"] == 7
    assert body["extra_body"] == {"top_k": 20, "repetition_penalty": 1.05, "min_p": 0.01}
    assert "top_k" not in body


def test_chat_template_kwargs_passthrough():
    body = build_request([], cfg(chat_template_kwargs={"enable_thinking": False}), seed=None)
    assert body["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_logprobs_only_when_requested():
    c = cfg(sampling={})
    assert "logprobs" not in build_request([], c, None, capture_logprobs=False)
    b = build_request([], c, None, capture_logprobs=True)
    assert b["logprobs"] is True and b["top_logprobs"] == 20


def test_structured_output_off_by_default():
    assert "guided_json" not in build_request([], cfg(), None).get("extra_body", {})
    b = build_request([], cfg(), None, structured_output=True)
    assert "guided_json" in b["extra_body"]


def test_drop_removes_parameter():
    c = cfg(sampling={"top_k": 20, "temperature": 0.7})
    b = build_request([], c, seed=5, drop={"seed", "top_k"})
    assert "seed" not in b and "extra_body" not in b


def test_unknown_sampling_key_rejected():
    with pytest.raises(UnknownSamplingKey):
        cfg(sampling={"temperture": 0.7})


def test_request_template_has_no_messages():
    t = request_template(cfg(sampling={"temperature": 0.7}), seed=1)
    assert "messages" not in t and t["temperature"] == 0.7
