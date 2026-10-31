"""Model configuration loading.  Precedence: CLI > model YAML > defaults.yaml > omitted.

"Omitted" is a real state and survives all three layers -- an unset sampling key
never becomes a default anywhere in this file.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .io import read_yaml
from .sampling import check_sampling_keys

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPO_ROOT / "configs" / "defaults.yaml"


@dataclass
class ModelConfig:
    config_id: str
    model_name: str
    base_url: str
    api_key: str = "EMPTY"
    family: str = "unknown"

    # analysis metadata -- recorded, not used at inference
    params_total_b: float | None = None
    params_active_b: float | None = None
    is_moe: bool | None = None
    kazakh_specialized: bool | None = None
    reasoning_regime: str | None = None

    system_prompt: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    sampling: dict[str, Any] = field(default_factory=dict)

    seed_base: int = 1234
    max_concurrency: int = 32
    timeout_s: int = 600
    max_attempts: int = 5
    capture_logprobs: bool = False
    structured_output: bool = False
    endpoint_kind: str = "vllm"

    notes: str | None = None

    def __post_init__(self):
        # An explicit `null` in YAML means "do not send this key".  Drop it here
        # so it can never be serialized as None.
        self.sampling = {k: v for k, v in (self.sampling or {}).items() if v is not None}
        check_sampling_keys(self.sampling)
        if self.chat_template_kwargs is not None and not self.chat_template_kwargs:
            self.chat_template_kwargs = None
        if self.endpoint_kind not in ("vllm", "hosted"):
            raise ValueError(f"endpoint_kind must be vllm|hosted, got {self.endpoint_kind!r}")

    @property
    def system_prompt_sha256(self) -> str | None:
        if self.system_prompt is None:
            return None
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


_SHARED_KEYS = ("seed_base", "max_concurrency", "timeout_s", "max_attempts",
                "capture_logprobs", "structured_output", "endpoint_kind")


def load_defaults(path: str | Path = DEFAULTS_PATH) -> dict:
    d = read_yaml(path)
    root = Path(path).resolve().parents[1]
    for k in ("raw_data_dir", "work_dir", "results_dir"):
        if k in d and not Path(d[k]).is_absolute():
            d[k] = str((root / d[k]).resolve())
    return d


def load_model_config(path: str | Path, defaults: dict | None = None,
                      overrides: dict | None = None) -> ModelConfig:
    defaults = defaults if defaults is not None else load_defaults()
    spec = dict(read_yaml(path))

    merged: dict[str, Any] = {}
    for k in _SHARED_KEYS:
        if k in defaults:
            merged[k] = defaults[k]
    merged.update({k: v for k, v in spec.items() if k != "sampling"})
    merged["sampling"] = dict(spec.get("sampling") or {})

    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k.startswith("sampling."):
            merged["sampling"][k.split(".", 1)[1]] = v
        elif k == "chat_template_kwargs":
            merged.setdefault("chat_template_kwargs", {})
            merged["chat_template_kwargs"] = {**(merged.get("chat_template_kwargs") or {}), **v}
        else:
            merged[k] = v

    known = set(ModelConfig.__dataclass_fields__)
    unknown = sorted(set(merged) - known)
    if unknown:
        raise ValueError(f"{path}: unknown config keys {unknown}")
    return ModelConfig(**merged)
