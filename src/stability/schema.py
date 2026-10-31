"""Unified data schema: Item, ConditionItem, ResultRow.

`Item` is the normalized source record.  `ConditionItem` is one *rendered*
evaluation unit (item x condition x rotation).  `ResultRow` is the item-level
output contract for the whole analysis layer (GUIDELINE.md §8) -- if a metric
is not recoverable from a ResultRow, this schema is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

LETTERS = ("A", "B", "C", "D", "E")


@dataclass
class Item:
    uid: str                       # f"{dataset}::{orig_id}" -- unique across everything
    orig_id: str
    dataset: str                   # kazmmlu | kazculture | kazbench_kk
    category: str
    subcategory: str | None
    question: str
    options: dict[str, str]        # {"A":..., "B":..., "C":..., "D":..., ["E":...]}
    answer: str                    # gold letter, uppercase
    relevant_noise: str | None
    irrelevant_noise: str | None
    question_paraphrase: str | None
    meta: dict = field(default_factory=dict)

    # -- derived -------------------------------------------------------
    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def has_option_e(self) -> bool:
        return "E" in self.options

    @property
    def letters(self) -> list[str]:
        return [L for L in LETTERS if L in self.options]

    def validate(self) -> None:
        if not self.uid or "::" not in self.uid:
            raise ValueError(f"bad uid: {self.uid!r}")
        if not self.question or not self.question.strip():
            raise ValueError(f"{self.uid}: empty question")
        if len(self.options) < 2:
            raise ValueError(f"{self.uid}: fewer than two options")
        for L, text in self.options.items():
            if L not in LETTERS:
                raise ValueError(f"{self.uid}: option letter {L!r} outside A-E")
            if text is None or not str(text).strip():
                raise ValueError(f"{self.uid}: empty option text for {L}")
        if self.answer not in self.options:
            raise ValueError(
                f"{self.uid}: gold letter {self.answer!r} not among options "
                f"{sorted(self.options)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class ConditionItem:
    """One rendered evaluation unit.  Carries the effective option mapping."""

    uid: str                        # unique row key incl. condition/rotation
    parent_uid: str                 # -> Item.uid ; load-bearing for all pairing
    dataset: str
    category: str
    subcategory: str | None
    condition: str                  # canonical code, see conditions.CondCode
    condition_group: str            # validity_preserving | distractor | diagnostic

    question: str                   # rendered stem ("" for NQR/NQI)
    options: dict[str, str]         # effective, post-permutation
    gold_answer: str                # gold AFTER remapping
    gold_answer_original: str
    option_permutation: dict[str, str] | None   # original letter -> new letter
    gold_position_index: int        # 0-based slot of gold in the rendered list
    rotation_index: int | None      # 0-3 for O1C

    noise_type: str | None          # relevant | irrelevant | None
    noise_position: str | None      # before | after | None
    noise_text: str | None
    instruction_lang: str           # en | kk
    question_removed: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def has_option_e(self) -> bool:
        return "E" in self.options

    @property
    def noise_char_len(self) -> int | None:
        return None if self.noise_text is None else len(self.noise_text)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionItem":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


# Field order of the item-level result row.  Kept explicit so every shard has
# the same columns and 06_aggregate can type them without inference surprises.
RESULT_ROW_FIELDS: tuple[str, ...] = (
    # identity
    "uid", "parent_uid", "dataset", "category", "subcategory",
    "condition", "condition_group", "run_index", "seed",
    # configuration
    "config_id", "family", "model_name", "reasoning_regime",
    "params_total_b", "params_active_b", "is_moe", "kazakh_specialized",
    "prompt_version", "prompt_sha256", "system_prompt_sha256",
    # item state as evaluated
    "n_options", "has_option_e", "gold_answer", "gold_answer_original",
    "option_permutation", "gold_position_index", "rotation_index",
    "noise_type", "noise_position", "noise_char_len", "instruction_lang",
    # model output
    "raw_response", "reasoning_content", "reasoning_truncated_in_log",
    "parsed_answer", "parse_tier", "format_valid", "parse_error",
    "homoglyph_normalized", "is_correct",
    # generation accounting
    "finish_reason", "truncated", "prompt_tokens", "completion_tokens",
    "reasoning_tokens", "answer_logprobs", "latency_s",
    # provenance
    "error", "error_kind", "attempt_count", "endpoint_kind", "seed_honored",
    "timestamp_utc", "harness_git_sha", "harness_code_sha256", "vllm_version",
)


def blank_result_row() -> dict:
    return {k: None for k in RESULT_ROW_FIELDS}
