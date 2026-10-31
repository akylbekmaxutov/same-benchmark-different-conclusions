"""Prompt assembly.  Instruction texts are frozen in configs/prompts.yaml.

`render(item, condition) -> (system_prompt | None, user_prompt)`.
The system prompt is NOT part of this module -- it comes from the model config
or the CLI, because it is how reasoning effort is set for Muse-Glimmer and
gpt-oss.  If no system prompt is given, no system message is sent at all.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from .io import read_yaml
from .schema import ConditionItem

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"


class PromptBook:
    """Frozen prompt texts + the hash that proves which text was used."""

    REQUIRED = ("prompt_version", "instruction_en", "instruction_kk",
                "output_contract_en", "output_contract_kk",
                "block_separator", "option_line", "option_separator")

    def __init__(self, spec: dict):
        missing = [k for k in self.REQUIRED if k not in spec]
        if missing:
            raise ValueError(f"prompts.yaml missing keys: {missing}")
        self.spec = spec
        self.version: str = str(spec["prompt_version"])

    # -- frozen texts --------------------------------------------------
    def instruction(self, lang: str) -> str:
        return self.spec["instruction_en" if lang == "en" else "instruction_kk"]

    def contract(self, lang: str) -> str:
        return self.spec["output_contract_en" if lang == "en" else "output_contract_kk"]

    def options_block(self, options: dict[str, str]) -> str:
        line = self.spec["option_line"]
        sep = self.spec["option_separator"]
        return sep.join(line.format(letter=L, text=options[L]) for L in sorted(options))

    # -- assembly ------------------------------------------------------
    def render(self, ci: ConditionItem) -> str:
        """instruction -> noise(before) -> question -> noise(after) -> options -> contract"""
        lang = ci.instruction_lang
        sep = self.spec["block_separator"]
        blocks: list[str] = [self.instruction(lang).strip()]

        if ci.noise_text and ci.noise_position == "before":
            blocks.append(ci.noise_text.strip())
        if not ci.question_removed:
            blocks.append(ci.question.strip())
        if ci.noise_text and ci.noise_position == "after":
            blocks.append(ci.noise_text.strip())
        # NQR / NQI: noise carries no position, the question is gone
        if ci.question_removed and ci.noise_text:
            blocks.append(ci.noise_text.strip())

        blocks.append(self.options_block(ci.options))
        blocks.append(self.contract(lang).strip())
        return sep.join(b for b in blocks if b)

    @property
    def sha256(self) -> str:
        """Hash of every frozen string, so the manifest proves the texts used."""
        h = hashlib.sha256()
        for k in self.REQUIRED:
            h.update(k.encode("utf-8"))
            h.update(b"\x00")
            h.update(str(self.spec[k]).encode("utf-8"))
            h.update(b"\x01")
        return h.hexdigest()


@lru_cache(maxsize=8)
def load_prompts(path: str | Path = DEFAULT_PROMPTS_PATH) -> PromptBook:
    return PromptBook(read_yaml(path))


def render(ci: ConditionItem, book: PromptBook | None = None) -> tuple[str | None, str]:
    book = book or load_prompts()
    return None, book.render(ci)


def sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
