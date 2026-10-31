"""Condition builders.  Canonical condition codes live here and nowhere else.

The literal strings are B0 N1 N2 N3 N4 I1 O1 P1 O1C NQR NQI.  No hyphens (they
break `<dataset>__<condition>__run<k>.jsonl` parsing), no aliases, no free-text
condition strings anywhere in the codebase.  `O1C` renders as "O1-C" in paper
output only, via `paper_label()`.
"""
from __future__ import annotations

import random
from enum import Enum
from typing import Iterable, Sequence

from .schema import ConditionItem, Item


class Cond(str, Enum):
    B0 = "B0"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    N4 = "N4"
    I1 = "I1"
    O1 = "O1"
    P1 = "P1"
    O1C = "O1C"
    NQR = "NQR"
    NQI = "NQI"

    def __str__(self) -> str:      # so f"{Cond.B0}" == "B0"
        return self.value


ALL_CONDITIONS: tuple[str, ...] = tuple(c.value for c in Cond)
MAIN_GRID: tuple[str, ...] = ("B0", "N1", "N2", "N3", "N4", "I1", "O1", "P1")
DIAGNOSTICS: tuple[str, ...] = ("O1C", "NQR", "NQI")

VALIDITY_PRESERVING = frozenset({"B0", "N1", "N2", "I1", "O1", "P1"})
DISTRACTOR = frozenset({"N3", "N4"})
DIAGNOSTIC = frozenset({"O1C", "NQR", "NQI"})

CONDITION_GROUP: dict[str, str] = {
    **{c: "validity_preserving" for c in VALIDITY_PRESERVING},
    **{c: "distractor" for c in DISTRACTOR},
    **{c: "diagnostic" for c in DIAGNOSTIC},
}

# code -> (noise_type, noise_position)
_NOISE_SPEC: dict[str, tuple[str | None, str | None]] = {
    "B0": (None, None),
    "N1": ("irrelevant", "before"),
    "N2": ("irrelevant", "after"),
    "N3": ("relevant", "before"),
    "N4": ("relevant", "after"),
    "I1": (None, None),
    "O1": (None, None),
    "P1": (None, None),
    "O1C": (None, None),
    "NQR": ("relevant", None),
    "NQI": ("irrelevant", None),
}

_NOISE_FIELD = {"relevant": "relevant_noise", "irrelevant": "irrelevant_noise"}


def paper_label(code: str) -> str:
    return "O1-C" if code == "O1C" else code


def check_code(code: str) -> str:
    if code not in ALL_CONDITIONS:
        raise ValueError(f"unknown condition {code!r}; valid: {ALL_CONDITIONS}")
    return code


def group_of(code: str) -> str:
    return CONDITION_GROUP[check_code(code)]


def o1c_eligible(item: Item) -> bool:
    """Four-option items only: a five-ring needs five rotations, not four."""
    return not item.has_option_e


# ---------------------------------------------------------------- helpers
def _base_kwargs(item: Item, code: str) -> dict:
    return dict(
        parent_uid=item.uid,
        dataset=item.dataset,
        category=item.category,
        subcategory=item.subcategory,
        condition=code,
        condition_group=group_of(code),
        gold_answer_original=item.answer,
    )


def _identity_options(item: Item) -> dict[str, str]:
    """Byte-identical copy of the source options."""
    return {L: item.options[L] for L in item.letters}


def _gold_slot(letters: Sequence[str], gold: str) -> int:
    return list(letters).index(gold)


def _noise_for(item: Item, code: str) -> tuple[str | None, str | None, str | None]:
    ntype, npos = _NOISE_SPEC[code]
    if ntype is None:
        return None, None, None
    text = getattr(item, _NOISE_FIELD[ntype])
    if text is None:
        raise MissingPerturbation(f"{item.uid}: condition {code} needs {_NOISE_FIELD[ntype]}, which is empty")
    return ntype, npos, text


class MissingPerturbation(ValueError):
    """The source row lacks the generated field this condition needs."""


# ---------------------------------------------------------------- builders
def build(item: Item, code: str, shuffle_seed: int | str = 0) -> list[ConditionItem]:
    """Return every rendered unit for (item, condition).  O1C returns four."""
    code = check_code(code)
    if code == "O1C":
        return _build_o1c(item)
    if code == "O1":
        return [_build_o1(item, shuffle_seed)]
    return [_build_plain(item, code)]


def _build_plain(item: Item, code: str) -> ConditionItem:
    ntype, npos, ntext = _noise_for(item, code)
    options = _identity_options(item)
    question = item.question
    removed = False

    if code == "P1":
        if item.question_paraphrase is None:
            raise MissingPerturbation(f"{item.uid}: P1 needs question_paraphrase, which is empty")
        question = item.question_paraphrase
    elif code in ("NQR", "NQI"):
        question = ""
        removed = True

    # Options are byte-identical to source in every condition except O1/O1C.
    assert options == {L: item.options[L] for L in item.letters}, f"{item.uid}: option text mutated under {code}"

    return ConditionItem(
        uid=item.uid,
        question=question,
        options=options,
        gold_answer=item.answer,
        option_permutation=None,
        gold_position_index=_gold_slot(item.letters, item.answer),
        rotation_index=None,
        noise_type=ntype,
        noise_position=npos,
        noise_text=ntext,
        instruction_lang="kk" if code == "I1" else "en",
        question_removed=removed,
        meta={},
        **_base_kwargs(item, code),
    )


def _build_o1(item: Item, shuffle_seed: int | str) -> ConditionItem:
    """Options shuffled once.  Deterministic per uid, independent of iteration
    order.  Five-option items participate normally: E is an ordinary option."""
    letters = item.letters
    rng = random.Random(f"{item.uid}:{shuffle_seed}")
    source_order = list(letters)
    rng.shuffle(source_order)

    # slot `letters[i]` receives the text originally at `source_order[i]`
    options = {letters[i]: item.options[source_order[i]] for i in range(len(letters))}
    perm = {source_order[i]: letters[i] for i in range(len(letters))}   # original -> new
    assert sorted(perm) == sorted(letters) and sorted(perm.values()) == sorted(letters)

    gold_new = perm[item.answer]
    assert options[gold_new] == item.options[item.answer]

    return ConditionItem(
        uid=item.uid,
        question=item.question,
        options=options,
        gold_answer=gold_new,
        option_permutation=perm,
        gold_position_index=_gold_slot(letters, gold_new),
        rotation_index=None,
        noise_type=None,
        noise_position=None,
        noise_text=None,
        instruction_lang="en",
        meta={"permutation_is_identity": all(k == v for k, v in perm.items())},
        **_base_kwargs(item, "O1"),
    )


def _build_o1c(item: Item) -> list[ConditionItem]:
    """Four cyclic rotations of the option ring.  Relative order preserved,
    only absolute position changes; gold visits A, B, C, D exactly once."""
    if not o1c_eligible(item):
        raise ValueError(f"{item.uid}: O1C excludes five-option items")
    letters = item.letters
    n = len(letters)
    if n != 4:
        raise ValueError(f"{item.uid}: O1C needs exactly 4 options, got {n}")

    out: list[ConditionItem] = []
    for r in range(n):
        options = {letters[i]: item.options[letters[(i + r) % n]] for i in range(n)}
        perm = {letters[(i + r) % n]: letters[i] for i in range(n)}
        gold_new = perm[item.answer]
        out.append(ConditionItem(
            uid=f"{item.uid}#r{r}",
            question=item.question,
            options=options,
            gold_answer=gold_new,
            option_permutation=perm,
            gold_position_index=_gold_slot(letters, gold_new),
            rotation_index=r,
            noise_type=None,
            noise_position=None,
            noise_text=None,
            instruction_lang="en",
            meta={},
            **_base_kwargs(item, "O1C"),
        ))
    slots = sorted(ci.gold_position_index for ci in out)
    assert slots == list(range(n)), f"{item.uid}: O1C gold slots {slots} != each position once"
    return out


def build_many(items: Iterable[Item], code: str, shuffle_seed: int | str = 0,
               skip_missing: bool = False) -> tuple[list[ConditionItem], list[str]]:
    """Returns (condition items, uids skipped for a missing generated field)."""
    out: list[ConditionItem] = []
    skipped: list[str] = []
    for item in items:
        try:
            out.extend(build(item, code, shuffle_seed))
        except MissingPerturbation:
            if not skip_missing:
                raise
            skipped.append(item.uid)
    return out, skipped
