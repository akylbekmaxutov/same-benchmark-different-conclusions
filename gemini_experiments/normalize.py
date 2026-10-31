"""Phase 1-2: load the three source benchmarks and normalize them into one
shared MCQ schema.

Unified record:
    {"id", "dataset", "category", "question", "A".."E", "answer", "meta": {...}}

`meta` keeps every original field we did not promote, so nothing useful from the
source repositories is discarded.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .config import HF_SOURCES, HF_SPLITS, OPTION_LETTERS

# Subset names that indicate the Kazakh-language portion of KazMMLU.
KAZ_SUBSET_RE = re.compile(
    r"(kazakh|қазақ|\bkaz\b|kaz[_\-]|[_\-]kaz|\bkk\b|[_\-]kk\b|\bkk[_\-])",
    re.IGNORECASE,
)

# Values a source may use for "this option does not exist".
_EMPTY = {"", "none", "null", "nan", "n/a", "-"}


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").strip()
    if text.lower() in _EMPTY:
        return None
    return text


def _first_present(row: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    # case-insensitive fallback
    lowered = {str(k).lower(): k for k in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None and row[key] is not None:
            return row[key]
    return None


def normalize_answer(raw: Any, options: Dict[str, Optional[str]]) -> Optional[str]:
    """Map a source answer value onto an option letter present in `options`."""
    present = [L for L in OPTION_LETTERS if options.get(L)]
    if raw is None:
        return None

    # Integer index (0-based or 1-based).
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        for base in (0, 1):
            idx = raw - base
            if 0 <= idx < len(present):
                return present[idx]
        return None

    text = str(raw).strip()
    if not text:
        return None

    # Plain letter, possibly decorated: "B", "b)", "(C)", "Option D", "Жауап: A"
    m = re.search(r"(?<![A-Za-z])([A-Ea-e])(?![A-Za-z])", text)
    if m and len(text) <= 24:
        letter = m.group(1).upper()
        if letter in present:
            return letter

    # Numeric string.
    if text.isdigit():
        return normalize_answer(int(text), options)

    # Full option text.
    for letter in present:
        if options[letter] and options[letter].strip() == text:
            return letter
    return None


def _build_record(
    dataset: str,
    index: int,
    question: Any,
    options: Dict[str, Optional[str]],
    raw_answer: Any,
    category: Any,
    meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    q = _clean(question)
    opts = {L: _clean(options.get(L)) for L in OPTION_LETTERS}
    answer = normalize_answer(raw_answer, opts)

    if not q or answer is None:
        return None
    if sum(1 for L in OPTION_LETTERS if opts[L]) < 2:
        return None

    record = {
        "id": f"{dataset}-{index:06d}",
        "dataset": dataset,
        "category": _clean(category) or "unknown",
        "question": q,
        **{L: opts[L] for L in OPTION_LETTERS},
        "answer": answer,
        "meta": {k: v for k, v in meta.items() if v is not None},
    }
    return record


# ------------------------------------------------------------------ KazMMLU --
def kazmmlu_kazakh_configs(all_configs: Sequence[str]) -> List[str]:
    return [c for c in all_configs if KAZ_SUBSET_RE.search(c)]


def load_kazmmlu(configs: Optional[Sequence[str]] = None) -> Iterator[Dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    repo = HF_SOURCES["kazmmlu"]
    split = HF_SPLITS["kazmmlu"]

    all_configs = list(get_dataset_config_names(repo))
    chosen = list(configs) if configs else kazmmlu_kazakh_configs(all_configs)
    if not chosen:
        raise SystemExit(
            f"No Kazakh subsets matched in {repo}.\nAvailable configs: {all_configs}\n"
            "Pass them explicitly with --kazmmlu-configs."
        )
    print(f"[kazmmlu] {len(chosen)}/{len(all_configs)} subsets selected as Kazakh:")
    for c in chosen:
        print(f"          - {c}")

    index = 0
    for config in chosen:
        ds = load_dataset(repo, config, split=split)
        for row_i, row in enumerate(ds):
            row = dict(row)

            # Some KazMMLU builds carry an explicit language column; if it is
            # there and says Russian, the row does not belong in this study.
            lang = _clean(_first_present(row, ["language", "Language", "lang"]))
            if lang and not KAZ_SUBSET_RE.search(lang):
                continue

            options = {
                "A": _first_present(row, ["Option A", "option_a", "A", "a"]),
                "B": _first_present(row, ["Option B", "option_b", "B", "b"]),
                "C": _first_present(row, ["Option C", "option_c", "C", "c"]),
                "D": _first_present(row, ["Option D", "option_d", "D", "d"]),
                "E": _first_present(row, ["Option E", "option_e", "E", "e"]),
            }
            promoted = {
                "Question", "Option A", "Option B", "Option C", "Option D",
                "Option E", "Answer Key", "answer_key",
            }
            meta = {k: v for k, v in row.items() if k not in promoted}
            meta.update({
                "source_repo": repo,
                "source_config": config,
                "source_split": split,
                "source_row_index": row_i,
            })

            rec = _build_record(
                dataset="kazmmlu",
                index=index,
                question=_first_present(row, ["Question", "question"]),
                options=options,
                raw_answer=_first_present(row, ["Answer Key", "answer_key", "answer", "Answer"]),
                category=config,          # Phase 1: category <- KazMMLU subset name
                meta=meta,
            )
            if rec is not None:
                index += 1
                yield rec


# ---------------------------------------------------------------- KazCulture --
def load_kazculture() -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    repo = HF_SOURCES["kazculture"]
    split = HF_SPLITS["kazculture"]
    ds = load_dataset(repo, split=split)

    index = 0
    for row_i, row in enumerate(ds):
        row = dict(row)
        options = {
            "A": _first_present(row, ["a", "A", "option_a"]),
            "B": _first_present(row, ["b", "B", "option_b"]),
            "C": _first_present(row, ["c", "C", "option_c"]),
            "D": _first_present(row, ["d", "D", "option_d"]),
            "E": None,                      # KazCulture has no option E
        }
        promoted = {"question", "a", "b", "c", "d", "answer_label"}
        meta = {k: v for k, v in row.items() if k not in promoted}
        meta.update({
            "source_repo": repo,
            "source_split": split,
            "source_row_index": row_i,
            # keep the Kazakh label alongside the canonical English category
            "category_kazakh": _clean(_first_present(row, ["category_kazakh", "category_kk"])),
        })

        category = _first_present(row, ["category_english", "category_en", "category", "domain"])
        rec = _build_record(
            dataset="kazculture",
            index=index,
            question=_first_present(row, ["question"]),
            options=options,
            raw_answer=_first_present(row, ["answer_label", "answer", "label"]),
            category=category,
            meta=meta,
        )
        if rec is not None:
            index += 1
            yield rec


# --------------------------------------------------------------- KazBench-KK --
def load_kazbench_kk() -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    repo = HF_SOURCES["kazbench_kk"]
    split = HF_SPLITS["kazbench_kk"]     # repository exposes only `train`
    ds = load_dataset(repo, split=split)

    index = 0
    for row_i, row in enumerate(ds):
        row = dict(row)
        options = {
            "A": _first_present(row, ["A", "a", "option_a"]),
            "B": _first_present(row, ["B", "b", "option_b"]),
            "C": _first_present(row, ["C", "c", "option_c"]),
            "D": _first_present(row, ["D", "d", "option_d"]),
            "E": None,
        }
        promoted = {"question", "A", "B", "C", "D", "answer"}
        meta = {k: v for k, v in row.items() if k not in promoted}
        meta.update({
            "source_repo": repo,
            "source_split": split,
            "source_split_note": "repository labels this split `train`; used here as the evaluation set",
            "source_row_index": row_i,
            "subcategory": _clean(_first_present(row, ["subcategory", "sub_category"])),
        })

        rec = _build_record(
            dataset="kazbench_kk",
            index=index,
            question=_first_present(row, ["question"]),
            options=options,
            raw_answer=_first_present(row, ["answer"]),
            category=_first_present(row, ["category"]),
            meta=meta,
        )
        if rec is not None:
            index += 1
            yield rec


LOADERS = {
    "kazmmlu": load_kazmmlu,
    "kazculture": load_kazculture,
    "kazbench_kk": load_kazbench_kk,
}


def present_letters(record: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(L for L in OPTION_LETTERS if record.get(L))
