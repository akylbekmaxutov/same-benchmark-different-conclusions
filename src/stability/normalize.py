"""raw Gemini perturbation JSONL -> unified `Item` schema.

The corpus in `raw_data_dir` is READ-ONLY.  This module only reads it.

Discovered source shape (see reports/inventory.md): all three benchmarks were
already emitted by the Gemini stage in one common row layout --
  id, dataset, category, question, A, B, C, D, E, answer,
  relevant_noise, irrelevant_noise, question_paraphrase,
  generation_ok, generation_round, generation_error, meta, validation_passed
-- so normalization is a mapping, not a per-benchmark rescue operation.  The
per-benchmark rules below are the ones that are genuinely per-benchmark.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .io import read_jsonl
from .schema import LETTERS, Item

DATASETS = ("kazmmlu", "kazculture", "kazbench_kk")

# Cyrillic option letters have been seen in generated text; gold letters in the
# corpus are Latin, but normalize defensively so a homoglyph never silently
# becomes an "gold not among options" abort.
_HOMOGLYPH_GOLD = {"А": "A", "В": "B", "С": "C", "Е": "E", "а": "A", "в": "B", "с": "C", "е": "E"}


def source_path(raw_data_dir: str | Path, dataset: str, variant: str = "released") -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    stem = f"{dataset}_perturbations" + ("_full" if variant == "full" else "")
    return Path(raw_data_dir) / "perturbations" / f"{stem}.jsonl"


def _orig_id(dataset: str, raw_id: str) -> str:
    """`kazmmlu-000123` -> `000123`; anything else passes through verbatim."""
    prefix = f"{dataset}-"
    return raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id


def _norm_letter(x: str) -> str:
    x = str(x).strip()
    x = _HOMOGLYPH_GOLD.get(x, x)
    return x.upper()


def normalize_row(row: dict, dataset: str) -> Item:
    raw_id = str(row["id"])
    orig_id = _orig_id(dataset, raw_id)
    meta = dict(row.get("meta") or {})

    options: dict[str, str] = {}
    for L in LETTERS:
        text = row.get(L)
        if text is None:
            continue
        text = str(text)
        if not text.strip():
            # An empty string is a data defect, not "no option E".  Fail loudly.
            raise ValueError(f"{dataset}::{orig_id}: option {L} present but empty")
        options[L] = text

    answer = _norm_letter(row["answer"])

    # --- per-benchmark rules -----------------------------------------
    subcategory: str | None = None
    if dataset == "kazbench_kk":
        # Only benchmark with a two-level hierarchy.
        subcategory = meta.get("subcategory")
        # The repository names the split `train`; the paper describes it as the
        # test set.  Follow the paper, keep the discrepancy machine-visible.
        meta.setdefault("split_source", meta.get("source_split", "train"))
        meta["split_role"] = "test_per_paper"
        if "E" in options:
            raise ValueError(f"{dataset}::{orig_id}: unexpected option E (all items are 4-option)")
    elif dataset == "kazculture":
        # Source uses lowercase a/b/c/d + `answer_label`; the Gemini stage already
        # upper-cased both.  Re-assert rather than trust.
        if "E" in options:
            raise ValueError(f"{dataset}::{orig_id}: unexpected option E")
    elif dataset == "kazmmlu":
        # Kazakh subsets only; `category` is the subset name.
        subset = str(row.get("category") or meta.get("Subset") or "")
        if "kaz" not in subset.lower():
            raise ValueError(f"{dataset}::{orig_id}: non-Kazakh subset {subset!r}")

    for k in ("generation_ok", "generation_round", "generation_error",
              "validation_passed", "validation_errors", "validation_warnings"):
        if k in row:
            meta[f"gen_{k}"] = row[k]
    meta["source_id"] = raw_id

    def _clean(x) -> str | None:
        if x is None:
            return None
        x = str(x).strip()
        return x or None

    item = Item(
        uid=f"{dataset}::{orig_id}",
        orig_id=orig_id,
        dataset=dataset,
        category=str(row["category"]),
        subcategory=subcategory,
        question=str(row["question"]),
        options=options,
        answer=answer,
        relevant_noise=_clean(row.get("relevant_noise")),
        irrelevant_noise=_clean(row.get("irrelevant_noise")),
        question_paraphrase=_clean(row.get("question_paraphrase")),
        meta=meta,
    )
    item.validate()
    return item


def load_dataset(raw_data_dir: str | Path, dataset: str,
                 variant: str = "released") -> list[Item]:
    path = source_path(raw_data_dir, dataset, variant)
    if not path.exists():
        raise FileNotFoundError(f"missing perturbation file: {path}")
    items: list[Item] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        item = normalize_row(row, dataset)
        if item.uid in seen:
            raise ValueError(f"duplicate uid {item.uid} in {path}")
        seen.add(item.uid)
        items.append(item)
    return items


def load_all(raw_data_dir: str | Path, datasets=DATASETS,
             variant: str = "released") -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    global_uids: set[str] = set()
    for ds in datasets:
        items = load_dataset(raw_data_dir, ds, variant)
        clash = global_uids & {i.uid for i in items}
        if clash:
            raise ValueError(f"uid collision across datasets: {sorted(clash)[:5]}")
        global_uids |= {i.uid for i in items}
        out[ds] = items
    return out


def iter_normalized(path: str | Path) -> Iterator[Item]:
    for row in read_jsonl(path):
        yield Item.from_dict(row)
