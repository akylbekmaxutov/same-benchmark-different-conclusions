"""Deterministic, offline perturbation checks.  No server needed."""
from __future__ import annotations

import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

from .conditions import ALL_CONDITIONS
from .schema import ConditionItem, Item

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def kk_normalize(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace, NFKC.  Kazakh-aware in
    the sense that it never touches Cyrillic letter identity."""
    t = unicodedata.normalize("NFKC", text or "")
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.casefold().strip()


def id_alignment(normalized: dict[str, list[Item]]) -> dict:
    out = {}
    for ds, items in normalized.items():
        uids = [i.uid for i in items]
        out[ds] = {"n": len(uids), "unique": len(set(uids)) == len(uids)}
    all_uids = [u for items in normalized.values() for u in (i.uid for i in items)]
    out["_global"] = {"n": len(all_uids), "unique_across_files": len(set(all_uids)) == len(all_uids)}
    return out


def missing_fields(items: list[Item]) -> dict:
    fields = ("relevant_noise", "irrelevant_noise", "question_paraphrase")
    return {f: sum(1 for i in items if not (getattr(i, f) or "").strip()) for f in fields}


def leakage(items: list[Item], min_len: int = 8) -> dict:
    """Flag noise/paraphrase containing the gold option text as a normalized
    substring.  Distractor options are tested too, as the base rate: a gold hit
    rate that matches the distractor rate is topical overlap, not leakage."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    flagged: list[dict] = []
    for it in items:
        gold_norm = kk_normalize(it.options[it.answer])
        distractors = [kk_normalize(v) for k, v in it.options.items() if k != it.answer]
        for field in ("relevant_noise", "irrelevant_noise", "question_paraphrase"):
            text = getattr(it, field)
            if not text:
                continue
            hay = kk_normalize(text)
            counts[field]["n"] += 1
            if len(gold_norm) >= min_len and gold_norm in hay:
                counts[field]["gold_hit"] += 1
                flagged.append({"uid": it.uid, "field": field, "gold": it.options[it.answer]})
            d_hits = sum(1 for d in distractors if len(d) >= min_len and d in hay)
            counts[field]["distractor_hits"] += d_hits
            counts[field]["distractor_slots"] += sum(1 for d in distractors if len(d) >= min_len)
    out = {}
    for field, c in counts.items():
        n = max(1, c["n"])
        slots = max(1, c["distractor_slots"])
        out[field] = {
            "n_tested": c["n"],
            "gold_hit": c["gold_hit"],
            "gold_hit_rate": c["gold_hit"] / n,
            "distractor_hit_rate": c["distractor_hits"] / slots,
        }
    return {"per_field": out, "flagged": flagged[:500], "n_flagged": len(flagged)}


def noise_lengths(items: list[Item]) -> dict:
    out = {}
    for field in ("relevant_noise", "irrelevant_noise", "question_paraphrase"):
        lens = [len(getattr(i, field)) for i in items if getattr(i, field)]
        words = [len(getattr(i, field).split()) for i in items if getattr(i, field)]
        if not lens:
            out[field] = {"n": 0}
            continue
        out[field] = {
            "n": len(lens),
            "chars_mean": round(statistics.mean(lens), 1),
            "chars_median": statistics.median(lens),
            "chars_p05": sorted(lens)[int(0.05 * len(lens))],
            "chars_p95": sorted(lens)[int(0.95 * len(lens))],
            "words_mean": round(statistics.mean(words), 1),
        }
    return out


def option_byte_identity(items: dict[str, Item], cond_items: list[ConditionItem]) -> dict:
    """Option text must be byte-identical to source in every condition except
    O1 / O1C, where it must be an exact permutation of the source options."""
    bad_identity: list[str] = []
    bad_perm: list[str] = []
    for ci in cond_items:
        src = items[ci.parent_uid].options
        if ci.condition in ("O1", "O1C"):
            if sorted(ci.options.values()) != sorted(src.values()):
                bad_perm.append(ci.uid)
                continue
            perm = ci.option_permutation or {}
            if any(ci.options[perm[L]] != src[L] for L in src):
                bad_perm.append(ci.uid)
            if ci.gold_answer != perm.get(ci.gold_answer_original):
                bad_perm.append(ci.uid)
        else:
            if ci.options != src:
                bad_identity.append(ci.uid)
    return {"n_checked": len(cond_items),
            "n_option_text_mutated": len(bad_identity),
            "examples_mutated": bad_identity[:10],
            "n_bad_permutation": len(bad_perm),
            "examples_bad_permutation": bad_perm[:10]}


def paraphrase_leaves_options(items: dict[str, Item], cond_items: list[ConditionItem]) -> int:
    return sum(1 for ci in cond_items
               if ci.condition == "P1" and ci.options != items[ci.parent_uid].options)


def embedding_screen(items: list[Item], model_name: str, bottom_pct: float = 0.05) -> dict:
    """Screening statistic ONLY.  Flags the bottom percentile of
    original-vs-paraphrase similarity for inspection; not ground truth."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return {"skipped": "sentence-transformers not installed"}
    pairs = [(i.uid, i.question, i.question_paraphrase) for i in items if i.question_paraphrase]
    if not pairs:
        return {"skipped": "no paraphrases"}
    model = SentenceTransformer(model_name)
    a = model.encode([p[1] for p in pairs], normalize_embeddings=True, show_progress_bar=False)
    b = model.encode([p[2] for p in pairs], normalize_embeddings=True, show_progress_bar=False)
    sims = (a * b).sum(axis=1)
    k = max(1, int(len(sims) * bottom_pct))
    order = sims.argsort()[:k]
    return {
        "model": model_name, "n": len(sims),
        "mean": float(sims.mean()), "p05": float(np.quantile(sims, 0.05)),
        "median": float(np.median(sims)),
        "flagged_bottom_pct": bottom_pct,
        "flagged": [{"uid": pairs[i][0], "similarity": float(sims[i])} for i in order],
        "note": "screening statistic only; flagged items need manual inspection",
    }
