"""Phase 8: validate the generated perturbation datasets.

    python -m gemini_experiments.validate

Produces:
  reports/validation_report.json      - counts, acceptance rate, error breakdown
  reports/invalid_ids.json            - {dataset: [ids]} to feed back into batch_build
  reports/manual_inspection/*.md      - random + per-category samples for human review

Findings are split into `errors` (item is rejected and must be regenerated) and
`warnings` (item is kept but flagged for manual inspection).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from typing import Any, Dict, List, Set, Tuple

from .config import (DATASETS, GENERATIONS_DIR, OPTION_LETTERS, REPORTS_DIR,
                     normalized_path, release_full_path)
from .jsonl import read_jsonl

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Explicit option-letter markers, e.g. "A)", "(B)", "C.", "нұсқа D"
OPTION_MARKER_RE = re.compile(r"(?<![A-Za-z])[A-E]\s*[\)\.\:]|\([A-E]\)")

LEAKAGE_PHRASES = (
    "дұрыс жауап", "дұрыс нұсқа", "жауабы —", "жауабы -", "жауап:",
    "correct answer", "the answer is", "правильный ответ",
)

# Kazakh negation is mostly MORPHOLOGICAL, not lexical: a paraphrase of
# "... жатпайды?" as "... кірмейді?" or "... сәйкеспейтін ..." is a perfectly
# preserved negation. Matching only a fixed word list rejects thousands of good
# paraphrases, so the negative verb suffixes are matched too.
#   -ма/-ме/-ба/-бе/-па/-пе  +  -йды/-йді/-йтын/-йтін/-ған/-ген/-қан/-кен
# The -й- forms are required for the present/future endings, because bare
# "-ба+ды" also occurs inside ordinary positive verbs such as "жабады".
NEG_LEXICAL_RE = re.compile(
    r"(?<![^\W\d_])(?:емес|жоқ|ешбір|ешқандай|ешкім|ештеңе|еш|not|except)(?![^\W\d_])"
    r"|тыйым\s+сал|қате[^\W\d_]*\s+нұсқа",
    re.IGNORECASE,
)
# Negative verb forms: finite (-майды), participle (-майтын), past (-маған),
# converb (-май), verbal noun (-мау). The "-ды/-ді" and "-с" endings are left
# out deliberately: they also end ordinary positive verbs ("жабады") and common
# nouns ("бас", "бес").
NEG_MORPH_RE = re.compile(
    r"(?:ма|ме|ба|бе|па|пе)(?:й(?:ды|ді|тын|тін)?|ған|ген|қан|кен|у)(?![^\W\d_])",
    re.IGNORECASE,
)


def has_negation(text: str) -> bool:
    if NEG_LEXICAL_RE.search(text):
        return True
    for match in NEG_MORPH_RE.finditer(text):
        start = match.start()
        while start > 0 and ALPHA_RE.match(text[start - 1]):
            start -= 1
        # Skip proper names: "-бай" ends Сарыбай/Жанбай/Абай, not a negation.
        if not text[start].isupper():
            return True
    return False


# --- Kazakh numerals, so a paraphrase may spell a number out in words --------
_UNITS = ["", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз"]
_TENS = ["", "он", "жиырма", "отыз", "қырық", "елу", "алпыс", "жетпіс", "сексен", "тоқсан"]
NUMERAL_WORDS = tuple(w for w in _UNITS + _TENS if w) + ("жүз", "мың", "миллион")


def spell_kk(n: int) -> str:
    """Kazakh reading of an integer, e.g. 1989 -> 'мың тоғыз жүз сексен тоғыз'."""
    if n <= 0 or n >= 10 ** 9:
        return ""
    parts: List[str] = []
    for base, word in ((10 ** 6, "миллион"), (1000, "мың"), (100, "жүз")):
        if n >= base:
            head = n // base
            if head > 1:
                parts.append(spell_kk(head))
            parts.append(word)
            n %= base
    if n >= 10:
        parts.append(_TENS[n // 10])
        n %= 10
    if n > 0:
        parts.append(_UNITS[n])
    return " ".join(p for p in parts if p)

MIN_NOISE_CHARS = 60
MAX_NOISE_CHARS = 1200
MIN_PARAPHRASE_CHARS = 8
NOISE_LENGTH_RATIO = 3.0        # relevant vs irrelevant noise length must be within this factor


def kazakh_ratio(text: str) -> float:
    letters = ALPHA_RE.findall(text)
    if not letters:
        return 0.0
    cyr = sum(1 for ch in letters if CYRILLIC_RE.match(ch))
    return cyr / len(letters)


def contains_option_text(text: str, item: Dict[str, Any], question: str,
                         min_len: int = 12) -> List[str]:
    """Options quoted by the generated text.

    An option whose wording already appears in the ORIGINAL question is exempt:
    reproducing it is unavoidable for a faithful paraphrase and is not leakage.
    """
    hits = []
    lowered = text.lower()
    q_low = question.lower()
    for letter in OPTION_LETTERS:
        opt = item.get(letter)
        if not opt or len(opt) < min_len:
            continue
        opt_low = opt.lower()
        if opt_low in lowered and opt_low not in q_low:
            hits.append(letter)
    return hits


def check_leakage(text: str, item: Dict[str, Any], field: str, question: str) -> List[str]:
    errs = []
    lowered = text.lower()
    for phrase in LEAKAGE_PHRASES:
        if phrase in lowered:
            errs.append(f"{field}: leakage phrase '{phrase}'")
    if OPTION_MARKER_RE.search(text):
        errs.append(f"{field}: contains an option-letter marker")
    hits = contains_option_text(text, item, question)
    if hits:
        errs.append(f"{field}: reproduces option text {hits}")
    return errs


def validate_item(item: Dict[str, Any], source: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    # -- generation present & parseable -------------------------------------
    if item.get("generation_error") and not item.get("generation_ok"):
        errors.append(f"generation error: {item['generation_error']}")
    rel = item.get("relevant_noise")
    irr = item.get("irrelevant_noise")
    par = item.get("question_paraphrase")
    for name, value in (("relevant_noise", rel), ("irrelevant_noise", irr),
                        ("question_paraphrase", par)):
        if not value or not value.strip():
            errors.append(f"{name}: empty")
    if errors:
        return errors, warnings

    # -- options must not have been modified --------------------------------
    for letter in OPTION_LETTERS:
        if (item.get(letter) or None) != (source.get(letter) or None):
            errors.append(f"option {letter} differs from the normalized source")
    if item.get("answer") != source.get("answer"):
        errors.append("gold answer differs from the normalized source")

    question = source["question"]

    # -- language ------------------------------------------------------------
    for name, value in (("relevant_noise", rel), ("irrelevant_noise", irr),
                        ("question_paraphrase", par)):
        ratio = kazakh_ratio(value)
        if ratio < 0.5:
            errors.append(f"{name}: not Cyrillic/Kazakh (ratio={ratio:.2f})")

    # -- answer-label leakage ------------------------------------------------
    for name, value in (("relevant_noise", rel), ("irrelevant_noise", irr),
                        ("question_paraphrase", par)):
        errors.extend(check_leakage(value, item, name, question))

    # -- length control ------------------------------------------------------
    for name, value in (("relevant_noise", rel), ("irrelevant_noise", irr)):
        if len(value) < MIN_NOISE_CHARS:
            errors.append(f"{name}: too short ({len(value)} chars)")
        elif len(value) > MAX_NOISE_CHARS:
            warnings.append(f"{name}: unusually long ({len(value)} chars)")
    ratio = len(rel) / max(1, len(irr))
    if ratio > NOISE_LENGTH_RATIO or ratio < 1 / NOISE_LENGTH_RATIO:
        warnings.append(f"noise length imbalance (relevant/irrelevant = {ratio:.2f})")

    # -- paraphrase quality --------------------------------------------------
    if len(par) < MIN_PARAPHRASE_CHARS:
        errors.append("question_paraphrase: too short")
    if par.strip() == question.strip():
        errors.append("question_paraphrase: identical to the original question")
    if len(par) < 0.4 * len(question) or len(par) > 2.5 * len(question):
        warnings.append(f"question_paraphrase: length {len(par)} vs original {len(question)}")

    p_low = par.lower()

    # A number counts as preserved if it appears as a digit, or is spelled out
    # in Kazakh words ("1989" -> "мың тоғыз жүз сексен тоғыз").
    par_numbers = set(NUMBER_RE.findall(par))
    truly_missing, spelled_out = [], []
    for token in sorted(set(NUMBER_RE.findall(question))):
        if token in par_numbers:
            continue
        words = spell_kk(int(token)) if token.isdigit() else ""
        if words and words in p_low:
            spelled_out.append(token)
        elif any(w in p_low for w in NUMERAL_WORDS):
            spelled_out.append(token)          # numerals present, wording differs
        else:
            truly_missing.append(token)
    if truly_missing:
        errors.append(f"question_paraphrase: numbers dropped {truly_missing}")
    if spelled_out:
        warnings.append(f"question_paraphrase: numbers spelled out {spelled_out}")

    if has_negation(question) and not has_negation(par):
        errors.append("question_paraphrase: negation lost")

    # Named entities: capitalized mid-sentence words and Latin tokens.
    entities = {w for w in WORD_RE.findall(question)
                if len(w) > 3 and (w[0].isupper() or re.match(r"^[A-Za-z]+$", w))}
    kept = {e for e in entities if e.lower() in p_low}
    if entities and len(kept) / len(entities) < 0.6:
        warnings.append(f"question_paraphrase: entities possibly dropped "
                        f"{sorted(entities - kept)[:5]}")

    return errors, warnings


def error_key(message: str) -> str:
    """Collapse an error string into a stable, countable category."""
    field, _, detail = message.partition(":")
    detail = re.sub(r"[\[\(].*", "", detail).strip()
    detail = re.sub(r"\d+(\.\d+)?", "N", detail)[:48].strip()
    return f"{field}: {detail}" if detail else field


def write_inspection(dataset: str, rows: List[Dict[str, Any]], n_random: int, seed: int) -> None:
    rng = random.Random(seed)
    out_dir = REPORTS_DIR / "manual_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_rows = [r for r in rows if r.get("generation_ok")]
    sample = rng.sample(ok_rows, min(n_random, len(ok_rows))) if ok_rows else []

    by_cat: Dict[str, Dict[str, Any]] = {}
    for r in ok_rows:
        by_cat.setdefault(r["category"], r)

    def render(row: Dict[str, Any]) -> str:
        opts = "\n".join(f"- **{L}**: {row[L]}" for L in OPTION_LETTERS if row.get(L))
        return (
            f"### `{row['id']}`  ({row['category']})\n\n"
            f"**Question**\n\n{row['question']}\n\n"
            f"{opts}\n\n**Gold**: {row['answer']}\n\n"
            f"**Paraphrase**\n\n{row['question_paraphrase']}\n\n"
            f"**Relevant (misleading) noise**\n\n{row['relevant_noise']}\n\n"
            f"**Irrelevant noise**\n\n{row['irrelevant_noise']}\n\n---\n"
        )

    body = [f"# Manual inspection - {dataset}\n",
            f"\n## Random sample (n={len(sample)}, seed={seed})\n"]
    body += [render(r) for r in sample]
    body.append(f"\n## One example per category (n={len(by_cat)})\n")
    body += [render(r) for _, r in sorted(by_cat.items())]

    path = out_dir / f"{dataset}.md"
    path.write_text("\n".join(body), encoding="utf-8")
    print(f"[{dataset}] manual-inspection sheet -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the generated perturbation datasets.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--inspect-n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20250101)
    args = ap.parse_args()

    report: Dict[str, Any] = {"datasets": {}}
    invalid: Dict[str, List[str]] = {}

    for dataset in args.datasets:
        rel_path, norm_path = release_full_path(dataset), normalized_path(dataset)
        if not rel_path.exists():
            print(f"[{dataset}] SKIP - {rel_path} missing")
            continue

        rows = read_jsonl(rel_path)
        sources = {r["id"]: r for r in read_jsonl(norm_path)}
        by_id = {r["id"]: r for r in rows}

        missing_ids: Set[str] = set(sources) - set(by_id)
        extra_ids: Set[str] = set(by_id) - set(sources)

        error_counter: collections.Counter = collections.Counter()
        warn_counter: collections.Counter = collections.Counter()
        bad_ids: List[str] = sorted(missing_ids)
        n_warned = 0
        regenerated = 0

        for row in rows:
            source = sources.get(row["id"])
            if source is None:
                continue
            errs, warns = validate_item(row, source)
            if (row.get("generation_round") or 1) > 1:
                regenerated += 1
            for e in errs:
                error_counter[error_key(e)] += 1
            for w in warns:
                warn_counter[w.split(":")[0]] += 1
            if errs:
                bad_ids.append(row["id"])
            if warns:
                n_warned += 1

        n_total = len(sources)
        n_rejected = len(set(bad_ids))
        n_accepted = n_total - n_rejected
        invalid[dataset] = sorted(set(bad_ids))

        report["datasets"][dataset] = {
            "n_source_items": n_total,
            "n_generated": len(rows),
            "n_missing_generation": len(missing_ids),
            "n_unexpected_ids": len(extra_ids),
            "n_regenerated_items": regenerated,
            "n_rejected": n_rejected,
            "n_accepted": n_accepted,
            "n_with_warnings": n_warned,
            "acceptance_rate": round(n_accepted / n_total, 4) if n_total else 0.0,
            "error_breakdown": dict(error_counter.most_common()),
            "warning_breakdown": dict(warn_counter.most_common()),
        }
        print(f"[{dataset}] accepted {n_accepted}/{n_total} "
              f"({report['datasets'][dataset]['acceptance_rate']:.1%}), "
              f"rejected {n_rejected}, warnings {n_warned}")
        for err, count in error_counter.most_common(8):
            print(f"          {count:6d}  {err}")

        write_inspection(dataset, rows, args.inspect_n, args.seed)

    rpt = REPORTS_DIR / "validation_report.json"
    rpt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep a per-round copy as well: the un-suffixed file is overwritten on every
    # run, and the paper needs the history, not just the latest state.
    rounds = [int(p.name[1:]) for p in GENERATIONS_DIR.glob("r*")
              if p.is_dir() and p.name[1:].isdigit()]
    if rounds:
        stamped = REPORTS_DIR / f"validation_report_after_r{max(rounds)}.json"
        stamped.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Archived copy -> {stamped}")
    inv = REPORTS_DIR / "invalid_ids.json"
    inv.write_text(json.dumps(invalid, ensure_ascii=False, indent=2), encoding="utf-8")

    total_bad = sum(len(v) for v in invalid.values())
    print(f"\nValidation report -> {rpt}")
    print(f"Invalid ids ({total_bad}) -> {inv}")
    if total_bad:
        print("Regenerate with:\n"
              "  python -m gemini_experiments.batch_build  --round 2 "
              f"--only-ids {inv}\n"
              "  python -m gemini_experiments.batch_submit --round 2\n"
              "  python -m gemini_experiments.batch_fetch  --round 2 --wait\n"
              "  python -m gemini_experiments.merge_release\n"
              "  python -m gemini_experiments.validate")


if __name__ == "__main__":
    main()
