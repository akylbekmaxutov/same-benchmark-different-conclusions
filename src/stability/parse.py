"""Answer extraction from `message.content` ONLY.

Reasoning arrives in `message.reasoning_content` and is never parsed as the
answer: reasoning text leaking into the parsed answer would corrupt
format-compliance figures differentially by effort level, which is precisely
the comparison the effort gradient exists to support.

Cascade, recording which tier fired:
  1 json_strict    json.loads on the stripped content, `answer` key, valid letter
  2 json_fenced    strip ``` fences, retry
  3 json_embedded  first {...} block via a balanced-brace scan
  4 letter_fallback a lone letter, or an "answer"-adjacent regex
  5 failed         nothing extractable

`format_valid` is tier 1 only -- that is the compliance metric.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

PARSER_VERSION = "parse/v1"

#: Sentinel for "the model produced nothing parseable".  A distinct value, not
#: a missing one: FAIL -> "B" is a flip, FAIL -> FAIL is not.
FAIL = "FAIL"

TIERS = ("json_strict", "json_fenced", "json_embedded", "letter_fallback", "failed")

# Cyrillic letters that are visually identical to Latin option letters.  A
# Kazakh-tuned model prompted in Kazakh emits these regularly; untreated they
# parse as valid JSON, fail the option-set check, and are scored as format
# failures -- penalising exactly the models and the condition (I1) the paper
# is built on.
HOMOGLYPHS = {
    "А": "A",  # А
    "В": "B",  # В
    "С": "C",  # С
    "Е": "E",  # Е
    "а": "A",  # а
    "в": "B",  # в
    "с": "C",  # с
    "е": "E",  # е
}

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.S)
_ANSWER_ADJACENT_RE = re.compile(
    r"""["']?answer["']?\s*(?:[:=]|\bis\b|[—–-])\s*["'“‘]?\s*([A-EА-Яа-яa-e])\b""",
    re.I | re.X,
)
# Kazakh: "жауаб" / "дұрыс жауаб" -- a Kazakh-instructed model sometimes answers in prose.
_ANSWER_ADJACENT_KK_RE = re.compile(
    r"""жауаб\w*\s*[:—–-]?\s*["'“‘]?\s*([A-EА-Яа-яa-e])\b""",
    re.I | re.X,
)
_LONE_LETTER_RE = re.compile(r"^[\s\(\[\*\"'`]*([A-EА-Еа-еa-e])[\s\)\]\.\,\*\"'`:]*$")


@dataclass
class ParseResult:
    parsed_answer: str | None       # letter, or None when nothing extractable
    parse_tier: str
    format_valid: bool
    parse_error: str | None
    homoglyph_normalized: bool

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def answer_or_fail(self) -> str:
        return self.parsed_answer if self.parsed_answer is not None else FAIL


def normalize_letter(raw: str) -> tuple[str, bool]:
    """-> (uppercase Latin letter, whether a homoglyph substitution happened)"""
    ch = str(raw).strip()
    hit = False
    if ch in HOMOGLYPHS:
        ch = HOMOGLYPHS[ch]
        hit = True
    up = ch.upper()
    if up in HOMOGLYPHS:
        up = HOMOGLYPHS[up]
        hit = True
    return up, hit


def _strip_fence(text: str) -> str | None:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else None


def _iter_json_objects(text: str):
    """Yield every top-level {...} block, string- and escape-aware.

    Tier 3 takes the first block that actually carries an `answer` key: a model
    that emits `{"reasoning": {...}} {"answer": "A"}` is producing embedded JSON,
    and mis-tiering it as `letter_fallback` would distort the tier distribution.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    quote = ""
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]


def _answer_from_obj(obj) -> str | None:
    if not isinstance(obj, dict):
        return None
    if "answer" not in obj:
        return None
    val = obj["answer"]
    if isinstance(val, (str, int)):
        return str(val)
    return None


def parse_answer(content: str | None, valid_letters) -> ParseResult:
    """`valid_letters` is the item's *effective* option set (post-permutation).
    An `E` from a four-option item is a parse failure, not a wrong answer."""
    valid = {str(x).upper() for x in valid_letters}

    if content is None:
        return ParseResult(None, "failed", False, "no content", False)
    text = content.strip()
    if not text:
        return ParseResult(None, "failed", False, "empty content", False)

    homo = False
    last_error: str | None = None

    def finish(raw: str, tier: str) -> ParseResult | None:
        nonlocal homo, last_error
        letter, hit = normalize_letter(raw)
        homo = homo or hit
        if letter in valid:
            return ParseResult(letter, tier, tier == "json_strict", None, homo)
        last_error = f"letter {letter!r} not in option set {sorted(valid)}"
        return None

    # tier 1 -- strict JSON
    try:
        raw = _answer_from_obj(json.loads(text))
        if raw is not None:
            res = finish(raw, "json_strict")
            if res:
                return res
        else:
            last_error = last_error or "JSON parsed but no usable `answer` key"
    except (json.JSONDecodeError, ValueError):
        pass

    # tier 2 -- fenced JSON
    inner = _strip_fence(text)
    if inner is not None:
        try:
            raw = _answer_from_obj(json.loads(inner.strip()))
            if raw is not None:
                res = finish(raw, "json_fenced")
                if res:
                    return res
        except (json.JSONDecodeError, ValueError):
            pass

    # tier 3 -- first embedded {...} carrying an `answer` key
    candidates = list(_iter_json_objects(text))
    if inner:
        candidates += list(_iter_json_objects(inner))
    for candidate in candidates:
        try:
            raw = _answer_from_obj(json.loads(candidate))
        except (json.JSONDecodeError, ValueError):
            continue
        if raw is not None:
            res = finish(raw, "json_embedded")
            if res:
                return res

    # tier 4 -- lone letter, or an "answer"-adjacent regex
    m = _LONE_LETTER_RE.match(text)
    if m:
        res = finish(m.group(1), "letter_fallback")
        if res:
            return res
    for rx in (_ANSWER_ADJACENT_RE, _ANSWER_ADJACENT_KK_RE):
        for m in rx.finditer(text):
            res = finish(m.group(1), "letter_fallback")
            if res:
                return res

    return ParseResult(None, "failed", False,
                       last_error or "no answer extractable", homo)


def contains_think_tags(content: str | None) -> bool:
    """`<think>` in `content` means the server was started without a reasoning
    parser.  The run should be aborted, not analysed."""
    if not content:
        return False
    low = content.lower()
    return "<think>" in low or "</think>" in low


# ------------------------------------------------------------------ logprobs
def answer_letter_offset(content: str, letter: str) -> int | None:
    """Character offset of the answer letter inside the `"answer": "X"` span.

    Returns None if it cannot be located unambiguously -- in which case the
    caller writes `answer_logprobs: null` and moves on.  Do not guess.
    """
    if not content:
        return None
    hits = []
    for m in re.finditer(r'["\']?answer["\']?\s*[:=]\s*["\']?', content, re.I):
        j = m.end()
        while j < len(content) and content[j].isspace():
            j += 1
        if j < len(content):
            ch, _ = normalize_letter(content[j])
            if ch == letter.upper():
                hits.append(j)
    if len(hits) == 1:
        return hits[0]
    return None


def extract_answer_logprobs(logprobs_payload, content: str | None,
                            letter: str | None, top_k: int = 20):
    """Only the top-k distribution at the answer token -- never the whole
    response, which multiplies file size by an order of magnitude for data
    that will not be used."""
    if not logprobs_payload or not content or not letter:
        return None
    tokens = (logprobs_payload or {}).get("content") or []
    if not tokens:
        return None
    target = answer_letter_offset(content, letter)
    if target is None:
        return None
    pos = 0
    for tok in tokens:
        text = tok.get("token") or ""
        nxt = pos + len(text)
        if pos <= target < nxt:
            tops = tok.get("top_logprobs") or []
            return {
                "token": text,
                "logprob": tok.get("logprob"),
                "offset": target,
                "top_logprobs": [
                    {"token": t.get("token"), "logprob": t.get("logprob")}
                    for t in tops[:top_k]
                ],
            }
        pos = nxt
    return None
