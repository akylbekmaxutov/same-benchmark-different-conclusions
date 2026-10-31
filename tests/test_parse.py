"""~30 messy real-world outputs with expected tier and answer."""
import pytest

from stability.parse import FAIL, parse_answer, contains_think_tags, extract_answer_logprobs

ABCD = list("ABCD")
ABCDE = list("ABCDE")

CASES = [
    # (content, valid_letters, expected_answer, expected_tier, format_valid)
    ('{"answer": "B"}', ABCD, "B", "json_strict", True),
    ('{"answer":"A"}', ABCD, "A", "json_strict", True),
    ('  {"answer" : "d"}  ', ABCD, "D", "json_strict", True),
    ('{"answer": "E"}', ABCDE, "E", "json_strict", True),
    ('{"answer": "E"}', ABCD, None, "failed", False),          # E from a 4-option item
    ('{ "reasoning": "…", "answer": "A" }', ABCD, "A", "json_strict", True),
    ('```json\n{"answer": "C"}\n```', ABCD, "C", "json_fenced", False),
    ('```\n{"answer": "C"}\n```', ABCD, "C", "json_fenced", False),
    ('```JSON\n{"answer":"B"}\n```', ABCD, "B", "json_fenced", False),
    ('Жауабы мынау: {"answer": "D"}', ABCD, "D", "json_embedded", False),
    ('Sure! {"answer": "D", "why": "x"} Hope that helps.', ABCD, "D", "json_embedded", False),
    ('text {"note": "{nested}"} then {"answer": "A"}', ABCD, "A", "json_embedded", False),
    ('B', ABCD, "B", "letter_fallback", False),
    (' **C** ', ABCD, "C", "letter_fallback", False),
    ('(A)', ABCD, "A", "letter_fallback", False),
    ('D.', ABCD, "D", "letter_fallback", False),
    ('answer: c', ABCD, "C", "letter_fallback", False),
    ('The answer is A.', ABCD, "A", "letter_fallback", False),
    ('Жауабы: С', ABCD, "C", "letter_fallback", False),        # Cyrillic С
    ('', ABCD, None, "failed", False),
    ('   ', ABCD, None, "failed", False),
    ('{"answer":', ABCD, None, "failed", False),
    ('I cannot determine the answer.', ABCD, None, "failed", False),
    ('{"choice": "B"}', ABCD, None, "failed", False),
    ('{"answer": "Z"}', ABCD, None, "failed", False),
    ('{"answer": 2}', ABCD, None, "failed", False),
    ('Бұл сұраққа жауап беру мүмкін емес.', ABCD, None, "failed", False),
]

HOMOGLYPH_CASES = [
    ('{"answer": "А"}', "A"),   # Cyrillic А
    ('{"answer": "В"}', "B"),   # Cyrillic В
    ('{"answer": "С"}', "C"),   # Cyrillic С
    ('{"answer": "Е"}', "E"),   # Cyrillic Е
]


@pytest.mark.parametrize("content,valid,ans,tier,fmt", CASES)
def test_cascade(content, valid, ans, tier, fmt):
    r = parse_answer(content, valid)
    assert r.parsed_answer == ans, content
    assert r.parse_tier == tier, content
    assert r.format_valid is fmt, content


@pytest.mark.parametrize("content,expected", HOMOGLYPH_CASES)
def test_homoglyphs_resolve_and_are_flagged(content, expected):
    r = parse_answer(content, ABCDE)
    assert r.parsed_answer == expected
    assert r.homoglyph_normalized is True
    assert r.format_valid is True          # compliance is not penalised


def test_latin_letters_are_not_flagged():
    r = parse_answer('{"answer": "B"}', ABCD)
    assert r.homoglyph_normalized is False


def test_format_valid_is_tier_one_only():
    assert parse_answer('{"answer":"A"}', ABCD).format_valid
    assert not parse_answer('```json\n{"answer":"A"}\n```', ABCD).format_valid
    assert not parse_answer('A', ABCD).format_valid


def test_answer_or_fail_sentinel():
    assert parse_answer("", ABCD).answer_or_fail == FAIL
    assert parse_answer('{"answer":"A"}', ABCD).answer_or_fail == "A"


def test_think_tag_detection():
    assert contains_think_tags("<think>hmm</think>{\"answer\":\"A\"}")
    assert not contains_think_tags('{"answer":"A"}')
    assert not contains_think_tags(None)


def test_logprob_extraction_by_offset():
    content = '{"answer": "B"}'
    payload = {"content": [
        {"token": '{"answer": "', "logprob": -0.1, "top_logprobs": []},
        {"token": "B", "logprob": -0.2,
         "top_logprobs": [{"token": "B", "logprob": -0.2}, {"token": "A", "logprob": -1.9}]},
        {"token": '"}', "logprob": -0.05, "top_logprobs": []},
    ]}
    got = extract_answer_logprobs(payload, content, "B")
    assert got["token"] == "B"
    assert got["top_logprobs"][0]["token"] == "B"


def test_logprob_extraction_gives_up_when_ambiguous():
    content = '{"answer": "B"} ... {"answer": "B"}'
    assert extract_answer_logprobs({"content": []}, content, "B") is None
