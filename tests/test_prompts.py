"""Prompt assembly: block order, the frozen options format, and the invariant
that the JSON key `answer` is never translated."""
import pytest

from conftest import mk_item
from stability.conditions import build
from stability.prompts import load_prompts


@pytest.fixture
def book():
    return load_prompts()


def render(book, item, code, seed=1):
    return book.render(build(item, code, seed)[0])


def test_block_order_noise_before(book):
    it = mk_item()
    t = render(book, it, "N3")
    i_noise = t.index(it.relevant_noise)
    i_q = t.index(it.question)
    assert i_noise < i_q < t.index("A) ")


def test_block_order_noise_after(book):
    it = mk_item()
    t = render(book, it, "N4")
    assert t.index(it.question) < t.index(it.relevant_noise) < t.index("A) ")


def test_options_block_format_is_frozen(book):
    it = mk_item(n_options=5, answer="E")
    t = render(book, it, "B0")
    for L in "ABCDE":
        assert f"{L}) {it.options[L]}" in t


def test_answer_key_is_never_translated(book):
    it = mk_item()
    for code in ("B0", "I1"):
        assert '"answer"' in render(book, it, code)


def test_i1_switches_instruction_language(book):
    it = mk_item()
    en, kk = render(book, it, "B0"), render(book, it, "I1")
    assert book.instruction("en").split("\n")[0] in en
    assert book.instruction("kk").split("\n")[0] in kk
    assert it.question in en and it.question in kk


def test_nq_conditions_drop_the_question(book):
    it = mk_item()
    t = render(book, it, "NQR")
    assert it.question not in t
    assert it.relevant_noise in t
    assert "A) " in t


def test_prompt_sha_changes_with_the_text(book):
    from stability.prompts import PromptBook
    a = PromptBook(dict(book.spec))
    b = PromptBook({**book.spec, "instruction_en": "different"})
    assert a.sha256 == book.sha256
    assert b.sha256 != book.sha256


def test_render_returns_no_system_prompt(book):
    from stability.prompts import render as render_fn
    sysp, user = render_fn(build(mk_item(), "B0", 1)[0], book)
    assert sysp is None and user
