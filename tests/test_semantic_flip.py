"""Flip comparison under permutation is letter-invariant."""
import pandas as pd

from conftest import mk_item
from stability.aggregate import semantic
from stability.conditions import build
from stability.parse import FAIL


def test_semantic_maps_back_to_the_original_letter_space():
    item = mk_item(answer="C")
    ci = build(item, "O1", 42)[0]
    perm = ci.option_permutation                      # original -> new
    for orig, new in perm.items():
        assert semantic(new, perm) == orig


def test_identical_content_choice_is_not_a_flip_under_o1():
    item = mk_item(answer="C")
    ci = build(item, "O1", 42)[0]
    perm = ci.option_permutation
    baseline_letter = "B"                              # model picked option B's text
    perturbed_letter = perm["B"]                       # same TEXT, new slot
    assert semantic(perturbed_letter, perm) == baseline_letter


def test_letter_comparison_would_have_been_wrong():
    """The pitfall this exists to prevent: comparing letters inflates flips."""
    item = mk_item(answer="C")
    ci = build(item, "O1", 3)[0]
    perm = ci.option_permutation
    naive_flips = sum(1 for L in perm if perm[L] != L)
    semantic_flips = sum(1 for L in perm if semantic(perm[L], perm) != L)
    assert semantic_flips == 0
    assert naive_flips > 0


def test_semantic_passthrough_without_permutation():
    assert semantic("B", None) == "B"
    assert semantic("B", {}) == "B"
    assert semantic(FAIL, {"A": "B"}) == FAIL


def test_o1c_rotations_are_letter_invariant():
    item = mk_item(answer="A")
    for ci in build(item, "O1C"):
        perm = ci.option_permutation
        for orig, new in perm.items():
            assert ci.options[new] == item.options[orig]
            assert semantic(new, perm) == orig
