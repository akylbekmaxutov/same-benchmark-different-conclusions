"""N1/N2 use the identical noise string; option text byte-identical outside O1;
O1 gold remapping is a correct inverse; four O1C rotations put gold in each slot
exactly once; paraphrase leaves options untouched."""
import pytest

from conftest import mk_item
from stability.conditions import (ALL_CONDITIONS, CONDITION_GROUP, Cond,
                                  MissingPerturbation, build, o1c_eligible, paper_label)


def one(item, code, seed=7):
    out = build(item, code, seed)
    assert len(out) == 1
    return out[0]


def test_canonical_codes_have_no_hyphens():
    assert all("-" not in c for c in ALL_CONDITIONS)
    assert paper_label("O1C") == "O1-C"
    assert str(Cond.B0) == "B0"


def test_condition_groups():
    assert CONDITION_GROUP["N1"] == "validity_preserving"
    assert CONDITION_GROUP["N3"] == "distractor"
    assert CONDITION_GROUP["O1C"] == "diagnostic"


def test_n1_n2_identical_noise_position_isolated(item):
    a, b = one(item, "N1"), one(item, "N2")
    assert a.noise_text == b.noise_text == item.irrelevant_noise
    assert (a.noise_position, b.noise_position) == ("before", "after")
    assert a.question == b.question == item.question


def test_n3_n4_identical_noise(item):
    a, b = one(item, "N3"), one(item, "N4")
    assert a.noise_text == b.noise_text == item.relevant_noise
    assert a.noise_type == b.noise_type == "relevant"


def test_options_byte_identical_outside_o1(item_e):
    for code in ("B0", "N1", "N2", "N3", "N4", "I1", "P1", "NQR", "NQI"):
        ci = one(item_e, code)
        assert ci.options == item_e.options, code
        assert ci.option_permutation is None
        assert ci.gold_answer == item_e.answer


def test_paraphrase_changes_stem_only(item):
    ci = one(item, "P1")
    assert ci.question == item.question_paraphrase != item.question
    assert ci.options == item.options


def test_i1_switches_instruction_language_only(item):
    ci = one(item, "I1")
    assert ci.instruction_lang == "kk"
    assert ci.question == item.question and ci.options == item.options


def test_o1_gold_remapping_is_a_correct_inverse(item_e):
    ci = one(item_e, "O1")
    perm = ci.option_permutation
    assert sorted(perm) == sorted(item_e.options)
    assert sorted(perm.values()) == sorted(item_e.options)
    for orig, new in perm.items():
        assert ci.options[new] == item_e.options[orig]
    assert ci.gold_answer == perm[item_e.answer]
    assert ci.options[ci.gold_answer] == item_e.options[item_e.answer]


def test_o1_is_deterministic_per_uid_and_seed(item):
    assert one(item, "O1", 11).option_permutation == one(item, "O1", 11).option_permutation
    other = mk_item(uid="kazmmlu::999999")
    # a different uid gets its own draw, independent of iteration order
    perms = {tuple(sorted(one(mk_item(uid=f"kazmmlu::{i:06d}"), "O1", 11)
                          .option_permutation.items()))
             for i in range(20)}
    assert len(perms) > 1


def test_o1_five_option_items_participate(item_e):
    ci = one(item_e, "O1")
    assert "E" in ci.options and len(ci.option_permutation) == 5


def test_o1c_four_rotations_gold_visits_each_slot_once(item):
    rots = build(item, "O1C")
    assert len(rots) == 4
    assert sorted(r.gold_position_index for r in rots) == [0, 1, 2, 3]
    assert sorted(r.rotation_index for r in rots) == [0, 1, 2, 3]
    for r in rots:
        assert r.options[r.gold_answer] == item.options[item.answer]
        assert sorted(r.options.values()) == sorted(item.options.values())
        assert r.uid.startswith(item.uid + "#r") and r.parent_uid == item.uid


def test_o1c_preserves_relative_ring_order(item):
    letters = item.letters
    for r in build(item, "O1C"):
        texts = [r.options[L] for L in letters]
        src = [item.options[L] for L in letters]
        k = src.index(texts[0])
        assert texts == src[k:] + src[:k]


def test_o1c_excludes_five_option_items(item_e):
    assert not o1c_eligible(item_e)
    with pytest.raises(ValueError):
        build(item_e, "O1C")


def test_nq_removes_the_question(item):
    for code, field in (("NQR", "relevant_noise"), ("NQI", "irrelevant_noise")):
        ci = one(item, code)
        assert ci.question_removed and ci.question == ""
        assert ci.noise_text == getattr(item, field)
        assert ci.options == item.options


def test_missing_generated_field_fails_loudly():
    it = mk_item(relevant_noise=None, question_paraphrase=None)
    with pytest.raises(MissingPerturbation):
        build(it, "N3")
    with pytest.raises(MissingPerturbation):
        build(it, "P1")
    build(it, "B0")  # unaffected
