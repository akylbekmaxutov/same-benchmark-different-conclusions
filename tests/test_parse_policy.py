"""Under `incorrect`, FAIL -> "B" is a flip and FAIL -> FAIL is not; under
`excluded`, both pairs are dropped and the drop count is reported."""
import pandas as pd

from stability.metrics.core import apply_policy_paired, flip_rates
from stability.parse import FAIL, parse_answer


def mk_paired(pairs):
    rows = []
    for i, (b, p) in enumerate(pairs):
        rows.append({
            "config_id": "c", "dataset": "d", "condition": "N1",
            "parent_uid": f"u{i}", "uid": f"u{i}",
            "baseline_run_index": 0, "perturbed_run_index": 0,
            "semantic_baseline": b, "semantic_perturbed": p,
            "baseline_correct": b == "A", "perturbed_correct": p == "A",
            "pair_has_fail": (b == FAIL) or (p == FAIL),
            "flipped": b != p,
            "transition": ("C" if b == "A" else "I") + ("C" if p == "A" else "I"),
        })
    return pd.DataFrame(rows)


def test_fail_to_letter_is_a_flip_under_incorrect():
    df = mk_paired([(FAIL, "B")])
    f = flip_rates(df, policies=("incorrect",))
    assert f["flip_rate_mean"].iloc[0] == 1.0


def test_fail_to_fail_is_not_a_flip_under_incorrect():
    df = mk_paired([(FAIL, FAIL)])
    f = flip_rates(df, policies=("incorrect",))
    assert f["flip_rate_mean"].iloc[0] == 0.0


def test_excluded_drops_both_pairs_and_reports_the_count():
    df = mk_paired([(FAIL, "B"), (FAIL, FAIL), ("A", "A"), ("A", "B")])
    kept, dropped = apply_policy_paired(df, "excluded")
    assert dropped == 2 and len(kept) == 2
    f = flip_rates(df, policies=("excluded",))
    assert f["flip_rate_mean"].iloc[0] == 0.5
    assert int(f["n_pairs_dropped_by_policy"].iloc[0]) == 2
    assert int(f["n_pairs_before_policy"].iloc[0]) == 4


def test_incorrect_keeps_everything():
    df = mk_paired([(FAIL, "B"), (FAIL, FAIL), ("A", "A"), ("A", "B")])
    kept, dropped = apply_policy_paired(df, "incorrect")
    assert dropped == 0 and len(kept) == 4
    f = flip_rates(df, policies=("incorrect",))
    assert f["flip_rate_mean"].iloc[0] == 0.5


def test_homoglyph_resolves_with_the_flag_set():
    r = parse_answer('{"answer": "В"}', list("ABCD"))   # Cyrillic В
    assert (r.parsed_answer, r.homoglyph_normalized, r.format_valid) == ("B", True, True)


def test_parse_failure_is_a_value_not_a_missing_value():
    r = parse_answer("nonsense", list("ABCD"))
    assert r.parsed_answer is None and r.answer_or_fail == FAIL
