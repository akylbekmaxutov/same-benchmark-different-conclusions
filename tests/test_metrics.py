"""Flip rate, transitions, tau-b and the floor on a small synthetic table with
hand-computed answers."""
import numpy as np
import pandas as pd
import pytest

from conftest import mk_long
from stability.aggregate import build_paired
from stability.metrics import core, floor, margin, ranking

# gold = A throughout
BASE = {"B0": {
    0: {"u0": "A", "u1": "A", "u2": "B", "u3": "A"},
    1: {"u0": "A", "u1": "B", "u2": "B", "u3": "B"},
    2: {"u0": "A", "u1": "A", "u2": "B", "u3": "C"},
}}


@pytest.fixture
def paired_base():
    return build_paired(mk_long(BASE))


def test_self_flip_floor_hand_computed(paired_base):
    """Per-item disagreement across the three B0 runs:
         u0 0/3, u1 2/3, u2 0/3, u3 3/3
       Pairwise rates over the four items: (0,1)=.5  (0,2)=.25  (1,2)=.5
       Mean floor = 0.416666..."""
    f = floor.self_flip_floor(paired_base, policies=("incorrect",))
    assert f["self_flip_floor"].iloc[0] == pytest.approx(0.4166666, abs=1e-6)


def test_baseline_never_pairs_with_itself(paired_base):
    same = (paired_base["condition"] == "B0") & \
           (paired_base["perturbed_run_index"] == paired_base["baseline_run_index"])
    assert not same.any()
    assert len(paired_base) == 4 * 3 * 2      # items x ordered run pairs


def test_margin_strata_hand_computed(paired_base):
    got = paired_base.drop_duplicates("parent_uid").set_index("parent_uid")["margin_stratum"]
    assert got["u0"] == "3/3"
    assert got["u1"] == "2/3"
    assert got["u2"] == "3/3"
    assert got["u3"] == "1/3"


def test_flip_rate_against_each_baseline_run():
    """N1 answers A,A,A,A against baselines
         run0 A,A,B,A -> 1 flip (u2)
         run1 A,B,B,B -> 3 flips
         run2 A,A,B,C -> 2 flips
       mean = (0.25 + 0.75 + 0.50) / 3 = 0.5"""
    spec = dict(BASE)
    spec["N1"] = {0: {"u0": "A", "u1": "A", "u2": "A", "u3": "A"}}
    paired = build_paired(mk_long(spec))
    f = core.flip_rates(paired, policies=("incorrect",))
    n1 = f[f["condition"] == "N1"].iloc[0]
    assert n1["flip_rate_mean"] == pytest.approx(0.5)
    assert n1["flip_rate_min"] == pytest.approx(0.25)
    assert n1["flip_rate_max"] == pytest.approx(0.75)
    assert n1["n_run_pairs"] == 3


def test_transition_matrix_hand_computed():
    """N1 = A,A,A,A (all correct) against baseline run 0 = A,A,B,A:
         u0 C->C, u1 C->C, u2 I->C, u3 C->C"""
    spec = dict(BASE)
    spec["N1"] = {0: {"u0": "A", "u1": "A", "u2": "A", "u3": "A"}}
    paired = build_paired(mk_long(spec))
    p = paired[(paired["condition"] == "N1") & (paired["baseline_run_index"] == 0)]
    t = core.transition_matrix(p, policies=("incorrect",)).iloc[0]
    assert (t["CC"], t["CI"], t["IC"], t["II"]) == (3, 0, 1, 0)
    assert t["gross_churn_rate"] == pytest.approx(0.25)
    assert t["net_delta_rate"] == pytest.approx(0.25)


def test_flip_to_floor_ratio_is_a_ratio_not_a_subtraction():
    spec = dict(BASE)
    spec["N1"] = {0: {"u0": "A", "u1": "A", "u2": "A", "u3": "A"}}
    paired = build_paired(mk_long(spec))
    fl = floor.self_flip_floor(paired, policies=("incorrect",))
    fr = floor.attach_floor(core.flip_rates(paired, policies=("incorrect",)), fl)
    n1 = fr[fr["condition"] == "N1"].iloc[0]
    assert n1["flip_to_floor_ratio"] == pytest.approx(0.5 / 0.4166666, abs=1e-5)
    # the floor column is present next to every flip rate, and never subtracted
    assert "self_flip_floor" in fr.columns
    assert n1["flip_rate_mean"] == pytest.approx(0.5)


def test_semantic_flip_under_o1_is_zero_when_the_model_tracks_content():
    """The model picks the SAME option text before and after the shuffle."""
    perm = {"A": "C", "B": "D", "C": "A", "D": "B"}
    spec = dict(BASE)
    spec["O1"] = {0: {u: perm[a] for u, a in
                      {"u0": "A", "u1": "A", "u2": "B", "u3": "A"}.items()}}
    paired = build_paired(mk_long(spec, perm_by_condition={"O1": perm}))
    o1 = paired[(paired["condition"] == "O1") & (paired["baseline_run_index"] == 0)]
    assert o1["flipped"].sum() == 0


def test_accuracy_and_delta():
    spec = dict(BASE)
    spec["N1"] = {0: {"u0": "A", "u1": "B", "u2": "B", "u3": "B"}}   # 1/4 correct
    long = mk_long(spec)
    acc = core.accuracy(long, policies=("incorrect",))
    n1 = acc[acc["condition"] == "N1"].iloc[0]
    assert n1["accuracy"] == pytest.approx(0.25)
    b0 = acc[acc["condition"] == "B0"]["accuracy"].tolist()
    assert sorted(b0) == pytest.approx([0.25, 0.50, 0.75])   # runs 1, 2, 0
    d = core.delta_accuracy(acc).iloc[0]
    assert d["delta_acc_mean"] == pytest.approx(0.25 - 0.5)


def test_parse_failure_counts_as_a_distinct_value():
    spec = dict(BASE)
    spec["N1"] = {0: {"u0": None, "u1": "A", "u2": "A", "u3": "A"}}
    paired = build_paired(mk_long(spec))
    n1 = paired[(paired["condition"] == "N1") & (paired["baseline_run_index"] == 0)]
    assert n1.set_index("parent_uid").loc["u0", "flipped"]        # FAIL != "A"
    f_inc = core.flip_rates(paired, policies=("incorrect",))
    f_exc = core.flip_rates(paired, policies=("excluded",))
    n_inc = f_inc[f_inc["condition"] == "N1"].iloc[0]
    n_exc = f_exc[f_exc["condition"] == "N1"].iloc[0]
    assert n_exc["n_pairs"] < n_inc["n_pairs"]
    assert n_exc["n_pairs_dropped_by_policy"] == 3


def test_kendall_tau_b_perfect_and_reversed():
    a = pd.Series({"m1": 0.9, "m2": 0.8, "m3": 0.7, "m4": 0.6})
    assert ranking.kendall_tau_b(a, a)["tau_b"] == pytest.approx(1.0)
    assert ranking.kendall_tau_b(a, a[::-1].set_axis(a.index))["tau_b"] == pytest.approx(-1.0)


def test_kendall_tau_b_handles_ties():
    a = pd.Series({"m1": 0.9, "m2": 0.9, "m3": 0.7})
    b = pd.Series({"m1": 0.8, "m2": 0.8, "m3": 0.5})
    r = ranking.kendall_tau_b(a, b)
    assert r["tau_b"] == pytest.approx(1.0)
    assert r["n"] == 3


def test_near_chance_flagging():
    """A model at 25% on 4-option items must be flagged as near chance."""
    spec = {"B0": {0: {f"u{i}": ("A" if i % 4 == 0 else "B") for i in range(200)}}}
    long = mk_long(spec)
    nc = margin.near_chance_cells(long)
    assert bool(nc["near_chance"].iloc[0])
    assert not bool(nc["above_chance"].iloc[0])
