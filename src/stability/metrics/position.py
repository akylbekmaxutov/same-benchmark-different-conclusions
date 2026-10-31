"""O1-C position bias and letter preference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..parse import FAIL
from .core import POLICIES, apply_policy_long, apply_policy_paired

O1C = "O1C"


def accuracy_by_gold_slot(long: pd.DataFrame, policies=POLICIES) -> pd.DataFrame:
    """Accuracy as a function of the gold slot (A/B/C/D) -- pure position bias."""
    out = []
    for policy in policies:
        df, _ = apply_policy_long(long, policy)
        df = df[df["condition"] == O1C]
        g = df.groupby(["config_id", "dataset", "gold_position_index"], dropna=False).agg(
            n=("is_correct", "size"), accuracy=("is_correct", "mean")).reset_index()
        g["gold_letter"] = g["gold_position_index"].map(
            {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"})
        g["parse_policy"] = policy
        out.append(g)
    return pd.concat(out, ignore_index=True)


def letter_preference(long: pd.DataFrame) -> pd.DataFrame:
    """Distribution of SELECTED letters across all four rotations -- raw letter
    preference, independent of correctness."""
    df = long[long["condition"] == O1C]
    g = df.groupby(["config_id", "dataset", "answer_eff"], dropna=False).size().rename(
        "n").reset_index()
    tot = g.groupby(["config_id", "dataset"])["n"].transform("sum")
    g["share"] = g["n"] / tot
    return g.rename(columns={"answer_eff": "selected_letter"})


def rotation_consistency(paired: pd.DataFrame, policies=POLICIES) -> pd.DataFrame:
    """Semantic consistency across the four rotations: all-four-agree and
    majority-agree rates -- order sensitivity with position held constant."""
    out = []
    for policy in policies:
        df, _ = apply_policy_paired(paired, policy)
        df = df[df["condition"] == O1C]
        if df.empty:
            continue
        one = df.drop_duplicates(["config_id", "parent_uid", "rotation_index"])
        rows = []
        for (cid, ds, puid), g in one.groupby(["config_id", "dataset", "parent_uid"],
                                              sort=False):
            vals = list(g["semantic_perturbed"])
            if len(vals) < 2:
                continue
            vc = pd.Series(vals).value_counts()
            rows.append({"config_id": cid, "dataset": ds, "parent_uid": puid,
                         "n_rotations": len(vals),
                         "all_agree": bool(vc.iloc[0] == len(vals)),
                         "majority_agree": bool(vc.iloc[0] > len(vals) / 2),
                         "n_distinct": int(len(vc))})
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        g = r.groupby(["config_id", "dataset"]).agg(
            n_items=("all_agree", "size"),
            all_four_agree_rate=("all_agree", "mean"),
            majority_agree_rate=("majority_agree", "mean"),
            mean_distinct_answers=("n_distinct", "mean")).reset_index()
        g["parse_policy"] = policy
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def decompose_o1_flips(paired: pd.DataFrame, slot_acc: pd.DataFrame,
                       policy: str = "incorrect") -> pd.DataFrame:
    """Split the O1 flip rate into the share attributable to position bias and
    the share that is genuine instability.

    Position-attributable share = the fraction of O1 flips where the baseline
    answer's ORIGINAL slot and the perturbed answer's NEW slot are the slots
    that the O1-C diagnostic shows the configuration favours -- i.e. the model
    moved toward its preferred position.  The remainder is genuine instability.
    """
    df, _ = apply_policy_paired(paired, policy)
    o1 = df[(df["condition"] == "O1") & (df["flipped"])]
    if o1.empty:
        return pd.DataFrame()
    pref = (slot_acc[slot_acc["parse_policy"] == policy]
            .sort_values("accuracy", ascending=False)
            .drop_duplicates(["config_id", "dataset"])
            .set_index(["config_id", "dataset"])["gold_position_index"].to_dict())
    rows = []
    for (cid, ds), g in o1.groupby(["config_id", "dataset"], sort=False):
        favoured = pref.get((cid, ds))
        if favoured is None:
            rows.append({"config_id": cid, "dataset": ds, "parse_policy": policy,
                         "n_flips": len(g), "position_attributable_share": np.nan,
                         "genuine_instability_share": np.nan,
                         "note": "no O1C diagnostic for this cell"})
            continue
        letters = ["A", "B", "C", "D", "E"]
        moved_to_favoured = sum(
            1 for a in g["perturbed_answer"]
            if a != FAIL and a in letters and letters.index(a) == favoured)
        share = moved_to_favoured / len(g)
        rows.append({"config_id": cid, "dataset": ds, "parse_policy": policy,
                     "n_flips": len(g), "favoured_slot": favoured,
                     "position_attributable_share": share,
                     "genuine_instability_share": 1.0 - share, "note": None})
    return pd.DataFrame(rows)
