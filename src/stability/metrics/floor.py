"""Self-flip floor.

Same computation as every other flip rate, on the same table, differing only in
which rows it selects.  NEVER subtracted from a perturbation flip rate --
self-flips and perturbation flips partially overlap, so subtraction understates.
Presented next to every flip rate, and as a ratio.
"""
from __future__ import annotations

import pandas as pd

from .core import BASELINE, POLICIES, flip_rates


def self_flip_floor(paired: pd.DataFrame, by=("config_id", "dataset"),
                    policies=POLICIES) -> pd.DataFrame:
    b0 = paired[paired["condition"] == BASELINE]
    f = flip_rates(b0, by=tuple(by) + ("condition",), policies=policies)
    f = f.drop(columns=["condition"])
    return f.rename(columns={
        "flip_rate_mean": "self_flip_floor",
        "flip_rate_sd": "self_flip_floor_sd",
        "flip_rate_min": "self_flip_floor_min",
        "flip_rate_max": "self_flip_floor_max"})


def attach_floor(flips: pd.DataFrame, floor: pd.DataFrame,
                 on=("config_id", "dataset", "parse_policy")) -> pd.DataFrame:
    keep = list(on) + ["self_flip_floor", "self_flip_floor_sd"]
    out = flips.merge(floor[keep], on=list(on), how="left")
    out["flip_to_floor_ratio"] = out["flip_rate_mean"] / out["self_flip_floor"].replace(0, float("nan"))
    return out


def floor_transfer(paired: pd.DataFrame, condition: str = "P1",
                   by=("config_id", "dataset"), policies=POLICIES) -> pd.DataFrame:
    """Floor-transfer check: the internal floor of a condition run 3x, compared
    with the B0 floor.  A higher floor under perturbation is a finding.

    Requires the condition to have been run with --repeat <cond>=3; internal
    pairs are formed across its own run indices via the same pairing logic.
    """
    sub = paired[(paired["condition"] == condition)]
    if sub.empty or sub["perturbed_run_index"].nunique() < 2:
        return pd.DataFrame(columns=list(by) + ["condition", "internal_floor",
                                                "self_flip_floor", "parse_policy"])
    # internal disagreement across the condition's own runs, per item
    rows = []
    for policy in policies:
        from .core import apply_policy_paired
        df, _ = apply_policy_paired(sub, policy)
        # one representative pairing per (item, run) against baseline run 0 is
        # not what we want here: build run-vs-run disagreement directly
        piv = df.drop_duplicates(["config_id", "parent_uid", "perturbed_run_index"])[
            list(by) + ["parent_uid", "perturbed_run_index", "semantic_perturbed"]]
        for keys, g in piv.groupby(list(by), dropna=False):
            wide = g.pivot_table(index="parent_uid", columns="perturbed_run_index",
                                 values="semantic_perturbed", aggfunc="first")
            runs = list(wide.columns)
            pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
            if not pairs:
                continue
            dis = sum((wide[a] != wide[b]).mean() for a, b in pairs) / len(pairs)
            rec = dict(zip(list(by), keys if isinstance(keys, tuple) else (keys,)))
            rec.update({"condition": condition, "internal_floor": float(dis),
                        "n_items": int(len(wide)), "n_run_pairs": len(pairs),
                        "parse_policy": policy})
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base = self_flip_floor(paired, by=by, policies=policies)[
        list(by) + ["parse_policy", "self_flip_floor"]]
    out = out.merge(base, on=list(by) + ["parse_policy"], how="left")
    out["floor_transfer_ratio"] = out["internal_floor"] / out["self_flip_floor"].replace(0, float("nan"))
    return out
