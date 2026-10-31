"""Margin decomposition.

Primary method: run-agreement across the three baseline runs (3/3, 2/3, 1/3).
It must work without logprobs; `answer_logprobs` corroborate on whichever
configurations support them cleanly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import BASELINE, POLICIES, apply_policy_paired
from .stats import binom_vs_chance


def flips_by_stratum(paired: pd.DataFrame,
                     by=("config_id", "dataset", "condition"),
                     policies=POLICIES) -> pd.DataFrame:
    out = []
    for policy in policies:
        df, _ = apply_policy_paired(paired, policy)
        g = df.groupby(list(by) + ["margin_stratum"], dropna=False).agg(
            n_pairs=("flipped", "size"),
            flip_rate=("flipped", "mean"),
            CI_rate=("transition", lambda s: float((s == "CI").mean())),
            IC_rate=("transition", lambda s: float((s == "IC").mean())),
        ).reset_index()
        g["parse_policy"] = policy
        out.append(g)
    return pd.concat(out, ignore_index=True)


def low_margin_share(paired: pd.DataFrame, by=("config_id", "dataset"),
                     policies=POLICIES) -> pd.DataFrame:
    """Share of baseline-CORRECT items sitting in the low-margin stratum --
    the `indistinguishable from coin-flips` number."""
    out = []
    b0 = paired[paired["condition"] == BASELINE]
    for policy in policies:
        df, _ = apply_policy_paired(b0, policy)
        items = df.drop_duplicates(["config_id", "parent_uid"])[
            list(by) + ["parent_uid", "margin_stratum", "baseline_agreement",
                        "baseline_correct"]]
        g = items.groupby(list(by), dropna=False).apply(
            lambda s: pd.Series({
                "n_items": len(s),
                "n_baseline_correct": int(s["baseline_correct"].astype(bool).sum()),
                "low_margin_share_all": float((s["baseline_agreement"] < 1.0).mean()),
                "low_margin_share_of_correct": float(
                    (s.loc[s["baseline_correct"].astype(bool), "baseline_agreement"] < 1.0).mean()
                    if s["baseline_correct"].astype(bool).any() else np.nan),
            }), include_groups=False).reset_index()
        g["parse_policy"] = policy
        out.append(g)
    return pd.concat(out, ignore_index=True)


def near_chance_cells(long: pd.DataFrame, alpha: float = 0.05,
                      by=("config_id", "dataset")) -> pd.DataFrame:
    """Flag model x benchmark cells whose accuracy CI covers 1/n_options.
    Flip rate degenerates at chance: a randomly guessing model produces a high
    flip rate that says nothing about stability."""
    b0 = long[long["condition"] == BASELINE]
    rows = []
    for keys, g in b0.groupby(list(by), dropna=False):
        p_chance = float((1.0 / g["n_options"].astype(float)).mean())
        n = len(g)
        k = int(g["is_correct"].astype(bool).sum())
        res = binom_vs_chance(k, n, p_chance, alpha)
        rec = dict(zip(list(by), keys if isinstance(keys, tuple) else (keys,)))
        rec.update(res)
        rows.append(rec)
    return pd.DataFrame(rows)


def logprob_margin(long: pd.DataFrame) -> pd.DataFrame:
    """Corroborating margin where `answer_logprobs` were captured: the gap
    between the top-1 and top-2 letter logprobs at the answer token."""
    sub = long[long["answer_logprobs"].notna()]
    rows = []
    for _, r in sub.iterrows():
        lp = r["answer_logprobs"]
        tops = (lp or {}).get("top_logprobs") or []
        vals = sorted((t.get("logprob") for t in tops if t.get("logprob") is not None),
                      reverse=True)
        if len(vals) < 2:
            continue
        rows.append({"config_id": r["config_id"], "dataset": r["dataset"],
                     "condition": r["condition"], "parent_uid": r["parent_uid"],
                     "run_index": r["run_index"],
                     "logprob_top1": vals[0], "logprob_margin": vals[0] - vals[1]})
    return pd.DataFrame(rows)
