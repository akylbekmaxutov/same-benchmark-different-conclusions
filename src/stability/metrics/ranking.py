"""Ranking stability.

Kendall tau-b (not Spearman): n = 20 configurations with ties.  Baseline-internal
stability comes FIRST: if configurations swap places between identical reruns,
no perturbation-induced reversal involving them means anything.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as sps

from .core import BASELINE, POLICIES, apply_policy_long
from .stats import paired_bootstrap_diff


def _scores(long: pd.DataFrame, condition: str, run_index=None) -> pd.Series:
    sub = long[long["condition"] == condition]
    if run_index is not None:
        sub = sub[sub["run_index"] == run_index]
    return sub.groupby("config_id")["is_correct"].mean()


def kendall_tau_b(a: pd.Series, b: pd.Series) -> dict:
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return {"tau_b": float("nan"), "p_value": float("nan"), "n": len(common)}
    r = sps.kendalltau(a[common].values, b[common].values, variant="b")
    return {"tau_b": float(r.statistic), "p_value": float(r.pvalue), "n": len(common)}


def baseline_internal_stability(long: pd.DataFrame, dataset: str | None = None,
                                policies=POLICIES) -> pd.DataFrame:
    """tau-b between the three B0 runs.  Run this before reading any
    perturbation-induced reversal."""
    rows = []
    for policy in policies:
        df, _ = apply_policy_long(long, policy)
        if dataset:
            df = df[df["dataset"] == dataset]
        runs = sorted(df.loc[df["condition"] == BASELINE, "run_index"].unique())
        for i, j in itertools.combinations(runs, 2):
            r = kendall_tau_b(_scores(df, BASELINE, i), _scores(df, BASELINE, j))
            rows.append({"dataset": dataset or "pooled", "run_a": i, "run_b": j,
                         "parse_policy": policy, **r})
    return pd.DataFrame(rows)


def condition_ranking_stability(long: pd.DataFrame, dataset: str | None = None,
                                policies=POLICIES) -> pd.DataFrame:
    """tau-b between the B0 ranking (mean over baseline runs) and each condition."""
    rows = []
    for policy in policies:
        df, _ = apply_policy_long(long, policy)
        if dataset:
            df = df[df["dataset"] == dataset]
        base = _scores(df, BASELINE)
        for cond in sorted(df["condition"].unique()):
            if cond == BASELINE:
                continue
            r = kendall_tau_b(base, _scores(df, cond))
            top_base = base.idxmax() if len(base) else None
            s = _scores(df, cond)
            rows.append({"dataset": dataset or "pooled", "condition": cond,
                         "parse_policy": policy, **r,
                         "top1_baseline": top_base,
                         "top1_condition": s.idxmax() if len(s) else None,
                         "top1_changed": bool(len(s) and top_base != s.idxmax())})
    return pd.DataFrame(rows)


def _item_vectors(long: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = long[long["condition"] == condition]
    return sub.pivot_table(index="parent_uid", columns="config_id",
                           values="is_correct", aggfunc="mean")


def separable_pairs(long: pd.DataFrame, dataset: str | None = None,
                    n_resamples: int = 10_000, seed: int = 20260815,
                    alpha: float = 0.05, policy: str = "incorrect") -> pd.DataFrame:
    """Paired bootstrap over items: which configuration pairs are statistically
    separable at baseline."""
    df, _ = apply_policy_long(long, policy)
    if dataset:
        df = df[df["dataset"] == dataset]
    wide = _item_vectors(df, BASELINE).dropna(axis=0, how="any")
    configs = sorted(wide.columns)
    rows = []
    for a, b in itertools.combinations(configs, 2):
        r = paired_bootstrap_diff(wide[a].values, wide[b].values,
                                  n_resamples=n_resamples, seed=seed, alpha=alpha)
        rows.append({"dataset": dataset or "pooled", "config_a": a, "config_b": b,
                     "parse_policy": policy, **r})
    return pd.DataFrame(rows)


def reversals(long: pd.DataFrame, sep: pd.DataFrame, dataset: str | None = None,
              policy: str = "incorrect", conditions=None) -> pd.DataFrame:
    """The headline number: fraction of baseline-separable pairs whose ordering
    reverses under a reasonable perturbation."""
    df, _ = apply_policy_long(long, policy)
    if dataset:
        df = df[df["dataset"] == dataset]
    base = _scores(df, BASELINE)
    conds = conditions or [c for c in sorted(df["condition"].unique()) if c != BASELINE]
    sep_pairs = sep[sep["separable"]][["config_a", "config_b"]].itertuples(index=False)
    sep_pairs = list(sep_pairs)
    rows = []
    for cond in conds:
        s = _scores(df, cond)
        n_rev = 0
        n_pairs = 0
        rev_list = []
        for a, b in sep_pairs:
            if a not in s.index or b not in s.index:
                continue
            n_pairs += 1
            if np.sign(base[a] - base[b]) != np.sign(s[a] - s[b]):
                n_rev += 1
                rev_list.append(f"{a}|{b}")
        rows.append({"dataset": dataset or "pooled", "condition": cond,
                     "parse_policy": policy,
                     "n_separable_pairs": n_pairs, "n_reversed": n_rev,
                     "reversal_fraction": (n_rev / n_pairs) if n_pairs else float("nan"),
                     "reversed_pairs": ";".join(rev_list[:50])})
    return pd.DataFrame(rows)


def reasoning_onoff_reversals(long: pd.DataFrame, policy: str = "incorrect") -> pd.DataFrame:
    """Reasoning ON/OFF pairs of the same weights, reported explicitly."""
    df, _ = apply_policy_long(long, policy)
    meta = df.drop_duplicates("config_id").set_index("config_id")[
        ["model_name", "reasoning_regime"]]
    rows = []
    for model, g in meta.groupby("model_name"):
        on = [c for c in g.index if str(g.loc[c, "reasoning_regime"]) in ("on", "high", "medium", "low")]
        off = [c for c in g.index if str(g.loc[c, "reasoning_regime"]) == "off"]
        if not on or not off:
            continue
        base = _scores(df, BASELINE)
        for a in on:
            for b in off:
                for cond in sorted(c for c in df["condition"].unique() if c != BASELINE):
                    s = _scores(df, cond)
                    if a not in s.index or b not in s.index:
                        continue
                    rows.append({
                        "model_name": model, "config_on": a, "config_off": b,
                        "condition": cond, "parse_policy": policy,
                        "baseline_gap": float(base[a] - base[b]),
                        "condition_gap": float(s[a] - s[b]),
                        "reversed": bool(np.sign(base[a] - base[b]) != np.sign(s[a] - s[b])),
                    })
    return pd.DataFrame(rows)
