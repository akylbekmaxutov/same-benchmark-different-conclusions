"""Cross-configuration flip consensus and the exclusion-sensitivity table.

A genuinely broken perturbation breaks the item for nearly everyone; genuine
instability scatters idiosyncratically.  Top-decile items are broken-perturbation
candidates, and every headline result is recomputed with them removed.
"""
from __future__ import annotations

import pandas as pd

from .core import BASELINE, POLICIES, apply_policy_paired


def consensus_ci(paired: pd.DataFrame, policy: str = "incorrect") -> pd.DataFrame:
    """For each (parent_uid, condition), the fraction of configurations that
    flip C->I."""
    df, _ = apply_policy_paired(paired, policy)
    df = df[df["condition"] != BASELINE]
    per_config = df.groupby(["parent_uid", "dataset", "condition", "config_id"]).agg(
        any_ci=("transition", lambda s: bool((s == "CI").any())),
        was_correct=("baseline_correct", lambda s: bool(s.astype(bool).any())),
    ).reset_index()
    g = per_config.groupby(["parent_uid", "dataset", "condition"]).agg(
        n_configs=("config_id", "nunique"),
        n_ci=("any_ci", "sum"),
        n_baseline_correct=("was_correct", "sum")).reset_index()
    g["consensus_ci_fraction"] = g["n_ci"] / g["n_configs"].replace(0, pd.NA)
    g["parse_policy"] = policy
    return g.sort_values("consensus_ci_fraction", ascending=False)


def top_decile_items(cons: pd.DataFrame, fraction: float = 0.10) -> set[tuple[str, str]]:
    """-> {(parent_uid, condition)} in the top decile of consensus C->I."""
    if cons.empty:
        return set()
    k = max(1, int(len(cons) * fraction))
    top = cons.nlargest(k, "consensus_ci_fraction")
    return set(zip(top["parent_uid"], top["condition"]))


def exclude(paired: pd.DataFrame, excluded: set[tuple[str, str]]) -> pd.DataFrame:
    if not excluded:
        return paired
    keys = list(zip(paired["parent_uid"], paired["condition"]))
    mask = [k not in excluded for k in keys]
    return paired[pd.Series(mask, index=paired.index)]


def exclusion_sensitivity(paired: pd.DataFrame, cons: pd.DataFrame,
                          fraction: float = 0.10, policies=POLICIES) -> pd.DataFrame:
    """Recompute flip rate and C->I with the top-decile items removed, and show
    the conclusions do not move.  This is the table that answers the
    perturbation-validity objection."""
    from .core import flip_rates, transition_matrix
    excluded = top_decile_items(cons, fraction)
    kept = exclude(paired, excluded)

    def _summ(df, tag):
        f = flip_rates(df, by=("config_id", "dataset", "condition"), policies=policies)
        t = transition_matrix(df, by=("config_id", "dataset", "condition"), policies=policies)
        m = f.merge(t[["config_id", "dataset", "condition", "parse_policy", "CI_rate"]],
                    on=["config_id", "dataset", "condition", "parse_policy"], how="left")
        return m.rename(columns={"flip_rate_mean": f"flip_rate_{tag}",
                                 "CI_rate": f"CI_rate_{tag}"})[
            ["config_id", "dataset", "condition", "parse_policy",
             f"flip_rate_{tag}", f"CI_rate_{tag}"]]

    a = _summ(paired, "all")
    b = _summ(kept, "excl")
    out = a.merge(b, on=["config_id", "dataset", "condition", "parse_policy"], how="left")
    out["flip_rate_delta"] = out["flip_rate_excl"] - out["flip_rate_all"]
    out["CI_rate_delta"] = out["CI_rate_excl"] - out["CI_rate_all"]
    out["n_items_excluded"] = len(excluded)
    out["exclusion_fraction"] = fraction
    return out


def export_top_consensus_flips(cons: pd.DataFrame, long: pd.DataFrame,
                               n: int = 50) -> list[dict]:
    """`top50_consensus_flips.jsonl` for the targeted manual inspection pass."""
    top = cons.nlargest(n, "consensus_ci_fraction")
    items = long.drop_duplicates("parent_uid").set_index("parent_uid")
    rows = []
    for _, r in top.iterrows():
        base = items.loc[r["parent_uid"]] if r["parent_uid"] in items.index else None
        rows.append({
            "parent_uid": r["parent_uid"], "dataset": r["dataset"],
            "condition": r["condition"],
            "consensus_ci_fraction": float(r["consensus_ci_fraction"]),
            "n_configs": int(r["n_configs"]), "n_ci": int(r["n_ci"]),
            "category": None if base is None else base.get("category"),
            "gold_answer_original": None if base is None else base.get("gold_answer_original"),
        })
    return rows
