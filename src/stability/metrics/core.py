"""Accuracy, ΔAcc, flip rates, transition matrices.

Every function is computed under BOTH parse-failure treatments and tagged with
a `parse_policy` column.  The harness never decided; the metrics layer reports
both, per the pre-registration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..parse import FAIL

POLICIES = ("excluded", "incorrect")
BASELINE = "B0"

AXES = ["family", "params_total_b", "params_active_b", "is_moe",
        "reasoning_regime", "kazakh_specialized"]


# ------------------------------------------------------------------ policy
def apply_policy_long(long: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, int]:
    """`excluded` drops parse failures; `incorrect` keeps them as wrong answers."""
    if policy == "incorrect":
        return long, 0
    if policy != "excluded":
        raise ValueError(f"unknown parse policy {policy!r}")
    mask = long["answer_eff"] != FAIL
    return long[mask], int((~mask).sum())


def apply_policy_paired(paired: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, int]:
    """`excluded` drops any pair where EITHER side failed to parse."""
    if policy == "incorrect":
        return paired, 0
    if policy != "excluded":
        raise ValueError(f"unknown parse policy {policy!r}")
    mask = ~paired["pair_has_fail"].astype(bool)
    return paired[mask], int((~mask).sum())


# ------------------------------------------------------------------ accuracy
def accuracy(long: pd.DataFrame, by=("config_id", "dataset", "condition", "run_index"),
             policies=POLICIES) -> pd.DataFrame:
    out = []
    for policy in policies:
        df, dropped = apply_policy_long(long, policy)
        g = df.groupby(list(by), dropna=False).agg(
            n=("is_correct", "size"),
            n_correct=("is_correct", "sum"),
            format_valid_rate=("format_valid", "mean"),
        ).reset_index()
        # parse-failure rate is reported per cell from the UNFILTERED table so the
        # gap between the two policies is visible rather than inferred
        pf = long.assign(_fail=(long["answer_eff"] == FAIL)).groupby(
            list(by), dropna=False)["_fail"].agg(["mean", "sum", "size"]).reset_index()
        pf = pf.rename(columns={"mean": "parse_failure_rate", "sum": "n_parse_failed",
                                "size": "n_before_policy"})
        g = g.merge(pf, on=list(by), how="left")
        g["accuracy"] = g["n_correct"] / g["n"].replace(0, np.nan)
        g["parse_policy"] = policy
        g["n_dropped_by_policy"] = g["n_before_policy"] - g["n"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def baseline_accuracy(acc: pd.DataFrame,
                      by=("config_id", "dataset", "parse_policy")) -> pd.DataFrame:
    """Mean over the baseline runs, with the spread reported as a variance band."""
    b = acc[acc["condition"] == BASELINE]
    g = b.groupby(list(by), dropna=False)["accuracy"].agg(
        baseline_accuracy_mean="mean", baseline_accuracy_sd="std",
        baseline_accuracy_min="min", baseline_accuracy_max="max",
        baseline_n_runs="size").reset_index()
    return g


def delta_accuracy(acc: pd.DataFrame) -> pd.DataFrame:
    """ΔAcc(p) computed three times, once against each individual baseline run;
    mean and spread reported."""
    base = acc[acc["condition"] == BASELINE][
        ["config_id", "dataset", "parse_policy", "run_index", "accuracy"]
    ].rename(columns={"run_index": "baseline_run_index", "accuracy": "baseline_accuracy"})
    pert = acc[acc["condition"] != BASELINE]
    j = pert.merge(base, on=["config_id", "dataset", "parse_policy"], how="left")
    j["delta_acc"] = j["accuracy"] - j["baseline_accuracy"]
    g = j.groupby(["config_id", "dataset", "condition", "parse_policy"],
                  dropna=False)["delta_acc"].agg(
        delta_acc_mean="mean", delta_acc_sd="std",
        delta_acc_min="min", delta_acc_max="max", n_comparisons="size").reset_index()
    return g


# ------------------------------------------------------------------ flips
def flip_rates(paired: pd.DataFrame,
               by=("config_id", "dataset", "condition"),
               policies=POLICIES) -> pd.DataFrame:
    """Share of items whose prediction changes from baseline, computed
    symmetrically against each of the three individual baseline runs.  Mean and
    spread over those three; never against a majority vote."""
    out = []
    for policy in policies:
        df, dropped = apply_policy_paired(paired, policy)
        per_run = df.groupby(list(by) + ["baseline_run_index", "perturbed_run_index"],
                             dropna=False).agg(
            flip_rate=("flipped", "mean"), n_pairs=("flipped", "size")).reset_index()
        g = per_run.groupby(list(by), dropna=False).agg(
            flip_rate_mean=("flip_rate", "mean"),
            flip_rate_sd=("flip_rate", "std"),
            flip_rate_min=("flip_rate", "min"),
            flip_rate_max=("flip_rate", "max"),
            n_run_pairs=("flip_rate", "size"),
            n_pairs=("n_pairs", "sum")).reset_index()
        drop = paired.groupby(list(by), dropna=False)["pair_has_fail"].agg(
            ["sum", "size"]).reset_index().rename(
            columns={"sum": "n_pairs_with_fail", "size": "n_pairs_before_policy"})
        g = g.merge(drop, on=list(by), how="left")
        g["n_pairs_dropped_by_policy"] = (
            g["n_pairs_with_fail"] if policy == "excluded" else 0)
        g["parse_policy"] = policy
        out.append(g)
    return pd.concat(out, ignore_index=True)


def transition_matrix(paired: pd.DataFrame,
                      by=("config_id", "dataset", "condition"),
                      policies=POLICIES) -> pd.DataFrame:
    """C->C, C->I, I->C, I->I per cell.  C->I under a validity-preserving
    condition is the primary degradation indicator."""
    out = []
    for policy in policies:
        df, _ = apply_policy_paired(paired, policy)
        ct = df.groupby(list(by) + ["transition"], dropna=False).size().unstack(
            "transition", fill_value=0)
        for t in ("CC", "CI", "IC", "II"):
            if t not in ct.columns:
                ct[t] = 0
        ct = ct[["CC", "CI", "IC", "II"]].reset_index()
        ct["n"] = ct[["CC", "CI", "IC", "II"]].sum(axis=1)
        for t in ("CC", "CI", "IC", "II"):
            ct[f"{t}_rate"] = ct[t] / ct["n"].replace(0, np.nan)
        # net-vs-gross: how much churn hides under a flat accuracy delta
        ct["gross_churn_rate"] = ct["CI_rate"] + ct["IC_rate"]
        ct["net_delta_rate"] = ct["IC_rate"] - ct["CI_rate"]
        ct["parse_policy"] = policy
        out.append(ct)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ health
def format_health(long: pd.DataFrame,
                  by=("config_id", "dataset", "condition")) -> pd.DataFrame:
    """JSON compliance, parse tiers, homoglyphs -- reported separately from
    semantic correctness, and comparable across effort levels and instruction
    language."""
    g = long.groupby(list(by), dropna=False).agg(
        n=("format_valid", "size"),
        format_valid_rate=("format_valid", "mean"),
        parse_failure_rate=("parse_tier", lambda s: float((s == "failed").mean())),
        homoglyph_rate=("homoglyph_normalized", "mean"),
    ).reset_index()
    tiers = long.groupby(list(by) + ["parse_tier"], dropna=False).size().unstack(
        "parse_tier", fill_value=0)
    tiers.columns = [f"tier_{c}" for c in tiers.columns]
    return g.merge(tiers.reset_index(), on=list(by), how="left")


def generation_health(long: pd.DataFrame,
                      by=("config_id", "dataset", "condition")) -> pd.DataFrame:
    """Truncation, token volume, errors, latency -- the evidence that reasoning
    regimes genuinely differ, and the schedule risk indicator."""
    def q(s, p):
        s = s.dropna()
        return float(np.quantile(s, p)) if len(s) else float("nan")

    g = long.groupby(list(by), dropna=False).apply(lambda s: pd.Series({
        "n": len(s),
        "truncation_rate": float(s["truncated"].astype(bool).mean()),
        "completion_tokens_mean": float(s["completion_tokens"].dropna().mean()
                                        if s["completion_tokens"].notna().any() else np.nan),
        "completion_tokens_median": q(s["completion_tokens"], 0.5),
        "completion_tokens_p95": q(s["completion_tokens"], 0.95),
        "reasoning_tokens_mean": float(s["reasoning_tokens"].dropna().mean()
                                       if s["reasoning_tokens"].notna().any() else np.nan),
        "reasoning_tokens_median": q(s["reasoning_tokens"], 0.5),
        "reasoning_tokens_p95": q(s["reasoning_tokens"], 0.95),
        "prompt_tokens_mean": float(s["prompt_tokens"].dropna().mean()
                                    if s["prompt_tokens"].notna().any() else np.nan),
        "error_rate": float(s["error"].notna().mean()),
        "mean_attempts": float(s["attempt_count"].dropna().mean()
                               if s["attempt_count"].notna().any() else np.nan),
        "latency_mean_s": float(s["latency_s"].dropna().mean()
                                if s["latency_s"].notna().any() else np.nan),
        "wall_clock_s_est": float(s["latency_s"].dropna().sum()),
    }), include_groups=False).reset_index()
    errs = long[long["error_kind"].notna()].groupby(
        list(by) + ["error_kind"]).size().unstack("error_kind", fill_value=0)
    if not errs.empty:
        errs.columns = [f"error_{c}" for c in errs.columns]
        g = g.merge(errs.reset_index(), on=list(by), how="left")
    return g


def nq_probe(long: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """No-question leakage probe: NQR (relevant noise) vs NQI (irrelevant noise,
    the negative control), tested against per-cell chance rather than eyeballed."""
    from .stats import binom_vs_chance
    sub = long[long["condition"].isin(["NQR", "NQI"])]
    rows = []
    for (cid, ds, cond), g in sub.groupby(["config_id", "dataset", "condition"],
                                          dropna=False):
        p_chance = float((1.0 / g["n_options"].astype(float)).mean())
        res = binom_vs_chance(int(g["is_correct"].astype(bool).sum()), len(g),
                              p_chance, alpha)
        rows.append({"config_id": cid, "dataset": ds, "condition": cond,
                     "noise_type": "relevant" if cond == "NQR" else "irrelevant",
                     **res})
    return pd.DataFrame(rows)
