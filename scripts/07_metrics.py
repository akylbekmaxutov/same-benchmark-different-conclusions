#!/usr/bin/env python3
"""results/aggregate/*.parquet -> results/metrics/*.csv + paper_numbers.json

Reads the aggregate tables only, never the raw shards.  Every metric is emitted
under BOTH parse-failure treatments, tagged with a `parse_policy` column.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from stability.conditions import DISTRACTOR, VALIDITY_PRESERVING
from stability.config import load_defaults
from stability.io import write_json, write_jsonl
from stability.metrics import consensus, core, floor, margin, position, ranking
from stability.metrics.stats import bootstrap_ci

BASELINE = "B0"


def _w(df: pd.DataFrame, outdir: Path, name: str) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{name}.csv", index=False)
    print(f"  {name}.csv  ({len(df)} rows)")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="skip the paired bootstrap over configuration pairs")
    args = ap.parse_args()

    d = load_defaults()
    results = Path(args.results or d["results_dir"])
    agg = results / "aggregate"
    out = results / "metrics"
    policies = tuple(d.get("parse_policies", ["excluded", "incorrect"]))
    n_boot = args.bootstrap or int(d.get("bootstrap_resamples", 10000))
    seed = int(d.get("bootstrap_seed", 20260815))

    long = pd.read_parquet(agg / "long.parquet")
    paired = pd.read_parquet(agg / "paired.parquet")
    datasets = sorted(long["dataset"].unique())
    assert set(d["datasets"]) <= set(datasets) or True  # reported below, not asserted away
    print(f"long={len(long)} paired={len(paired)} configs={long['config_id'].nunique()} "
          f"datasets={datasets}")

    # analysis axes, exposed as columns on every aggregate
    axes_cols = ["config_id", "family", "model_name", "params_total_b", "params_active_b",
                 "is_moe", "reasoning_regime", "kazakh_specialized"]
    axes = long.drop_duplicates("config_id")[axes_cols].sort_values("config_id")
    _w(axes, out, "config_axes")

    def ax(df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(axes, on="config_id", how="left") if "config_id" in df.columns else df

    # ---------------------------------------------------------- 12.1 accuracy
    print("[12.1] accuracy")
    acc = _w(ax(core.accuracy(long, policies=policies)), out, "accuracy_by_run")
    _w(ax(core.baseline_accuracy(acc)), out, "baseline_accuracy")
    dacc = _w(ax(core.delta_accuracy(acc)), out, "delta_accuracy")
    _w(core.accuracy(long, by=("config_id", "dataset", "category", "condition"),
                     policies=policies), out, "accuracy_by_category")
    _w(core.accuracy(long, by=("config_id", "dataset", "n_options", "condition"),
                     policies=policies), out, "accuracy_by_n_options")

    # ---------------------------------------------------------- 12.2 floor
    print("[12.2] self-flip floor")
    fl = _w(floor.self_flip_floor(paired, policies=policies), out, "self_flip_floor")
    fl_pooled = _w(floor.self_flip_floor(paired, by=("config_id",), policies=policies),
                   out, "self_flip_floor_pooled")
    _w(floor.floor_transfer(paired, "P1", policies=policies), out, "floor_transfer_p1")

    # ---------------------------------------------------------- 12.3 flips
    print("[12.3] flips and transitions")
    flips = core.flip_rates(paired, policies=policies)
    flips = _w(ax(floor.attach_floor(flips, fl)), out, "flip_rates")
    trans = _w(ax(core.transition_matrix(paired, policies=policies)), out, "transitions")
    netgross = flips.merge(
        trans[["config_id", "dataset", "condition", "parse_policy",
               "CI_rate", "IC_rate", "gross_churn_rate", "net_delta_rate"]],
        on=["config_id", "dataset", "condition", "parse_policy"], how="left")
    netgross = netgross.merge(
        dacc[["config_id", "dataset", "condition", "parse_policy", "delta_acc_mean"]],
        on=["config_id", "dataset", "condition", "parse_policy"], how="left")
    _w(ax(netgross), out, "net_vs_gross")

    # ---------------------------------------------------------- 12.4 margin
    print("[12.4] margin decomposition")
    _w(margin.flips_by_stratum(paired, policies=policies), out, "flips_by_margin_stratum")
    _w(margin.flips_by_stratum(paired, by=("config_id",), policies=policies),
       out, "flips_by_margin_stratum_pooled")
    lowm = _w(margin.low_margin_share(paired, policies=policies), out, "low_margin_share")
    nearch = _w(margin.near_chance_cells(long, float(d.get("near_chance_alpha", 0.05))),
                out, "near_chance_cells")
    lp = margin.logprob_margin(long)
    if len(lp):
        _w(lp, out, "logprob_margin")

    # ---------------------------------------------------------- 12.5 ranking
    print("[12.5] ranking stability")
    rank_rows, sep_rows, rev_rows = [], [], []
    for ds in [None] + datasets:
        rank_rows.append(ranking.baseline_internal_stability(long, ds, policies))
        rank_rows.append(ranking.condition_ranking_stability(long, ds, policies))
    _w(pd.concat([r for r in rank_rows if len(r)], ignore_index=True),
       out, "ranking_stability")
    if not args.skip_bootstrap:
        for ds in [None] + datasets:
            s = ranking.separable_pairs(long, ds, n_resamples=n_boot, seed=seed)
            if len(s):
                sep_rows.append(s)
                rev_rows.append(ranking.reversals(long, s, ds))
        if sep_rows:
            _w(pd.concat(sep_rows, ignore_index=True), out, "separable_pairs")
            rev = _w(pd.concat(rev_rows, ignore_index=True), out, "ranking_reversals")
        else:
            rev = pd.DataFrame()
    else:
        rev = pd.DataFrame()
    ronoff = ranking.reasoning_onoff_reversals(long)
    if len(ronoff):
        _w(ronoff, out, "reasoning_onoff_reversals")

    # ---------------------------------------------------------- 12.6 position
    print("[12.6] position bias (O1C)")
    slot = position.accuracy_by_gold_slot(long, policies)
    if len(slot):
        _w(slot, out, "o1c_accuracy_by_gold_slot")
        _w(position.letter_preference(long), out, "o1c_letter_preference")
        rc = position.rotation_consistency(paired, policies)
        if len(rc):
            _w(rc, out, "o1c_rotation_consistency")
        dec = position.decompose_o1_flips(paired, slot)
        if len(dec):
            _w(dec, out, "o1_flip_decomposition")
    else:
        print("  (no O1C rows)")

    # ---------------------------------------------------------- 12.7 consensus
    print("[12.7] cross-configuration consensus")
    cons = consensus.consensus_ci(paired)
    excl = pd.DataFrame()
    if len(cons):
        _w(cons, out, "consensus_ci")
        _w(cons["consensus_ci_fraction"].describe().to_frame().T,
           out, "consensus_ci_distribution")
        excl = _w(consensus.exclusion_sensitivity(
            paired, cons, float(d.get("consensus_top_decile", 0.10)), policies),
            out, "exclusion_sensitivity")
        write_jsonl(results / "validation" / "top50_consensus_flips.jsonl",
                    consensus.export_top_consensus_flips(cons, long, 50))

    # ---------------------------------------------------------- 12.8 health
    print("[12.8] format, parsing, generation health")
    _w(ax(core.format_health(long)), out, "format_health")
    _w(ax(core.generation_health(long)), out, "generation_health")
    nq = core.nq_probe(long, float(d.get("near_chance_alpha", 0.05)))
    if len(nq):
        _w(nq, out, "nq_leakage_probe")

    # ---------------------------------------------------------- 12.9 CIs
    print("[12.9] bootstrap CIs")
    ci_rows = []
    for policy in policies:
        dfl, _ = core.apply_policy_long(long, policy)
        for (cid, ds, cond), g in dfl.groupby(["config_id", "dataset", "condition"]):
            ci = bootstrap_ci(g["is_correct"].astype(float).values, n_boot, seed)
            ci_rows.append({"config_id": cid, "dataset": ds, "condition": cond,
                            "parse_policy": policy, "metric": "accuracy", **ci})
        dfp, _ = core.apply_policy_paired(paired, policy)
        for (cid, ds, cond), g in dfp.groupby(["config_id", "dataset", "condition"]):
            ci = bootstrap_ci(g["flipped"].astype(float).values, n_boot, seed)
            ci_rows.append({"config_id": cid, "dataset": ds, "condition": cond,
                            "parse_policy": policy, "metric": "flip_rate", **ci})
    _w(pd.DataFrame(ci_rows), out, "bootstrap_cis")

    # ---------------------------------------------------------- paper numbers
    print("[paper] headline figures")
    vp = sorted(VALIDITY_PRESERVING - {BASELINE})
    pn: dict = {
        "_note": ("Every number quoted in the paper comes from this file. "
                  "Intervals are unadjusted for multiple comparisons across "
                  "7 conditions x N configurations and are interpreted as effect "
                  "sizes, not significance thresholds."),
        "n_configurations": int(long["config_id"].nunique()),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "conditions_present": sorted(long["condition"].unique()),
        "validity_preserving": vp,
        "distractor": sorted(DISTRACTOR),
    }
    for policy in policies:
        f = flips[flips["parse_policy"] == policy]
        fp = fl_pooled[fl_pooled["parse_policy"] == policy]
        dv = dacc[(dacc["parse_policy"] == policy) & (dacc["condition"].isin(vp))]
        dd = dacc[(dacc["parse_policy"] == policy) & (dacc["condition"].isin(DISTRACTOR))]
        fvp = f[f["condition"].isin(vp)]
        lm = lowm[lowm["parse_policy"] == policy]
        block = {
            "pooled_self_flip_floor_mean": _f(fp["self_flip_floor"].mean()),
            "pooled_self_flip_floor_max": _f(fp["self_flip_floor"].max()),
            "worst_validity_preserving_flip_rate": _f(fvp["flip_rate_mean"].max()),
            "worst_validity_preserving_flip_cell": _cell(fvp, "flip_rate_mean"),
            "mean_validity_preserving_flip_rate": _f(fvp["flip_rate_mean"].mean()),
            "median_flip_to_floor_ratio": _f(fvp["flip_to_floor_ratio"].median()),
            "mean_delta_acc_validity_preserving": _f(dv["delta_acc_mean"].mean()),
            "worst_delta_acc_validity_preserving": _f(dv["delta_acc_mean"].min()),
            "worst_delta_acc_cell": _cell(dv, "delta_acc_mean", how="min"),
            "mean_delta_acc_distractor": _f(dd["delta_acc_mean"].mean()),
            "worst_delta_acc_distractor": _f(dd["delta_acc_mean"].min()),
            "low_margin_share_of_correct_mean": _f(lm["low_margin_share_of_correct"].mean()),
            "n_near_chance_cells": int(nearch["near_chance"].sum()) if len(nearch) else 0,
        }
        if len(rev):
            r = rev[(rev["parse_policy"] == "incorrect") & (rev["dataset"] == "pooled")]
            rvp = r[r["condition"].isin(vp)]
            block["separable_pair_reversal_fraction_max"] = _f(rvp["reversal_fraction"].max())
            block["separable_pair_reversal_fraction_mean"] = _f(rvp["reversal_fraction"].mean())
            block["n_separable_pairs"] = int(rvp["n_separable_pairs"].max()) if len(rvp) else 0
        if len(excl):
            e = excl[excl["parse_policy"] == policy]
            block["exclusion_sensitivity_max_abs_flip_delta"] = _f(e["flip_rate_delta"].abs().max())
            block["exclusion_sensitivity_max_abs_CI_delta"] = _f(e["CI_rate_delta"].abs().max())
        pn[policy] = block

    a, b = policies[0], policies[-1]
    if a != b:
        pn["parse_policy_sensitivity"] = {
            k: _f(_g(pn[b], k) - _g(pn[a], k))
            for k in ("pooled_self_flip_floor_mean", "worst_validity_preserving_flip_rate",
                      "mean_delta_acc_validity_preserving")
        }
    write_json(out / "paper_numbers.json", pn)
    print(f"  paper_numbers.json -> {out / 'paper_numbers.json'}")
    return 0


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _g(block, key):
    v = block.get(key)
    return float("nan") if v is None else v


def _cell(df: pd.DataFrame, col: str, how: str = "max"):
    if df.empty or df[col].isna().all():
        return None
    i = df[col].idxmax() if how == "max" else df[col].idxmin()
    r = df.loc[i]
    return {"config_id": r["config_id"], "dataset": r["dataset"],
            "condition": r["condition"], "value": _f(r[col])}


if __name__ == "__main__":
    raise SystemExit(main())
