#!/usr/bin/env python3
"""results/metrics/*.csv -> results/figures/*.{pdf,png} + LaTeX appendix tables.

Matplotlib only, no seaborn, one chart per figure, 300 dpi.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from stability.conditions import DISTRACTOR, VALIDITY_PRESERVING, paper_label
from stability.config import load_defaults
from stability.io import write_text

DPI = 300


def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def _read(metrics: Path, name: str) -> pd.DataFrame | None:
    p = metrics / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


# --------------------------------------------------------------- fig 1
def fig_taxonomy_counts(work: Path, outdir: Path) -> None:
    rep = work / "conditions" / "_build_report.json"
    if not rep.exists():
        return
    import json
    cells = json.loads(rep.read_text())["cells"]
    rows = [{"dataset": k.split("__")[0], "condition": k.split("__")[1],
             "n": v["n_rows"]} for k, v in cells.items()]
    df = pd.DataFrame(rows).pivot(index="condition", columns="dataset", values="n").fillna(0)
    order = [c for c in ["B0", "N1", "N2", "N3", "N4", "I1", "O1", "P1", "O1C", "NQR", "NQI"]
             if c in df.index]
    df = df.loc[order]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = np.zeros(len(df))
    for ds in df.columns:
        ax.bar([paper_label(i) for i in df.index], df[ds], bottom=bottom, label=ds)
        bottom += df[ds].values
    ax.set_ylabel("rendered evaluation units")
    ax.set_xlabel("condition")
    ax.set_title("Perturbation taxonomy: item counts per condition")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, outdir, "fig1_taxonomy_counts")


# --------------------------------------------------------------- fig 2
def fig_baseline_table(metrics: Path, outdir: Path, policy: str) -> None:
    acc = _read(metrics, "baseline_accuracy")
    fl = _read(metrics, "self_flip_floor")
    if acc is None or fl is None:
        return
    a = acc[acc["parse_policy"] == policy]
    f = fl[fl["parse_policy"] == policy]
    m = a.merge(f[["config_id", "dataset", "self_flip_floor"]],
                on=["config_id", "dataset"], how="left")
    piv = m.pivot_table(index="config_id", columns="dataset",
                        values="baseline_accuracy_mean")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(piv))))
    y = np.arange(len(piv))
    width = 0.8 / max(1, len(piv.columns))
    for i, ds in enumerate(piv.columns):
        ax.barh(y + i * width, piv[ds].values, height=width, label=ds)
    ax.set_yticks(y + 0.4 - width / 2)
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xlabel("baseline accuracy (mean over B0 runs)")
    ax.set_title(f"Baseline performance  [parse policy: {policy}]")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, outdir, f"fig2_baseline_accuracy_{policy}")


# --------------------------------------------------------------- fig 3
def fig_flip_vs_floor(metrics: Path, outdir: Path, policy: str) -> None:
    fr = _read(metrics, "flip_rates")
    tr = _read(metrics, "transitions")
    if fr is None:
        return
    f = fr[(fr["parse_policy"] == policy) & (fr["condition"] != "B0")]
    vp = sorted(VALIDITY_PRESERVING - {"B0"})
    f = f[f["condition"].isin(vp + sorted(DISTRACTOR))]
    g = f.groupby("condition").agg(flip=("flip_rate_mean", "mean"),
                                   floor=("self_flip_floor", "mean")).reindex(
        [c for c in ["N1", "N2", "I1", "O1", "P1", "N3", "N4"] if c in f["condition"].values])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(g))
    ax.bar(x, g["flip"].values, width=0.6,
           color=["#4878A8" if c in vp else "#C0603A" for c in g.index],
           label="flip rate")
    ax.plot(x, g["floor"].values, "k--o", ms=4, lw=1.2, label="self-flip floor")
    if tr is not None:
        t = tr[(tr["parse_policy"] == policy)].groupby("condition")["CI_rate"].mean()
        ax.plot(x, [t.get(c, np.nan) for c in g.index], color="#666", marker="s",
                ms=4, lw=1.0, ls=":", label="C→I rate")
    ax.set_xticks(x)
    ax.set_xticklabels([paper_label(c) for c in g.index])
    ax.set_ylabel("rate")
    ax.set_title(f"Flip rate against the self-flip floor (never subtracted)  "
                 f"[{policy}]", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, outdir, f"fig3_flip_vs_floor_{policy}")


# --------------------------------------------------------------- fig 4
def fig_ranking(metrics: Path, outdir: Path, policy: str) -> None:
    rs = _read(metrics, "ranking_stability")
    rev = _read(metrics, "ranking_reversals")
    if rs is None:
        return
    r = rs[(rs["parse_policy"] == policy) & (rs["dataset"] == "pooled")
           & rs["condition"].notna()]
    if r.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(r))
    ax.bar(x, r["tau_b"].values, width=0.6, color="#4878A8")
    b = rs[(rs["parse_policy"] == policy) & (rs["dataset"] == "pooled")
           & rs["condition"].isna()]
    if len(b):
        ax.axhline(b["tau_b"].mean(), color="k", ls="--", lw=1.2,
                   label=f"B0-vs-B0 τ-b = {b['tau_b'].mean():.3f}")
    if rev is not None:
        rv = rev[(rev["dataset"] == "pooled")].set_index("condition")["reversal_fraction"]
        for i, c in enumerate(r["condition"]):
            if c in rv.index and not np.isnan(rv[c]):
                ax.text(i, r["tau_b"].values[i] + 0.02,
                        f"{rv[c]:.0%} rev", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([paper_label(c) for c in r["condition"]])
    ax.set_ylabel("Kendall τ-b vs. baseline ranking")
    ax.set_ylim(min(0, r["tau_b"].min() - 0.1), 1.05)
    ax.set_title(f"Ranking stability; labels give the share of baseline-separable "
                 f"pairs that reverse  [{policy}]", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, outdir, f"fig4_ranking_stability_{policy}")


# --------------------------------------------------------------- fig 5
def fig_margin(metrics: Path, outdir: Path, policy: str) -> None:
    ms = _read(metrics, "flips_by_margin_stratum")
    if ms is None:
        return
    m = ms[(ms["parse_policy"] == policy) & (ms["condition"] != "B0")]
    if m.empty:
        return
    g = m.groupby("margin_stratum").agg(flip=("flip_rate", "mean"),
                                        ci=("CI_rate", "mean"),
                                        n=("n_pairs", "sum")).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["flip"].values, width=0.4, label="flip rate", color="#4878A8")
    ax.bar(x + 0.2, g["ci"].values, width=0.4, label="C→I rate", color="#C0603A")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index)
    ax.set_xlabel("baseline run-agreement stratum")
    ax.set_ylabel("rate")
    ax.set_title(f"Margin decomposition  [{policy}]")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, outdir, f"fig5_margin_decomposition_{policy}")


# --------------------------------------------------------------- fig 6
def fig_panel(metrics: Path, outdir: Path, policy: str) -> None:
    fr = _read(metrics, "flip_rates")
    long_meta = _read(metrics, "generation_health")
    acc = _read(metrics, "baseline_accuracy")
    if fr is None or acc is None:
        return
    f = fr[(fr["parse_policy"] == policy) & (fr["condition"].isin(
        sorted(VALIDITY_PRESERVING - {"B0"})))]
    g = f.groupby("config_id")["flip_rate_mean"].mean()
    a = acc[acc["parse_policy"] == policy].groupby("config_id")["baseline_accuracy_mean"].mean()
    meta = _read(metrics, "config_axes")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    common = sorted(set(g.index) & set(a.index))
    axes[0].scatter(a[common], g[common], s=22, color="#4878A8")
    for c in common:
        axes[0].annotate(c, (a[c], g[c]), fontsize=5.5,
                         xytext=(2, 2), textcoords="offset points")
    axes[0].set_xlabel("baseline accuracy")
    axes[0].set_ylabel("mean validity-preserving flip rate")
    axes[0].set_title("accuracy vs. instability", fontsize=9)
    if meta is not None:
        m = meta.set_index("config_id")
        for ax, col, title in ((axes[1], "reasoning_regime", "by reasoning regime"),
                               (axes[2], "kazakh_specialized", "Kazakh-oriented")):
            keys = sorted({str(m.loc[c, col]) for c in common if c in m.index})
            vals = [[g[c] for c in common if c in m.index and str(m.loc[c, col]) == k]
                    for k in keys]
            ax.boxplot([v for v in vals if v],
                       tick_labels=[k for k, v in zip(keys, vals) if v])
            ax.set_title(title, fontsize=9)
            ax.set_ylabel("flip rate")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Reasoning regime × scale × Kazakh-oriented  [{policy}]", fontsize=10)
    save(fig, outdir, f"fig6_panel_{policy}")


# --------------------------------------------------------------- appendix
def latex_appendix(metrics: Path, outdir: Path) -> None:
    tables = {
        "accuracy_by_run": "Accuracy per configuration × dataset × condition × run",
        "flip_rates": "Flip rates with the self-flip floor",
        "transitions": "Transition matrices",
        "accuracy_by_category": "Per-category accuracy",
        "format_health": "Format compliance and parse tiers",
        "generation_health": "Generation health: truncation and token volume",
        "exclusion_sensitivity": "Exclusion sensitivity (top-decile consensus items removed)",
    }
    chunks = ["% auto-generated by scripts/08_figures.py -- do not hand-edit", ""]
    for name, caption in tables.items():
        df = _read(metrics, name)
        if df is None or df.empty:
            continue
        num = df.select_dtypes("number").columns
        df = df.copy()
        df[num] = df[num].round(4)
        chunks.append(df.to_latex(index=False, longtable=True, escape=True,
                                  caption=caption, label=f"tab:{name}",
                                  na_rep="--"))
        chunks.append("")
    outdir.mkdir(parents=True, exist_ok=True)
    write_text(outdir / "appendix_tables.tex", "\n".join(chunks))
    print("  appendix_tables.tex")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--policy", default="incorrect", choices=["incorrect", "excluded"])
    args = ap.parse_args()

    d = load_defaults()
    results = Path(args.results or d["results_dir"])
    metrics = results / "metrics"
    outdir = results / "figures"
    work = Path(d["work_dir"])
    if not metrics.exists():
        print(f"no metrics at {metrics}; run scripts/07_metrics.py first")
        return 1

    print("figures ->", outdir)
    fig_taxonomy_counts(work, outdir)
    for policy in ("incorrect", "excluded"):
        fig_baseline_table(metrics, outdir, policy)
        fig_flip_vs_floor(metrics, outdir, policy)
        fig_ranking(metrics, outdir, policy)
        fig_margin(metrics, outdir, policy)
        fig_panel(metrics, outdir, policy)
    latex_appendix(metrics, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
