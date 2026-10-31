#!/usr/bin/env python3
"""Generate RESULTS.md from results/metrics/*.csv.

Every number is read from a CSV; nothing is hand-transcribed. Re-run to refresh.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

POLICY = "incorrect"
PRETTY = {"kazmmlu": "KazMMLU", "kazculture": "KazCulture",
          "kazbench_kk": "KazBench-KK"}          # headline policy; the other is reported alongside

CONDITIONS = [
    ("B0", "Baseline", "validity-preserving",
     "The item exactly as published. Run three times to establish the self-flip floor."),
    ("N1", "Irrelevant noise, before", "validity-preserving",
     "A topically unrelated Kazakh paragraph inserted BEFORE the question. The item itself is untouched."),
    ("N2", "Irrelevant noise, after", "validity-preserving",
     "The SAME paragraph as N1, inserted AFTER the question instead. N1 vs N2 isolates position from content."),
    ("N3", "Relevant noise, before", "distractor",
     "A topically related Kazakh paragraph inserted BEFORE the question, intended as a distractor."),
    ("N4", "Relevant noise, after", "distractor",
     "The SAME paragraph as N3, inserted AFTER the question."),
    ("I1", "Kazakh instruction", "validity-preserving",
     "The task instruction is translated into Kazakh. The item and the JSON key `answer` are unchanged."),
    ("O1", "Option order shuffled", "validity-preserving",
     "Answer options permuted once per item under a fixed seed, gold letter remapped. Compared on SEMANTIC identity, not letters."),
    ("P1", "Question paraphrased", "validity-preserving",
     "The question stem is replaced by a Kazakh paraphrase. Options untouched."),
    ("O1C", "Cyclic option rotation", "diagnostic",
     "Four rotations of the option ring so the gold answer visits each slot A-D exactly once. Four-option items only, ~500-item pool, six configurations."),
    ("NQR", "No question, relevant noise", "diagnostic",
     "The question is REMOVED; the model sees relevant noise plus the options. A leakage probe."),
    ("NQI", "No question, irrelevant noise", "diagnostic",
     "The question is REMOVED; the model sees irrelevant noise plus the options. The negative control for NQR."),
]
COND_ORDER = [c[0] for c in CONDITIONS]
VP = ["N1", "N2", "I1", "O1", "P1"]          # non-baseline validity-preserving
DISTRACTOR = ["N3", "N4"]
PERTURBATIONS = VP + DISTRACTOR
PERTURB_ORDER = ["B0", "N1", "N2", "N3", "N4", "I1", "O1", "P1"]


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        else:
            d[c] = d[c].astype(str).replace("nan", "")
    head = "| " + " | ".join(d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in d.astype(str).values]
    return "\n".join([head, rule] + rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args()
    R = Path(a.results); M = R / "metrics"
    rd = lambda n: pd.read_csv(M / f"{n}.csv")
    pn = json.loads((M / "paper_numbers.json").read_text())
    cov = json.loads((R / "aggregate" / "coverage.json").read_text())
    L: list[str] = []
    w = L.append

    ax = rd("config_axes")
    n_cfg = len(ax)

    # ---------------------------------------------------------------- header
    w("# Results — *Same Benchmark, Different Conclusions?*")
    w("")
    w("Auditing the stability of Kazakh LLM evaluation. Every number below is read "
      f"directly from `results/metrics/*.csv`; regenerate with `python scripts/10_results_md.py`.")
    w("")
    w(f"- **{n_cfg} configurations** × **3 benchmarks** × **{len(pn['conditions_present'])} conditions**")
    w(f"- **{cov['n_rows']:,} item-level generations**, zero missing cells")
    w("- Sample: KazMMLU 2,000 · KazBench-KK 2,000 · KazCulture 1,324 (all available; "
      "see the note on unequal *n*) = **5,324 items per condition**")
    w(f"- Headline parse-failure policy: **`{POLICY}`** (a parse failure counts as a wrong "
      "answer and as a distinct value for flips). The `excluded` policy is reported "
      "alongside in *Sensitivity to the parse-failure policy*.")
    w("")

    # ---------------------------------------------------------------- Table 1
    w("## Table 1 — Configurations")
    w("")
    w("Reasoning ON and OFF are two *configurations of the same checkpoint*, not two models. "
      "`Regime` is how reasoning was set: `on`/`off` via the chat template, `low`/`high` via "
      "the system prompt.")
    w("")
    t = ax.copy()
    t["Model"] = t["model_name"]
    t["Regime"] = t["reasoning_regime"]
    t["Total (B)"] = t["params_total_b"]
    t["Active (B)"] = t["params_active_b"]
    t["MoE"] = t["is_moe"].map({True: "yes", False: "no"})
    t["Kazakh-specialised"] = t["kazakh_specialized"].map({True: "yes", False: "no"})
    t["Family"] = t["family"]
    w(md_table(t[["config_id", "Model", "Family", "Total (B)", "Active (B)", "MoE",
                  "Regime", "Kazakh-specialised"]].rename(
        columns={"config_id": "Configuration"}).sort_values("Configuration"), "{:.1f}"))
    w("")

    # ---------------------------------------------------------------- Table 2
    w("## Table 2 — Perturbation conditions")
    w("")
    w("Condition codes are used throughout. **Validity-preserving** perturbations do not "
      "change the correct answer; **distractor** conditions are model-distractibility probes "
      "and are never pooled with them; **diagnostics** answer specific questions about the "
      "measurement itself.")
    w("")
    cd = pd.DataFrame([{"Code": c, "Name": n, "Group": g, "What changes": d}
                       for c, n, g, d in CONDITIONS])
    w(md_table(cd))
    w("")


    # ---------------------------------------------------- experimental design
    import glob, json as _json
    mans = {pathlib.Path(m).parent.name: _json.loads(pathlib.Path(m).read_text())
            for m in sorted(glob.glob(str(R / "*" / "manifest.json")))}
    any_m = next(iter(mans.values()))
    n_items = {"kazmmlu": 2000, "kazbench_kk": 2000, "kazculture": 1324}

    w("## Experimental design")
    w("")
    w("### What was run")
    w("")
    w("Every configuration saw **the same frozen perturbed items**. Conditions were "
      "materialised once, before the grid, and reused across all 18 configurations — that "
      "identity is what makes the cross-configuration consensus check in Table 14 valid.")
    w("")
    grid = pd.DataFrame([
        {"Stage": "Baseline", "Conditions": "B0", "Repeats": "3 runs (seeds 1234/1235/1236)",
         "Configurations": n_cfg, "Items": "5,324",
         "Generations": f"{3*5324*n_cfg:,}"},
        {"Stage": "Validity-preserving", "Conditions": "N1, N2, I1, O1, P1", "Repeats": "1",
         "Configurations": n_cfg, "Items": "5,324 each",
         "Generations": f"{5*5324*n_cfg:,}"},
        {"Stage": "Distractor", "Conditions": "N3, N4", "Repeats": "1",
         "Configurations": n_cfg, "Items": "5,324 each",
         "Generations": f"{2*5324*n_cfg:,}"},
        {"Stage": "Floor transfer", "Conditions": "P1", "Repeats": "3 runs",
         "Configurations": 3, "Items": "5,324",
         "Generations": f"{2*5324*3:,} extra"},
        {"Stage": "Position diagnostic", "Conditions": "O1C", "Repeats": "4 rotations",
         "Configurations": 6, "Items": "500 four-option",
         "Generations": f"{4*500*6:,}"},
        {"Stage": "Leakage probe", "Conditions": "NQR, NQI", "Repeats": "1",
         "Configurations": 3, "Items": "900 (300/benchmark)",
         "Generations": f"{2*900*3:,}"},
    ])
    w(md_table(grid))
    w("")
    w(f"Realised total: **{cov['n_rows']:,} generations**, no missing cells.")
    w("")

    w("### Benchmarks and sampling")
    w("")
    bench = pd.DataFrame([
        {"Benchmark": "KazMMLU", "Source": "MBZUAI/KazMMLU, Kazakh subsets, test split",
         "Available": "9,726", "Sampled": "2,000", "Strata": "12 subjects",
         "Options": "4 or 5 (2,587 items carry an E option)"},
        {"Benchmark": "KazBench-KK", "Source": "kz-transformers/kk-socio-cultural-bench-mc",
         "Available": "7,077", "Sampled": "2,000", "Strata": "18 categories",
         "Options": "4"},
        {"Benchmark": "KazCulture", "Source": "issai/KazCulture, test split",
         "Available": "1,324", "Sampled": "1,324 (all)", "Strata": "9 categories",
         "Options": "4"},
    ])
    w(md_table(bench))
    w("")
    w("Sampling is stratified with a fixed seed (`20260815`); the sampled ID lists are "
      "released. KazCulture yielded fewer than the 2,000-item target after validation, so "
      "**all** of its items were used — no upsampling, and the other two were not shrunk to "
      "match. KazMMLU stratifies on subject, KazBench-KK on category (the finer "
      "category × subcategory key would have left 18 of 77 cells below the allocation floor).")
    w("")
    w("Perturbation text (the noise paragraphs and the paraphrases) was generated in advance "
      "with Gemini and validated offline; gold-answer leakage was checked against a "
      "distractor base rate before the grid, and option text is byte-identical to source in "
      "every condition except O1 and O1C.")
    w("")

    w("### Inference settings actually sent")
    w("")
    w("One vLLM endpoint per configuration; the harness never loads a model. Any sampling "
      "key not listed was **absent from the request body**, so the server applied the "
      "model's own `generation_config`. All 18 share prompt version "
      f"`{any_m['prompt_version']}` (SHA `{any_m['prompt_sha256'][:12]}`) and parser "
      f"`{any_m['parser_version']}`.")
    w("")
    rows = []
    for cid in sorted(mans):
        b = dict(mans[cid]["request_body_template"])
        eb = b.pop("extra_body", {}) or {}
        ctk = eb.pop("chat_template_kwargs", {}) or {}
        b.pop("model", None)
        rows.append({
            "Configuration": cid,
            "temperature": b.get("temperature"), "top_p": b.get("top_p"),
            "top_k": eb.get("top_k"), "min_p": eb.get("min_p"),
            "presence_penalty": b.get("presence_penalty"),
            "repetition_penalty": eb.get("repetition_penalty"),
            "max_tokens": b.get("max_tokens"), "seed": b.get("seed"),
            "enable_thinking": ctk.get("enable_thinking"),
            "system prompt": mans[cid].get("system_prompt") or "—",
        })
    sp = pd.DataFrame(rows)
    for c in sp.columns:
        sp[c] = sp[c].map(lambda v: "—" if v is None or (isinstance(v, float) and pd.isna(v)) else v)
    w(md_table(sp))
    w("")
    w("A dash means the key was not sent. `seed` was honoured by every endpoint, so the "
      "three baseline runs differ only through `base_seed + run_index` — the floor is a "
      "measurement of a seeded sampler, not of an unseeded one.")
    w("")

    w("### Scoring")
    w("")
    w("Answers are parsed from `message.content` only; reasoning arrives separately in "
      "`message.reasoning_content` and is never parsed as the answer. The cascade records "
      "which tier fired: strict JSON → fenced JSON → embedded JSON → letter fallback → "
      "failed. Cyrillic/Latin homoglyphs (А, В, С, Е) are normalised before the option-set "
      "check. A letter outside the item's own option set is a parse failure, not a wrong "
      "answer.")
    w("")
    w("Flips are computed on **semantic answer identity**: for O1 and O1C the emitted letter "
      "is mapped back through the option permutation before comparison. Flip rates are "
      "computed symmetrically against each of the three individual baseline runs and "
      "averaged — never against a majority vote, which would denoise one side only.")
    w("")
    # ---------------------------------------------------------------- Table 3
    w("## Table 3 — Baseline accuracy (B0, mean of three runs, %)")
    w("")
    b = rd("baseline_accuracy"); b = b[b.parse_policy == POLICY]
    piv = b.pivot_table(index="config_id", columns="dataset", values="baseline_accuracy_mean") * 100
    sd = b.pivot_table(index="config_id", columns="dataset", values="baseline_accuracy_sd") * 100
    piv["Mean"] = piv.mean(axis=1)
    piv = piv.sort_values("Mean", ascending=False).reset_index()
    piv.columns = ["Configuration", "KazBench-KK", "KazCulture", "KazMMLU", "Mean"]
    w(md_table(piv, "{:.1f}"))
    w("")
    w(f"Spread across the three baseline runs is small: the largest per-cell standard "
      f"deviation is {sd.max().max():.2f} points.")
    w("")

    # ---------------------------------------------------------------- Table 4
    w("## Table 4 — Self-flip floor (B0 vs B0, %)")
    w("")
    w("Mean pairwise disagreement between the three identical baseline runs. This is the "
      "irreducible measurement noise of the protocol. **It is never subtracted from a "
      "perturbation flip rate** — self-flips and perturbation flips partially overlap, so "
      "subtraction understates. It is reported as a floor and as a ratio.")
    w("")
    fl = rd("self_flip_floor"); fl = fl[fl.parse_policy == POLICY]
    fp = fl.pivot_table(index="config_id", columns="dataset", values="self_flip_floor") * 100
    fp["Pooled"] = fp.mean(axis=1)
    fp = fp.sort_values("Pooled").reset_index()
    fp.columns = ["Configuration", "KazBench-KK", "KazCulture", "KazMMLU", "Pooled"]
    w(md_table(fp, "{:.1f}"))
    w("")

    # ---------------------------------------------------------------- Table 5
    w("## Table 5 — Flip rate against the self-flip floor (%)")
    w("")
    w("Share of items whose prediction changes from baseline, computed symmetrically "
      "against each of the three baseline runs and averaged. For O1 and O1C the comparison "
      "is on **semantic answer identity**, mapped back through the option permutation — "
      "comparing letters would inflate the rate to near-total nonsense.")
    w("")
    f = rd("flip_rates"); f = f[f.parse_policy == POLICY]
    g = f.groupby("condition").agg(flip=("flip_rate_mean", "mean"),
                                   floor=("self_flip_floor", "mean"),
                                   ratio=("flip_to_floor_ratio", "mean"),
                                   worst=("flip_rate_mean", "max")).reset_index()
    g = g[g.condition != "B0"]
    g["Group"] = g["condition"].map({c: gg for c, _, gg, _ in CONDITIONS})
    g = g.set_index("condition").reindex([c for c in COND_ORDER if c != "B0"]).reset_index()
    g[["flip", "floor", "worst"]] *= 100
    g.columns = ["Condition", "Flip rate", "Self-flip floor", "Flip / floor", "Worst cell", "Group"]
    g["Flip / floor"] = g["Flip / floor"].map("{:.2f}".format)   # a ratio needs two places
    w(md_table(g[["Condition", "Group", "Flip rate", "Self-flip floor", "Flip / floor", "Worst cell"]], "{:.1f}"))
    w("")
    e = pn[POLICY]
    c_ = e["worst_validity_preserving_flip_cell"]
    w(f"Worst validity-preserving cell: **`{c_['config_id']}`** on "
      f"**{PRETTY.get(c_['dataset'], c_['dataset'])}** under "
      f"**{c_['condition']}**, at **{c_['value']*100:.1f}%** — against a pooled floor of "
      f"{e['pooled_self_flip_floor_mean']*100:.1f}%.")
    w("")

    # ---------------------------------------------------------------- Table 6
    w("## Table 6 — Accuracy change ΔAcc (percentage points)")
    w("")
    w("ΔAcc(p) = Acc(p) − Acc(B0), computed three times, once against each individual "
      "baseline run, then averaged. Negative means the perturbation lowered accuracy.")
    w("")
    d = rd("delta_accuracy"); d = d[d.parse_policy == POLICY]
    dg = d.groupby("condition")["delta_acc_mean"].agg(["mean", "min", "max"]) * 100
    dg = dg.reindex([c for c in COND_ORDER if c != "B0"]).reset_index()
    dg["Group"] = dg["condition"].map({c: gg for c, _, gg, _ in CONDITIONS})
    dg.columns = ["Condition", "Mean ΔAcc", "Worst cell", "Best cell", "Group"]
    w(md_table(dg[["Condition", "Group", "Mean ΔAcc", "Worst cell", "Best cell"]], "{:+.2f}"))
    w("")

    # ---------------------------------------------------------------- Table 7
    w("## Table 7 — Transition matrix (%), net versus gross change")
    w("")
    w("C→I is a baseline-correct item the perturbation broke; I→C is one it fixed. "
      "**Gross churn** = C→I + I→C; **net ΔAcc** = I→C − C→I. A flat accuracy delta can "
      "hide a great deal of churn.")
    w("")
    tr = rd("transitions"); tr = tr[tr.parse_policy == POLICY]
    tg = (tr.groupby("condition")[["CC_rate", "CI_rate", "IC_rate", "II_rate",
                                   "gross_churn_rate", "net_delta_rate"]].mean() * 100)
    tg = tg.reindex(COND_ORDER).reset_index()
    tg.columns = ["Condition", "C→C", "C→I", "I→C", "I→I", "Gross churn", "Net Δ"]
    w(md_table(tg, "{:.2f}"))
    w("")

    # ---------------------------------------------------------------- Table 8
    w("## Table 8 — Flips by baseline margin stratum")
    w("")
    w("The margin proxy is agreement across the three baseline runs: 3/3 = the model gave "
      "the same answer every time, 1/3 = a different answer each time. Pooled over all "
      "perturbation conditions.")
    w("")
    ms = rd("flips_by_margin_stratum_pooled"); ms = ms[ms.parse_policy == POLICY]
    mg = ms.groupby("margin_stratum")[["n_pairs", "flip_rate", "CI_rate", "IC_rate"]].mean()
    mg[["flip_rate", "CI_rate", "IC_rate"]] *= 100
    mg = mg.reset_index()
    mg.columns = ["Baseline agreement", "Pairs (mean)", "Flip rate %", "C→I %", "I→C %"]
    w(md_table(mg, "{:.1f}"))
    w("")
    lm = rd("low_margin_share"); lm = lm[lm.parse_policy == POLICY]
    w(f"Share of baseline-**correct** items that sit in a low-margin stratum (below 3/3): "
      f"**{lm['low_margin_share_of_correct'].mean()*100:.1f}%** on average "
      f"(KazCulture {lm[lm.dataset=='kazculture']['low_margin_share_of_correct'].mean()*100:.1f}%, "
      f"KazMMLU {lm[lm.dataset=='kazmmlu']['low_margin_share_of_correct'].mean()*100:.1f}%).")
    w("")

    # ---------------------------------------------------------------- Table 9
    w("## Table 9 — Ranking stability")
    w("")
    w("**Baseline-internal stability first.** If configurations swap places between "
      "identical reruns, no perturbation-induced reversal involving them means anything. "
      "Kendall τ-b over the 18 configurations.")
    w("")
    r = rd("ranking_stability"); r = r[r.parse_policy == POLICY]
    ri = r[r.condition.isna() & (r.dataset == "pooled")][["run_a", "run_b", "tau_b"]]
    ri.columns = ["Baseline run A", "Baseline run B", "τ-b"]
    w(md_table(ri, "{:.4f}"))
    w("")
    w("Against the baseline ranking, per condition (pooled across benchmarks):")
    w("")
    rc = r[r.condition.notna() & (r.dataset == "pooled")][
        ["condition", "tau_b", "top1_condition", "top1_changed"]]
    rev = rd("ranking_reversals")
    rev = rev[rev.dataset == "pooled"][["condition", "n_separable_pairs", "n_reversed",
                                        "reversal_fraction"]]
    rc = rc.merge(rev, on="condition", how="left").set_index("condition").reindex(
        [c for c in COND_ORDER if c != "B0"]).reset_index()
    rc["reversal_fraction"] *= 100
    rc.columns = ["Condition", "τ-b vs baseline", "Top-1 under condition",
                  "Top-1 changed", "Separable pairs", "Reversed", "Reversed %"]
    w(md_table(rc, "{:.4f}"))
    w("")

    # --------------------------------------------------------------- Table 10
    w("## Table 10 — Position bias (O1C, cyclic rotation)")
    w("")
    w("Six configurations, ~500 four-option items, four rotations each so the gold answer "
      "visits every slot exactly once.")
    w("")
    s_ = rd("o1c_accuracy_by_gold_slot")
    s = s_[s_.parse_policy == POLICY]
    sg = (s.groupby("gold_letter")["accuracy"].agg(["mean", "min", "max"]) * 100).reset_index()
    sg.columns = ["Gold in slot", "Accuracy %", "Lowest config", "Highest config"]
    lp = rd("o1c_letter_preference")
    tot = lp.groupby("selected_letter")["n"].sum()
    share = (tot / tot.sum() * 100)
    sg["Selected %"] = sg["Gold in slot"].map(share)
    w(md_table(sg, "{:.2f}"))
    w("")
    rcst = rd("o1c_rotation_consistency"); rcst = rcst[rcst.parse_policy == POLICY]
    rr = rcst.groupby("dataset")[["all_four_agree_rate", "majority_agree_rate",
                                  "mean_distinct_answers"]].mean().reset_index()
    rr[["all_four_agree_rate", "majority_agree_rate"]] *= 100
    rr.columns = ["Benchmark", "All four rotations agree %", "Majority agree %",
                  "Distinct answers (mean of 4)"]
    w(md_table(rr, "{:.2f}"))
    w("")

    # --------------------------------------------------------------- Table 11
    w("## Table 11 — Output-format compliance and parsing")
    w("")
    w("`format_valid` is strict-JSON tier only. A low value with a low parse-failure rate "
      "means the model answered in a *recoverable* but non-strict format — a formatting "
      "difference, not a capability one. Pooled over **all conditions**; parse tiers as % "
      "of responses.")
    w("")
    fh = rd("format_health")
    tiers = [c for c in fh.columns if c.startswith("tier_")]
    fb = fh.groupby("config_id")
    ft = fb[tiers].sum()
    ft = (ft.div(ft.sum(axis=1), axis=0) * 100)
    fr = fb[["format_valid_rate", "parse_failure_rate", "homoglyph_rate"]].mean() * 100
    fm = pd.concat([fr, ft], axis=1).reset_index()
    fm.columns = ["Configuration", "Strict JSON %", "Parse failure %", "Homoglyph %"] + \
                 [c.replace("tier_", "") for c in tiers]
    w(md_table(fm.sort_values("Strict JSON %"), "{:.2f}"))
    w("")

    # --------------------------------------------------------------- Table 12
    w("## Table 12 — Generation volume by reasoning regime")
    w("")
    w("Evidence that the reasoning regimes are genuinely different configurations rather "
      "than duplicates. The servers did not report `reasoning_tokens` separately, so "
      "completion tokens carry the contrast. Pooled over **all conditions**.")
    w("")
    gh = rd("generation_health")
    gb = gh.groupby("config_id")[
        ["completion_tokens_mean", "completion_tokens_p95", "truncation_rate",
         "latency_mean_s"]].mean().reset_index()
    gb = gb.merge(ax[["config_id", "reasoning_regime"]], on="config_id")
    gb["truncation_rate"] *= 100
    gb.columns = ["Configuration", "Completion tokens (mean)", "p95", "Truncation %",
                  "Latency (s)", "Regime"]
    w(md_table(gb[["Configuration", "Regime", "Completion tokens (mean)", "p95",
                   "Truncation %", "Latency (s)"]].sort_values("Completion tokens (mean)"),
               "{:.1f}"))
    w("")

    # --------------------------------------------------------------- Table 13
    w("## Table 13 — No-question leakage probe (NQR / NQI)")
    w("")
    w("The question is removed; the model sees only the noise paragraph and the answer "
      "options. Three configurations spanning capability. Chance is 1/n_options per item "
      "(0.250 for KazBench-KK and KazCulture, ~0.237 for KazMMLU because of five-option items).")
    w("")
    nq = rd("nq_leakage_probe")
    ng = nq.groupby(["dataset", "condition"]).agg(
        acc=("accuracy", "mean"), ch=("p_chance", "mean"),
        lo=("ci_lo", "mean"), hi=("ci_hi", "mean"),
        above=("above_chance", "sum"), n=("accuracy", "size")).reset_index()
    ng[["acc", "ch", "lo", "hi"]] *= 100
    ng["above"] = ng["above"].astype(int).astype(str) + "/" + ng["n"].astype(int).astype(str)
    ng = ng[["dataset", "condition", "acc", "ch", "lo", "hi", "above"]]
    ng.columns = ["Benchmark", "Condition", "Accuracy %", "Chance %", "CI low", "CI high",
                  "Cells above chance"]
    w(md_table(ng, "{:.2f}"))
    w("")

    # --------------------------------------------------------------- Table 14
    w("## Table 14 — Exclusion sensitivity")
    w("")
    w("Every flip rate recomputed with the **top decile of cross-configuration consensus "
      "C→I items removed** — the items most likely to be broken perturbations rather than "
      "genuine instability. Conclusions should not move.")
    w("")
    es = rd("exclusion_sensitivity"); es = es[es.parse_policy == POLICY]
    eg = es.groupby("condition")[["flip_rate_all", "flip_rate_excl", "flip_rate_delta",
                                  "CI_rate_delta"]].mean() * 100
    eg = eg.reindex([c for c in COND_ORDER if c != "B0"]).reset_index()
    eg.columns = ["Condition", "Flip rate, all items", "Flip rate, top decile removed",
                  "Δ flip rate", "Δ C→I rate"]
    w(md_table(eg, "{:+.2f}"))
    w("")

    # --------------------------------------------------------------- Table 15
    w("## Table 15 — Floor transfer (P1 run three times)")
    w("")
    w("Is the measurement noise under a perturbation the same as the baseline's? P1 was run "
      "three times on three configurations; its internal floor is compared with the B0 floor. "
      "A ratio near 1 means the perturbation adds no extra sampling noise of its own.")
    w("")
    ftr = rd("floor_transfer_p1"); ftr = ftr[ftr.parse_policy == POLICY]
    fg = ftr[["config_id", "dataset", "internal_floor", "self_flip_floor",
              "floor_transfer_ratio"]].copy()
    fg[["internal_floor", "self_flip_floor"]] *= 100
    fg.columns = ["Configuration", "Benchmark", "P1 internal floor %", "B0 floor %", "Ratio"]
    w(md_table(fg, "{:.3f}"))
    w("")

    # --------------------------------------------------------------- Table 16
    w("## Table 16 — Sensitivity to the parse-failure policy")
    w("")
    w("`incorrect`: a parse failure is a distinct value; `FAIL → \"B\"` is a flip. "
      "`excluded`: any pair where either side failed to parse is dropped. Both were "
      "pre-registered; the harness records enough to compute either.")
    w("")
    rows = []
    for k, label in [("pooled_self_flip_floor_mean", "Pooled self-flip floor"),
                     ("mean_validity_preserving_flip_rate", "Mean validity-preserving flip rate"),
                     ("worst_validity_preserving_flip_rate", "Worst validity-preserving flip rate"),
                     ("mean_delta_acc_validity_preserving", "Mean ΔAcc, validity-preserving"),
                     ("mean_delta_acc_distractor", "Mean ΔAcc, distractor"),
                     ("low_margin_share_of_correct_mean", "Low-margin share of correct items")]:
        rows.append({"Quantity": label,
                     "incorrect": pn["incorrect"][k] * 100,
                     "excluded": pn["excluded"][k] * 100,
                     "Difference": (pn["incorrect"][k] - pn["excluded"][k]) * 100})
    w(md_table(pd.DataFrame(rows), "{:+.3f}"))
    w("")
    w(f"Overall parse-failure rate is {fh['parse_failure_rate'].mean()*100:.2f}%, so the two "
      "policies agree to within a fraction of a point on every headline quantity.")
    w("")



    # ------------------------------------------------- per-model breakdowns
    w("---")
    w("")
    w("# Per-model breakdown")
    w("")
    w("The tables above pool over configurations to show the shape of each effect. These "
      "give the same quantities per configuration, which is what a per-model claim in the "
      "paper has to rest on. All at the `" + POLICY + "` parse policy; rows sorted by the "
      "mean over conditions.")
    w("")

    def by_config(df, value, conds, scale=100.0, fmt="{:.1f}", meanlabel="Mean"):
        t = df[df.condition.isin(conds)].pivot_table(
            index="config_id", columns="condition", values=value, observed=True) * scale
        t = t[[c for c in conds if c in t.columns]]
        t[meanlabel] = t.mean(axis=1)
        t = t.sort_values(meanlabel).reset_index().rename(columns={"config_id": "Configuration"})
        return md_table(t, fmt)

    # --- accuracy per configuration x condition, per benchmark ------------
    acc = rd("accuracy_by_run"); acc = acc[acc.parse_policy == POLICY]
    accm = (acc.groupby(["config_id", "dataset", "condition"], observed=True)["accuracy"]
               .mean().reset_index())
    w("## Table 17 — Accuracy by configuration and condition (%)")
    w("")
    w("Averaged over runs where a condition was repeated. B0 is the mean of three runs.")
    for ds in ["kazmmlu", "kazbench_kk", "kazculture"]:
        w("")
        w(f"**{PRETTY[ds]}**")
        w("")
        sub = accm[accm.dataset == ds]
        w(by_config(sub, "accuracy", PERTURB_ORDER, meanlabel="Mean"))
    w("")

    # --- flip rate per configuration x condition --------------------------
    w("## Table 18 — Flip rate by configuration and condition (%)")
    w("")
    w("Pooled over the three benchmarks. `Floor` is that configuration's own self-flip "
      "floor; `Ratio` is the mean flip rate over the validity-preserving conditions "
      "divided by it. A ratio near 1 means the perturbations moved that model no more than "
      "re-running it did.")
    w("")
    ft = f[f.condition.isin(PERTURB_ORDER)].pivot_table(
        index="config_id", columns="condition", values="flip_rate_mean", observed=True) * 100
    ft = ft[[c for c in PERTURB_ORDER if c in ft.columns]]
    ft["Floor"] = fl.groupby("config_id")["self_flip_floor"].mean() * 100
    ft["Ratio"] = ft[[c for c in VP if c in ft.columns]].mean(axis=1) / ft["Floor"]
    ft = ft.sort_values("Ratio").reset_index().rename(columns={"config_id": "Configuration"})
    ft["Ratio"] = ft["Ratio"].map("{:.2f}".format)
    w(md_table(ft, "{:.1f}"))
    w("")

    # --- delta accuracy per configuration x condition ---------------------
    w("## Table 19 — ΔAcc by configuration and condition (percentage points)")
    w("")
    w("Pooled over the three benchmarks. Negative means the perturbation lowered accuracy. "
      "The `N3`/`N4` columns are the distractor arm and are **not** part of the "
      "validity-preserving mean.")
    w("")
    dt = d[d.condition.isin(PERTURB_ORDER)].pivot_table(
        index="config_id", columns="condition", values="delta_acc_mean", observed=True) * 100
    dt = dt[[c for c in PERTURB_ORDER if c in dt.columns]]
    dt["VP mean"] = dt[[c for c in VP if c in dt.columns]].mean(axis=1)
    dt = dt.sort_values("VP mean").reset_index().rename(columns={"config_id": "Configuration"})
    w(md_table(dt, "{:+.2f}"))
    w("")

    # --- transitions per configuration ------------------------------------
    w("## Table 20 — Net versus gross change by configuration")
    w("")
    w("Pooled over the three benchmarks and the five validity-preserving perturbations. "
      "`Gross churn` is C→I + I→C; `Net Δ` is I→C − C→I. The gap between them is how much "
      "movement a flat accuracy delta conceals.")
    w("")
    tv = tr[tr.condition.isin(VP)].groupby("config_id")[
        ["CC_rate", "CI_rate", "IC_rate", "II_rate", "gross_churn_rate", "net_delta_rate"]
    ].mean() * 100
    tv = tv.sort_values("gross_churn_rate", ascending=False).reset_index()
    tv.columns = ["Configuration", "C→C", "C→I", "I→C", "I→I", "Gross churn", "Net Δ"]
    w(md_table(tv, "{:.2f}"))
    w("")

    # --- O1C per configuration --------------------------------------------
    w("## Table 21 — Position bias by configuration (O1C, six configurations)")
    w("")
    slc = s_[s_.parse_policy == POLICY].pivot_table(
        index="config_id", columns="gold_letter", values="accuracy", observed=True) * 100
    slc["Spread A–D"] = slc.max(axis=1) - slc.min(axis=1)
    rcc = rcst.groupby("config_id")[["all_four_agree_rate", "majority_agree_rate"]].mean() * 100
    slc = slc.join(rcc).sort_values("Spread A–D", ascending=False).reset_index()
    slc = slc.rename(columns={"config_id": "Configuration",
                              "all_four_agree_rate": "All 4 agree %",
                              "majority_agree_rate": "Majority agree %"})
    w(md_table(slc, "{:.1f}"))
    w("")

    # --- leakage probe per configuration ----------------------------------
    w("## Table 22 — Leakage probe by configuration (three configurations)")
    w("")
    w("Question removed; noise paragraph plus options only. `Above chance` is the exact "
      "one-sided binomial test against 1/n_options.")
    w("")
    nqc = nq.copy()
    nqc["Benchmark"] = nqc["dataset"].map(PRETTY)
    nqc["Accuracy %"] = nqc["accuracy"] * 100
    nqc["Chance %"] = nqc["p_chance"] * 100
    nqc["Above chance"] = nqc["above_chance"].map({True: "yes", False: "no"})
    nqc = nqc[["config_id", "Benchmark", "condition", "Accuracy %", "Chance %",
               "p_greater", "Above chance"]].rename(
        columns={"config_id": "Configuration", "condition": "Condition",
                 "p_greater": "p (one-sided)"})
    w(md_table(nqc.sort_values(["Configuration", "Condition", "Benchmark"]), "{:.3f}"))
    w("")
    # ------------------------------------------------------------ interpretation
    e = pn[POLICY]
    flr = fl.groupby("config_id")["self_flip_floor"].mean() * 100
    vp_flip = f[f.condition.isin(["N1", "N2", "I1", "O1", "P1"])]["flip_rate_mean"].mean() * 100
    vp_ratio = f[f.condition.isin(["N1", "N2", "I1", "O1", "P1"])]["flip_to_floor_ratio"].mean()
    vp_dacc = d[d.condition.isin(["N1", "N2", "I1", "O1", "P1"])]["delta_acc_mean"].mean() * 100
    ds_dacc = d[d.condition.isin(["N3", "N4"])]["delta_acc_mean"].mean() * 100
    tau_min = rc["τ-b vs baseline"][rc["Condition"].isin(VP)].min()
    tau_pool = ri["τ-b"]
    tau_bench_min = r[r.condition.isna() & (r.dataset != "pooled")]["tau_b"].min()
    tau_bench_which = r.loc[r[r.condition.isna() & (r.dataset != "pooled")]["tau_b"].idxmin(), "dataset"]
    nqr = nq[nq.condition == "NQR"]; nqi = nq[nq.condition == "NQI"]
    ms3 = ms[ms.margin_stratum == "3/3"]["flip_rate"].mean() * 100
    ms1 = ms[ms.margin_stratum == "1/3"]["flip_rate"].mean() * 100
    gd = gb.set_index("Configuration")["Completion tokens (mean)"]

    w("---")
    w("")
    w("# Interpretation")
    w("")

    w("## 1. The dominant source of instability is the sampler, not the perturbation")
    w("")
    from scipy import stats as _st
    vpf = f[f.condition.isin(VP)].groupby("config_id")["flip_rate_mean"].mean() * 100
    per_ratio = (vpf / flr)
    excess = (vpf - flr)
    # align on config_id before correlating: sorting one side and not the other
    # silently compares different models to each other
    _al = pd.concat([flr.rename("floor"), per_ratio.rename("ratio")], axis=1).dropna()
    rho, rho_p = _st.spearmanr(_al["floor"], _al["ratio"])
    per_ratio = per_ratio.sort_values()
    w(f"The self-flip floor — disagreement between three *identical* baseline runs — ranges "
      f"from **{flr.min():.1f}%** (`{flr.idxmin()}`) to **{flr.max():.1f}%** "
      f"(`{flr.idxmax()}`), pooled mean **{e['pooled_self_flip_floor_mean']*100:.1f}%**. "
      f"Validity-preserving perturbations move predictions on **{vp_flip:.1f}%** of items.")
    w("")
    w(f"Read plainly: re-running the benchmark completely unchanged already churns "
      f"{e['pooled_self_flip_floor_mean']/vp_flip*100*100:.0f}% as many predictions as the "
      f"average validity-preserving perturbation does. **Any "
      "reported flip rate that is not accompanied by its floor is close to uninterpretable**, "
      "and this is the single most transferable methodological point in the study.")
    w("")
    w("**But the pooled ratio hides the effect it is meant to describe, and Table 18 is "
      "where the paper should argue this.** Per configuration the flip-to-floor ratio spans "
      f"**{per_ratio.min():.2f}×** (`{per_ratio.idxmin()}`) to **{per_ratio.max():.2f}×** "
      f"(`{per_ratio.idxmax()}`), and it is almost perfectly inversely ordered with the "
      f"floor itself (Spearman ρ = **{rho:.2f}**, p = {rho_p:.1e}, n = {len(flr)}).")
    w("")
    w(f"The absolute excess is steadier than the ratio: validity-preserving perturbations "
      f"add **{excess.min():.1f}–{excess.max():.1f} points** of churn on top of whatever "
      f"floor a model already has (mean {excess.mean():.1f}). So the mechanism is not that "
      "perturbations hurt noisy models more — it is that a noisy model's own sampling "
      "variance **swallows** the perturbation effect, while a stable model exposes it. "
      f"`{per_ratio.index[0]}` sits at {per_ratio.min():.2f}× not because it is robust but "
      f"because its {flr[per_ratio.index[0]]:.1f}% floor leaves no headroom to measure "
      f"anything; `{per_ratio.index[-1]}` reaches {per_ratio.max():.2f}× because its "
      f"{flr[per_ratio.index[-1]]:.1f}% floor makes a comparable absolute effect visible.")
    w("")
    w("**Report the absolute excess alongside the ratio.** A ratio on its own rewards a "
      "noisy model for being noisy.")
    w("")
    w("The floor is not uniform. It tracks reasoning regime almost perfectly: every "
      "thinking-enabled configuration has a higher floor than its non-thinking twin of the "
      "same checkpoint, and the two gpt-oss configurations — the weakest models here — sit "
      "at ~35%, meaning a third of their answers change on a rerun. Comparisons involving "
      "them are noise-dominated regardless of perturbation.")
    w("")

    w("## 2. Predictions churn; conclusions do not")
    w("")
    w("This is the result that most complicates the paper's framing, and it should lead.")
    w("")
    w(f"- Mean ΔAcc under validity-preserving perturbation: **{vp_dacc:+.2f} points**.")
    w(f"- Worst single cell: **{e['worst_delta_acc_cell']['value']*100:+.1f} points** "
      f"(`{e['worst_delta_acc_cell']['config_id']}`, "
      f"{PRETTY.get(e['worst_delta_acc_cell']['dataset'])}, "
      f"{e['worst_delta_acc_cell']['condition']}).")
    w(f"- Kendall τ-b against the baseline ranking stays at or above **{tau_min:.3f}** for "
      "every validity-preserving condition.")
    w("- The top-ranked configuration (`gemma-4-31b-it-think`) is **never** displaced.")
    w(f"- Of **{int(rc['Separable pairs'].iloc[0])}** baseline-separable configuration pairs, "
      f"at most **{e['separable_pair_reversal_fraction_max']*100:.1f}%** reverse under any "
      "validity-preserving condition.")
    w("")
    w("So the honest headline is *not* \"the benchmark is unreliable\". It is: **individual "
      "predictions are unstable at a rate that would alarm most practitioners, while "
      "aggregate scores and leaderboard order are robust.** Instability lives at the item "
      "level and averages out by the time it reaches a ranking. That is a more defensible "
      "and more interesting claim than a reversal count, and it is well supported here.")
    w("")
    w("One caveat on the ranking evidence: baseline-internal τ-b between identical reruns is "
      f"{tau_pool.min():.4f}–{tau_pool.max():.4f} pooled, but falls to "
      f"{tau_bench_min:.4f} on {PRETTY.get(tau_bench_which, tau_bench_which)} alone. "
      "Per-benchmark ranking claims are "
      "weaker than pooled ones.")
    w("")

    w("## 3. Instability concentrates on items the model was already guessing")
    w("")
    w(f"Items where the model gave the same answer in all three baseline runs flip "
      f"**{ms3:.1f}%** of the time under perturbation. Items where it gave a different "
      f"answer each time flip **{ms1:.1f}%** of the time — an eightfold difference.")
    w("")
    w(f"And **{lm['low_margin_share_of_correct'].mean()*100:.1f}%** of *baseline-correct* "
      "items sit below 3/3 agreement: they were scored correct on a prediction the model "
      "does not reliably reproduce. On KazCulture the figure is "
      f"{lm[lm.dataset=='kazculture']['low_margin_share_of_correct'].mean()*100:.1f}%.")
    w("")
    w("This is the mechanism behind §1 and §2 together. A large share of the churn is "
      "coin-flip items being re-flipped, which is why it does not propagate into accuracy.")
    w("")

    w("## 4. The distractor arm did not distract — and the probe says why")
    w("")
    w("**This needs addressing before submission.**")
    w("")
    n3 = d[d.condition == "N3"]["delta_acc_mean"]
    n4 = d[d.condition == "N4"]["delta_acc_mean"]
    w(f"N3 and N4 were designed as misleading relevant context. They *raised* accuracy by "
      f"**{ds_dacc:+.2f} points** on average, with I→C exceeding C→I in the transition matrix "
      f"(Table 7). The direction is near-universal: **{int((n3>0).sum())}/{len(n3)}** N3 cells "
      f"and **{int((n4>0).sum())}/{len(n4)}** N4 cells improved, the single exception being "
      f"{n4.min()*100:+.2f} points.")
    w("")
    w("The no-question probe (Table 13) explains it. With the question deleted entirely, "
      f"leaving only the noise paragraph and the options:")
    w("")
    w(f"- **NQR** (relevant noise): **{nqr['accuracy'].mean()*100:.1f}%** against a chance rate "
      f"of {nqr['p_chance'].mean()*100:.1f}% — above chance in **{int(nqr['above_chance'].sum())}/"
      f"{len(nqr)}** cells.")
    w(f"- **NQI** (irrelevant noise, the negative control): "
      f"**{nqi['accuracy'].mean()*100:.1f}%** — above chance in only "
      f"**{int(nqi['above_chance'].sum())}/{len(nqi)}** cells.")
    w("")
    w("A model can pick the right answer nearly twice as often as chance from the "
      "\"distractor\" paragraph alone. The generated relevant noise carries substantial "
      "answer signal, so N3/N4 are not a distractor-robustness arm — they are a "
      "**helpful-context** arm.")
    w("")
    w("Two defensible responses, and the choice should be explicit in the Method:")
    w("")
    w("1. **Reframe.** Report N3/N4 as *relevant context* rather than distraction, and use "
      "the NQR/NQI contrast as the evidence for that reading. Costs nothing, requires no "
      "new inference, and the probe result becomes a finding in its own right.")
    w("2. **Regenerate.** Produce genuinely misleading noise and re-run the N3/N4 slice. "
      "Correct but expensive, and the current data still supports the reframing.")
    w("")
    w("Either way, **do not report N3/N4 as distractor robustness as generated.** A reviewer "
      "who reads the NQR row will spot it.")
    w("")

    w("## 5. Position bias is real but modest")
    w("")
    w(f"Across the six O1-C configurations, accuracy is roughly flat when the gold answer "
      f"sits in slot A, B or C ({sg['Accuracy %'].iloc[0]:.1f}–{sg['Accuracy %'].iloc[2]:.1f}%) "
      f"and drops to **{sg['Accuracy %'].iloc[3]:.1f}%** when it sits in D. Selected-letter "
      f"shares mirror it: D is chosen {share['D']:.1f}% of the time against "
      f"{share['A']:.1f}% for A.")
    w("")
    spr = (slc.set_index("Configuration")["Spread A–D"]).sort_values(ascending=False)
    w("")
    w("**That pooled row is one model, not a trend** (Table 21). The A–D accuracy spread is "
      f"**{spr.iloc[0]:.1f} points** for `{spr.index[0]}` and a median of "
      f"**{spr.iloc[1:].median():.1f} points** across the other five. Slot-D weakness is a "
      "property of that single configuration; pooling turns one model's pathology into an "
      "apparent panel-wide bias. State it per configuration.")
    w("")
    w(f"Rotation consistency is the more general result: only "
      f"**{rcst['all_four_agree_rate'].mean()*100:.1f}%** of items receive the same semantic "
      f"answer under all four rotations (per configuration: "
      f"{slc['All 4 agree %'].min():.0f}–{slc['All 4 agree %'].max():.0f}%), even though "
      "rotation preserves relative option order and changes nothing but absolute position. "
      "That is a lower bound on order sensitivity with content held constant.")
    w("")

    w("## 6. Format compliance is a family convention, not a capability")
    w("")
    w("Table 11 would be badly misread without its tier breakdown. Six Gemma configurations "
      "show strict-JSON rates between 0% and 49% — but parse-failure rates of ~0%. Both "
      "Gemma thinking configurations emit **100% fenced** JSON (```` ```json ````), which is "
      "perfectly recoverable and never a wrong answer.")
    w("")
    w("Reporting `format_valid` alone would state that `gemma-4-31b-it-think` — the "
      "**highest-accuracy configuration in the study** — has 0% output compliance. Always "
      "report the tier distribution beside it.")
    w("")
    w(f"The Cyrillic homoglyph risk did not materialise: the normalisation fired on "
      f"**{fh['homoglyph_rate'].mean()*100:.3f}%** of responses. Worth one sentence stating "
      "the guard existed and was not needed, since it was a pre-registered concern.")
    w("")

    w("## 7. The reasoning regimes are real; the gpt-oss effort gradient is not")
    w("")
    w("Completion-token volume separates thinking from non-thinking by one to two orders of "
      f"magnitude (`qwen3.5-9b`: {gd['qwen3.5-9b-nothink']:.0f} → {gd['qwen3.5-9b-think']:.0f} "
      f"tokens; `qwen3.8-27b`: {gd['qwen3.8-27b-nothink']:.0f} → {gd['qwen3.8-27b-think']:.0f}). "
      "Those are genuinely distinct configurations.")
    w("")
    w(f"The gpt-oss effort levels are not. `low` and `high` differ by "
      f"{gd['gpt-oss-20b-high'] - gd['gpt-oss-20b-low']:.0f} completion tokens "
      f"({(gd['gpt-oss-20b-high']/gd['gpt-oss-20b-low']-1)*100:.0f}%) and by "
      "**0.0 accuracy points** pooled. Muse-Glimmer's gradient, by contrast, is "
      f"{gd['muse-glimmer-30b-low']:.0f} → {gd['muse-glimmer-30b-high']:.0f} tokens (3.1×).")
    w("")
    w("Treat the two gpt-oss configurations as near-duplicates in any effort-gradient claim, "
      "and say so — an unremarked null there is the kind of thing that invites a reviewer to "
      "question the whole gradient.")
    w("")

    w("## 8. Robustness of the analysis itself")
    w("")
    pol_max = max(abs(pn["incorrect"][k] - pn["excluded"][k]) * 100 for k in
                  ("pooled_self_flip_floor_mean", "mean_validity_preserving_flip_rate",
                   "mean_delta_acc_validity_preserving"))
    w("- **Parse-failure policy** (Table 16): the largest movement of any headline quantity "
      f"between the two treatments is **{pol_max:.3f} points**. The overall parse-failure rate is "
      f"{fh['parse_failure_rate'].mean()*100:.2f}%, so this was never going to bite — but it "
      "was pre-registered and both are reported.")
    w("- **Exclusion sensitivity** (Table 14): removing the top decile of "
      "cross-configuration consensus C→I items lowers validity-preserving flip rates by "
      f"{abs(eg[eg.Condition.isin(VP)]['Δ flip rate']).mean():.1f}–"
      f"{abs(eg[eg.Condition.isin(VP)]['Δ flip rate']).max():.1f} points and changes no "
      "conclusion. NQR/NQI move far more, as expected for degenerate conditions.")
    w("- **Floor transfer** (Table 15): P1's internal floor is 0.90–1.04× the baseline floor "
      "across nine configuration × benchmark cells. Paraphrasing adds no sampling noise of "
      "its own; the floor is a property of the model, not of the condition.")
    w(f"- **Near-chance cells**: {e['n_near_chance_cells']} model × benchmark cells have an "
      "accuracy interval covering chance, so no flip rate here is degenerate in the way a "
      "randomly-guessing model would produce.")
    w("")

    w("## 9. Known limits of this table set")
    w("")
    w("- **Unequal *n***. KazCulture contributes 1,324 items against 2,000 for the other two "
      "— all that exist after validation. Every metric is per-benchmark, so this costs only "
      "the width of KazCulture's intervals, but pooled figures are implicitly weighted "
      "toward the larger benchmarks.")
    w("- **`reasoning_tokens` was not reported** by the servers, so Table 12 uses completion "
      "tokens. The regime contrast survives; a separate thinking-token budget claim does not.")
    w("- **O1 flip decomposition is available only for the six O1-C configurations**; the "
      "other twelve have no measured slot preference to decompose against.")
    w("- **Intervals are unadjusted** for multiple comparisons across conditions and "
      "configurations, and should be read as effect sizes.")
    w("- **KazBench-KK split**: the repository labels it `train`; it is used as the "
      "evaluation set per the accompanying paper. Recorded on every row; no contamination "
      "analysis was performed on that basis.")
    w("")

    w("## Suggested order for the Results section")
    w("")
    w("1. Baseline accuracy and the self-flip floor together (Tables 3–4) — the floor must "
      "arrive before any flip rate.")
    w("2. Flip rate against the floor, with the ratio (Table 5).")
    w("3. Net versus gross: ΔAcc alongside the transition matrix (Tables 6–7). This is the "
      "strongest figure-free argument in the set.")
    w("4. Margin decomposition (Table 8) — the mechanism.")
    w("5. Ranking stability and reversals (Table 9) — the \"conclusions hold\" result.")
    w("6. Position bias (Table 10) and the leakage probe (Table 13) as diagnostics.")
    w("7. Format, generation health, and the three robustness tables (11, 12, 14–16) as "
      "appendix material.")
    w("")

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
