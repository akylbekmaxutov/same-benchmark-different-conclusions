"""The paired join.  Every analysis in §12 operates on the output of this
module; no metric function ever recomputes a pairing.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .io import read_json, read_jsonl
from .parse import FAIL
from .runner import parse_shard_name
from .schema import RESULT_ROW_FIELDS

BASELINE = "B0"

#: manifest fields that must agree across every configuration before pooling
MANIFEST_COMPAT = ("prompt_version", "prompt_sha256", "parser_version",
                   "shuffle_seed", "sample_index_sha256")


class AggregationRefused(RuntimeError):
    pass


def discover_configs(results_dir: str | Path) -> list[Path]:
    root = Path(results_dir)
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "raw").is_dir() and p.name not in
                  ("aggregate", "metrics", "figures", "validation"))


def load_manifests(config_dirs: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    missing = []
    for d in config_dirs:
        m = d / "manifest.json"
        if not m.exists():
            missing.append(d.name)
            continue
        out[d.name] = read_json(m)
    if missing:
        raise AggregationRefused(f"configurations without a manifest: {missing}")
    return out


def check_manifest_compatibility(manifests: dict[str, dict]) -> None:
    """Refuse to aggregate across mismatched manifests, printing which differ."""
    if not manifests:
        raise AggregationRefused("no manifests found")
    ref_id = sorted(manifests)[0]
    ref = manifests[ref_id]
    bad: dict[str, dict] = {}
    for cid, m in manifests.items():
        diff = {k: (ref.get(k), m.get(k)) for k in MANIFEST_COMPAT if ref.get(k) != m.get(k)}
        if diff:
            bad[cid] = diff
    if bad:
        lines = [f"  {cid}: " + ", ".join(f"{k}: {a!r} != {b!r}" for k, (a, b) in d.items())
                 for cid, d in bad.items()]
        raise AggregationRefused(
            f"manifest mismatch against reference {ref_id!r}; pooling across these is "
            "silent corruption:\n" + "\n".join(lines))


def load_long(results_dir: str | Path) -> pd.DataFrame:
    dirs = discover_configs(results_dir)
    if not dirs:
        raise AggregationRefused(f"no configuration directories under {results_dir}")
    manifests = load_manifests(dirs)
    check_manifest_compatibility(manifests)

    frames = []
    for d in dirs:
        rows = []
        for shard in sorted((d / "raw").glob("*.jsonl")):
            if parse_shard_name(shard) is None:
                continue
            rows.extend(read_jsonl(shard))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for col in RESULT_ROW_FIELDS:
            if col not in df.columns:
                df[col] = None
        frames.append(df[list(RESULT_ROW_FIELDS)])
    if not frames:
        raise AggregationRefused("no result rows found")
    long = pd.concat(frames, ignore_index=True)

    dup = long.duplicated(subset=["config_id", "uid", "condition", "run_index"])
    if dup.any():
        ex = long.loc[dup, ["config_id", "uid", "condition", "run_index"]].head(10)
        raise AggregationRefused(
            f"{int(dup.sum())} duplicate (config_id, uid, condition, run_index) keys, e.g.\n{ex}")

    long["answer_eff"] = long["parsed_answer"].fillna(FAIL)
    return long


def expected_counts(work_dir: str | Path, manifests: dict[str, dict]) -> dict:
    """Expected rows per (config_id, dataset, condition, run_index) from the
    materialized condition files and the repeat spec in each manifest."""
    work = Path(work_dir)
    sizes = {}
    for p in sorted((work / "conditions").glob("*__*.jsonl")):
        ds, cond = p.stem.split("__")
        sizes[(ds, cond)] = sum(1 for _ in read_jsonl(p))
    exp: dict = {}
    for cid, m in manifests.items():
        reps = m.get("repeats") or {"default": 1}
        for key, n in (m.get("item_counts") or {}).items():
            ds, cond = key.split("__")
            exp[(cid, ds, cond)] = n * int(reps.get(cond, reps.get("default", 1)))
    return exp


def coverage_report(long: pd.DataFrame, manifests: dict[str, dict],
                    work_dir: str | Path, datasets=("kazmmlu", "kazculture", "kazbench_kk")) -> dict:
    exp = expected_counts(work_dir, manifests)
    got = long.groupby(["config_id", "dataset", "condition"]).size().to_dict()
    shortfalls = []
    for key, want in sorted(exp.items()):
        have = got.get(key, 0)
        if have != want:
            shortfalls.append({"config_id": key[0], "dataset": key[1], "condition": key[2],
                               "expected": want, "observed": have, "delta": have - want})
    missing_ds = []
    for cid in sorted(long["config_id"].unique()):
        seen = set(long.loc[long["config_id"] == cid, "dataset"].unique())
        gap = set(datasets) - seen
        if gap:
            missing_ds.append({"config_id": cid, "missing_datasets": sorted(gap)})
    return {"shortfalls": shortfalls, "configs_missing_datasets": missing_ds,
            "n_configs": int(long["config_id"].nunique()),
            "n_rows": int(len(long))}


def _inverse(perm) -> dict[str, str] | None:
    if not isinstance(perm, dict) or not perm:
        return None
    return {v: k for k, v in perm.items()}


def semantic(answer: str, perm) -> str:
    """Map an emitted letter back through `option_permutation` to the original
    option space.  Happens here, once; no metric function touches the
    permutation again."""
    if answer == FAIL:
        return FAIL
    inv = _inverse(perm)
    if inv is None:
        return answer
    return inv.get(answer, FAIL)


def margin_strata(long: pd.DataFrame) -> pd.DataFrame:
    """Run-agreement margin proxy: agreement across the baseline runs.
    3/3, 2/3, 1/3 -> strata.  The primary margin method; logprobs corroborate."""
    b0 = long[long["condition"] == BASELINE]
    rows = []
    for (cid, puid), g in b0.groupby(["config_id", "parent_uid"], sort=False):
        answers = list(g["answer_eff"])
        n = len(answers)
        top, cnt = Counter(answers).most_common(1)[0]
        rows.append({"config_id": cid, "parent_uid": puid,
                     "baseline_n_runs": n, "baseline_modal_answer": top,
                     "baseline_agreement": cnt / n if n else float("nan"),
                     "margin_stratum": f"{cnt}/{n}"})
    return pd.DataFrame(rows)


def build_paired(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (config_id, parent_uid, condition, run_index, baseline_run_index).

    Perturbation rows are joined against EACH baseline run separately -- never
    against a majority vote, which would denoise one side only.  B0-vs-B0 pairs
    are generated by the same code path, so the self-flip floor is the same
    computation as any other flip rate.
    """
    base = long[long["condition"] == BASELINE][
        ["config_id", "parent_uid", "run_index", "answer_eff", "is_correct",
         "parse_tier", "gold_answer_original"]
    ].rename(columns={"run_index": "baseline_run_index",
                      "answer_eff": "baseline_answer",
                      "is_correct": "baseline_correct",
                      "parse_tier": "baseline_parse_tier",
                      "gold_answer_original": "baseline_gold"})

    keep = ["config_id", "family", "params_total_b", "params_active_b", "is_moe",
            "reasoning_regime", "kazakh_specialized", "dataset", "category",
            "subcategory", "parent_uid", "uid", "condition", "condition_group",
            "run_index", "rotation_index", "n_options", "gold_answer",
            "gold_answer_original", "gold_position_index", "option_permutation",
            "answer_eff", "is_correct", "parse_tier", "format_valid",
            "homoglyph_normalized", "truncated", "noise_type", "noise_position"]
    pert = long[keep].rename(columns={"answer_eff": "perturbed_answer",
                                      "is_correct": "perturbed_correct",
                                      "parse_tier": "perturbed_parse_tier",
                                      "run_index": "perturbed_run_index"})

    paired = pert.merge(base, on=["config_id", "parent_uid"], how="inner")
    # a baseline row is never paired with itself
    same = (paired["condition"] == BASELINE) & \
           (paired["perturbed_run_index"] == paired["baseline_run_index"])
    paired = paired[~same].reset_index(drop=True)

    paired["semantic_baseline"] = paired["baseline_answer"]
    paired["semantic_perturbed"] = [
        semantic(a, p) for a, p in zip(paired["perturbed_answer"], paired["option_permutation"])
    ]

    paired["pair_has_fail"] = ((paired["semantic_baseline"] == FAIL) |
                               (paired["semantic_perturbed"] == FAIL))
    paired["flipped"] = paired["semantic_perturbed"] != paired["semantic_baseline"]
    paired["transition"] = [
        ("C" if b else "I") + ("C" if p else "I")
        for b, p in zip(paired["baseline_correct"].astype(bool),
                        paired["perturbed_correct"].astype(bool))
    ]

    strata = margin_strata(long)
    paired = paired.merge(strata, on=["config_id", "parent_uid"], how="left")
    return paired
