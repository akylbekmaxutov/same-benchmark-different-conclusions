"""The paired join: row counts, refusal on mismatched manifests, and the floor
reproduced through the same code path as any other flip rate."""
import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import mk_long
from stability.aggregate import (AggregationRefused, build_paired,
                                 check_manifest_compatibility, load_long)
from stability.io import write_json, write_jsonl
from stability.metrics import core, floor

CONDS = ["N1", "N2", "P1"]
N_ITEMS = 5


def _spec():
    spec = {"B0": {r: {f"u{i}": ("A" if (i + r) % 3 else "B") for i in range(N_ITEMS)}
                   for r in range(3)}}
    for c in CONDS:
        spec[c] = {0: {f"u{i}": ("A" if i % 2 else "C") for i in range(N_ITEMS)}}
    return spec


def _write_config(root: Path, config_id: str, long: pd.DataFrame, manifest_extra=None):
    raw = root / config_id / "raw"
    for (ds, cond, run), g in long.groupby(["dataset", "condition", "run_index"]):
        rows = g.drop(columns=["answer_eff"]).to_dict("records")
        write_jsonl(raw / f"{ds}__{cond}__run{run}.jsonl", rows)
    m = {"prompt_version": "v1", "prompt_sha256": "abc", "parser_version": "parse/v1",
         "shuffle_seed": 1, "sample_index_sha256": "s",
         "manifest_hash": "h", "repeats": {"B0": 3, "default": 1},
         "item_counts": {f"d__{c}": N_ITEMS for c in ["B0"] + CONDS}}
    m.update(manifest_extra or {})
    write_json(root / config_id / "manifest.json", m)


def test_paired_join_row_counts():
    paired = build_paired(mk_long(_spec()))
    for cond in CONDS:
        n = len(paired[paired["condition"] == cond])
        assert n == N_ITEMS * 3, f"{cond}: expected n_items x 3 baseline runs"
    # B0 pairs each run against the two OTHER baseline runs
    assert len(paired[paired["condition"] == "B0"]) == N_ITEMS * 3 * 2
    assert len(paired) == N_ITEMS * (len(CONDS) * 3 + 6)


def test_b0_pairs_exist_and_reproduce_the_floor_through_the_same_path():
    long = mk_long(_spec())
    paired = build_paired(long)
    b0 = paired[paired["condition"] == "B0"]
    assert len(b0) > 0

    via_floor = floor.self_flip_floor(paired, policies=("incorrect",))["self_flip_floor"].iloc[0]
    via_generic = core.flip_rates(b0, by=("config_id", "dataset"),
                                  policies=("incorrect",))["flip_rate_mean"].iloc[0]
    assert via_floor == pytest.approx(via_generic)


def test_semantic_mapping_happens_once_in_the_join():
    perm = {"A": "B", "B": "C", "C": "D", "D": "A"}
    spec = _spec()
    spec["O1"] = {0: {f"u{i}": "B" for i in range(N_ITEMS)}}
    paired = build_paired(mk_long(spec, perm_by_condition={"O1": perm}))
    o1 = paired[paired["condition"] == "O1"]
    assert set(o1["perturbed_answer"]) == {"B"}
    assert set(o1["semantic_perturbed"]) == {"A"}


def test_refuses_mismatched_manifests():
    ok = {"a": {"prompt_version": "v1", "prompt_sha256": "x", "parser_version": "p",
                "shuffle_seed": 1, "sample_index_sha256": "s"}}
    check_manifest_compatibility(ok)
    bad = dict(ok)
    bad["b"] = {**ok["a"], "prompt_sha256": "y"}
    with pytest.raises(AggregationRefused, match="manifest mismatch"):
        check_manifest_compatibility(bad)


def test_load_long_refuses_duplicate_keys(tmp_path):
    long = mk_long(_spec())
    _write_config(tmp_path, "c1", long)
    dup = long[long["condition"] == "B0"].head(1).drop(columns=["answer_eff"])
    with open(tmp_path / "c1" / "raw" / "d__B0__run0.jsonl", "a") as fh:
        fh.write(json.dumps(dup.to_dict("records")[0], default=str) + "\n")
    with pytest.raises(AggregationRefused, match="duplicate"):
        load_long(tmp_path)


def test_load_long_refuses_mismatched_manifests_on_disk(tmp_path):
    long = mk_long(_spec())
    _write_config(tmp_path, "c1", long)
    _write_config(tmp_path, "c2", mk_long(_spec(), config_id="c2"),
                  manifest_extra={"prompt_sha256": "DIFFERENT"})
    with pytest.raises(AggregationRefused, match="manifest mismatch"):
        load_long(tmp_path)


def test_load_long_refuses_missing_manifest(tmp_path):
    long = mk_long(_spec())
    _write_config(tmp_path, "c1", long)
    (tmp_path / "c1" / "manifest.json").unlink()
    with pytest.raises(AggregationRefused, match="without a manifest"):
        load_long(tmp_path)


def test_load_long_round_trips(tmp_path):
    long = mk_long(_spec())
    _write_config(tmp_path, "c1", long)
    _write_config(tmp_path, "c2", mk_long(_spec(), config_id="c2"))
    got = load_long(tmp_path)
    assert len(got) == 2 * len(long)
    assert set(got["config_id"]) == {"c1", "c2"}
    assert build_paired(got)["config_id"].nunique() == 2
