"""Sample indices are released with the paper: they must be regenerable
byte-identically, and the small-benchmark rule must not quietly change what
`~2,000 per benchmark` means."""
import orjson
import pytest

from conftest import mk_item
from stability.subsample import (allocate, build_o1c_pool, choose_key,
                                 stratified_sample)


def items(n, dataset="ds", cats=("a", "b", "c"), subcats=None, n_options=4):
    out = []
    for i in range(n):
        cat = cats[i % len(cats)]
        sub = None if subcats is None else subcats[i % len(subcats)]
        out.append(mk_item(uid=f"{dataset}::{i:06d}", category=cat, subcategory=sub,
                           dataset=dataset, n_options=n_options,
                           answer="A"))
    return out


def test_allocation_sums_to_target_and_respects_capacity():
    sizes = {("a",): 100, ("b",): 10, ("c",): 1}
    alloc = allocate(sizes, 50)
    assert sum(alloc.values()) == 50
    assert all(alloc[k] <= sizes[k] for k in sizes)
    assert all(v >= 1 for v in alloc.values())


def test_allocation_returns_everything_when_target_exceeds_supply():
    sizes = {("a",): 3, ("b",): 4}
    assert allocate(sizes, 100) == sizes


def test_sample_is_byte_identical_across_runs():
    its = items(500)
    a = stratified_sample(its, "ds", 100, 20260815, [["category"]])
    b = stratified_sample(list(reversed(its)), "ds", 100, 20260815, [["category"]])
    assert orjson.dumps(a) == orjson.dumps(b)


def test_sample_changes_with_the_seed():
    its = items(500)
    a = stratified_sample(its, "ds", 100, 1, [["category"]])
    b = stratified_sample(its, "ds", 100, 2, [["category"]])
    assert a["uids"] != b["uids"]


def test_small_benchmark_rule_uses_every_item():
    its = items(40)
    s = stratified_sample(its, "ds", 2000, 1, [["category"]])
    assert s["sampling_applied"] is False
    assert s["n_sampled"] == s["n_available"] == 40
    assert len(set(s["uids"])) == 40


def test_two_level_key_falls_back_when_cells_would_be_thin():
    its = items(600, subcats=tuple(f"s{i}" for i in range(30)))
    key, info = choose_key(its, [["category", "subcategory"], ["category"]],
                           n_target=100, min_cell_after_alloc=15)
    assert key == ["category"]
    assert info["reason"] == "no thin cells"
    assert info["trace"][0]["n_thin_cells"] > 0


def test_two_level_key_is_kept_when_cells_survive():
    its = items(600, cats=("a", "b"), subcats=("s0", "s1"))
    key, info = choose_key(its, [["category", "subcategory"], ["category"]],
                           n_target=400, min_cell_after_alloc=15)
    assert key == ["category", "subcategory"]


def test_key_with_nulls_is_rejected():
    its = items(100)                       # subcategory is None
    key, info = choose_key(its, [["category", "subcategory"], ["category"]],
                           n_target=50, min_cell_after_alloc=15)
    assert key == ["category"]
    assert "null" in info["trace"][0]["rejected"]


def test_o1c_pool_draws_across_all_benchmarks_proportionally():
    sampled = {"big": items(800, dataset="big"),
               "small": items(200, dataset="small"),
               "withE": items(100, dataset="withE", n_options=5)}
    pool = build_o1c_pool(sampled, 100, 1)
    per = {k: v["sampled"] for k, v in pool["per_dataset"].items()}
    assert pool["n_sampled"] == 100
    assert per["withE"] == 0          # five-option items are ineligible
    assert per["big"] > per["small"] > 0
    assert all(u.startswith(("big::", "small::")) for u in pool["uids"])


def test_o1c_pool_is_deterministic():
    sampled = {"a": items(300, dataset="a"), "b": items(300, dataset="b")}
    assert orjson.dumps(build_o1c_pool(sampled, 50, 7)) == \
           orjson.dumps(build_o1c_pool(sampled, 50, 7))
