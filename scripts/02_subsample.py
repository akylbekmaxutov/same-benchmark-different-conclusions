#!/usr/bin/env python3
"""Stratified subsampling + the O1-C and NQ pools.  Fixed seeds, released."""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from stability.config import load_defaults
from stability.io import write_json
from stability.normalize import DATASETS, iter_normalized
from stability.subsample import (build_nq_pool, build_o1c_pool, sample_index_path,
                                 stratified_sample, write_sample_index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=None, help="items per benchmark")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    d = load_defaults()
    work = Path(d["work_dir"])
    sub = d["subsample"]
    n_target = args.n if args.n is not None else int(sub["n_target"])
    seed = args.seed if args.seed is not None else int(sub["seed"])
    min_cell = int(sub.get("min_cell_after_alloc", 15))

    all_items = {ds: list(iter_normalized(work / "normalized" / f"{ds}.jsonl"))
                 for ds in d.get("datasets", DATASETS)}
    assert set(all_items) == set(DATASETS), f"missing datasets: {set(DATASETS) - set(all_items)}"

    sampled: dict[str, list] = {}
    overview = {"n_target": n_target, "seed": seed, "datasets": {}}
    for ds, items in all_items.items():
        keys = sub["stratify_keys"][ds]
        payload = stratified_sample(items, ds, n_target, seed, keys, min_cell)
        write_sample_index(work, payload)
        keep = set(payload["uids"])
        sampled[ds] = [i for i in items if i.uid in keep]
        overview["datasets"][ds] = {k: payload[k] for k in
                                    ("n_available", "n_target", "n_sampled",
                                     "sampling_applied", "stratify_key")}
        overview["datasets"][ds]["key_reason"] = payload["key_selection"].get("reason")
        print(f"{ds:14s} available={payload['n_available']:6d} "
              f"sampled={payload['n_sampled']:5d} "
              f"applied={payload['sampling_applied']} "
              f"key={payload['stratify_key']} "
              f"({payload['key_selection'].get('reason')}) -> "
              f"{sample_index_path(work, ds)}")

    o1c = build_o1c_pool(sampled, int(d["o1c_pool"]["n_target"]), int(d["o1c_pool"]["seed"]))
    write_json(work / "sample_index" / "o1c_pool.json", o1c)
    print(f"o1c pool      n={o1c['n_sampled']} per_dataset="
          f"{ {k: v['sampled'] for k, v in o1c['per_dataset'].items()} }")

    nq = build_nq_pool(sampled, int(d["nq_pool"]["n_per_dataset"]), int(d["nq_pool"]["seed"]))
    write_json(work / "sample_index" / "nq_pool.json", nq)
    print(f"nq pool       n={nq['n_sampled']} per_dataset="
          f"{ {k: v['sampled'] for k, v in nq['per_dataset'].items()} }")

    total = sum(v["n_sampled"] for v in overview["datasets"].values())
    overview["n_total_sampled"] = total
    write_json(work / "sample_index" / "_overview.json", overview)
    print(f"\nTOTAL sampled items across the three benchmarks: {total}")
    if total < 3 * n_target:
        print("NOTE: total is below 3 x n_target -- the small-benchmark rule fired. "
              "Recompute the generation volume before committing the grid.")


if __name__ == "__main__":
    main()
