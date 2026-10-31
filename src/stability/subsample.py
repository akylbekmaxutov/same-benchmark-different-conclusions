"""Stratified subsampling with a fixed seed.

The index files written here are released with the paper, so they must be
regenerable byte-identically: every random draw is keyed by an explicit string,
never by iteration order, and every list is written in sorted order.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .conditions import o1c_eligible
from .io import write_json
from .schema import Item


# ------------------------------------------------------------------ allocation
def allocate(cell_sizes: dict[tuple, int], n_target: int) -> dict[tuple, int]:
    """Largest-remainder proportional allocation, capped by cell size, with at
    least one item per non-empty cell.  Deterministic: ties break on sorted key."""
    cells = {k: v for k, v in cell_sizes.items() if v > 0}
    total = sum(cells.values())
    if total <= n_target:
        return dict(cells)

    raw = {k: v * n_target / total for k, v in cells.items()}
    alloc = {k: min(int(raw[k]), cells[k]) for k in cells}
    left = n_target - sum(alloc.values())
    order = sorted(cells, key=lambda k: (-(raw[k] - int(raw[k])), _skey(k)))
    i = 0
    while left > 0 and order:
        k = order[i % len(order)]
        if alloc[k] < cells[k]:
            alloc[k] += 1
            left -= 1
        elif all(alloc[c] >= cells[c] for c in order):
            break
        i += 1

    # every non-empty cell contributes at least one item
    deficit = [k for k in cells if alloc[k] == 0]
    for k in sorted(deficit, key=_skey):
        alloc[k] = 1
    over = sum(alloc.values()) - n_target
    if over > 0:
        donors = sorted(cells, key=lambda k: (-alloc[k], _skey(k)))
        i = 0
        while over > 0:
            k = donors[i % len(donors)]
            if alloc[k] > 1:
                alloc[k] -= 1
                over -= 1
            i += 1
            if i > 10 * len(donors) * max(1, over):  # pragma: no cover
                raise RuntimeError("allocation failed to converge")
    return alloc


def _skey(k) -> str:
    return "\x1f".join("" if p is None else str(p) for p in k)


# ------------------------------------------------------------------ stratify
def _cell_of(item: Item, key: Sequence[str]) -> tuple:
    return tuple(getattr(item, k) for k in key)


def _key_usable(items: list[Item], key: Sequence[str]) -> bool:
    return all(all(getattr(i, k) is not None for k in key) for i in items)


def choose_key(items: list[Item], candidate_keys: Sequence[Sequence[str]],
               n_target: int, min_cell_after_alloc: int) -> tuple[list[str], dict]:
    """First candidate key whose allocation keeps every cell that *could* reach
    the floor at or above it.  Cells smaller than the floor to begin with are
    not held against the key -- they are exhausted, not thinned."""
    trace = []
    usable: list[list[str]] = []
    last = len(candidate_keys) - 1
    for idx, raw_key in enumerate(candidate_keys):
        key = list(raw_key)
        if not _key_usable(items, key):
            trace.append({"key": key, "rejected": "field is null on some items"})
            continue
        usable.append(key)
        sizes: dict[tuple, int] = defaultdict(int)
        for it in items:
            sizes[_cell_of(it, key)] += 1
        alloc = allocate(dict(sizes), n_target)
        thin = {_skey(k): alloc[k] for k in alloc
                if alloc[k] < min_cell_after_alloc and sizes[k] >= min_cell_after_alloc}
        trace.append({"key": key, "n_cells": len(sizes),
                      "min_alloc": min(alloc.values()) if alloc else 0,
                      "n_thin_cells": len(thin),
                      "thin_cells": sorted(thin)[:10]})
        if not thin:
            return key, {"trace": trace, "accepted": key, "reason": "no thin cells"}
        if idx == last:
            return key, {"trace": trace, "accepted": key,
                         "reason": "last candidate; thin cells tolerated"}
    if not usable:
        raise ValueError("no usable stratification key: all candidates have null fields")
    key = usable[-1]
    return key, {"trace": trace, "accepted": key, "reason": "fallback to last usable candidate"}


# ------------------------------------------------------------------ sampling
def stratified_sample(items: list[Item], dataset: str, n_target: int, seed: int,
                      candidate_keys: Sequence[Sequence[str]],
                      min_cell_after_alloc: int = 15) -> dict:
    """Returns the sample-index payload (also the on-disk format)."""
    n_available = len(items)
    if n_available == 0:
        raise ValueError(f"{dataset}: no items to sample")

    # Small-benchmark rule: fewer items than the target -> use all of them.
    if n_available <= n_target:
        return {
            "dataset": dataset,
            "seed": seed,
            "stratify_key": None,
            "key_selection": {"reason": "sampling not applied"},
            "n_available": n_available,
            "n_target": n_target,
            "n_sampled": n_available,
            "sampling_applied": False,
            "strata": {},
            "uids": sorted(i.uid for i in items),
        }

    key, key_info = choose_key(items, candidate_keys, n_target, min_cell_after_alloc)
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for it in items:
        buckets[_cell_of(it, key)].append(it.uid)
    sizes = {k: len(v) for k, v in buckets.items()}
    alloc = allocate(sizes, n_target)

    picked: list[str] = []
    strata: dict[str, dict] = {}
    for cell in sorted(buckets, key=_skey):
        pool = sorted(buckets[cell])
        take = alloc.get(cell, 0)
        rng = random.Random(f"{dataset}|{seed}|{_skey(cell)}")
        chosen = sorted(rng.sample(pool, take)) if take < len(pool) else list(pool)
        picked.extend(chosen)
        strata[_skey(cell)] = {"available": len(pool), "sampled": len(chosen)}

    picked = sorted(picked)
    return {
        "dataset": dataset,
        "seed": seed,
        "stratify_key": key,
        "key_selection": key_info,
        "n_available": n_available,
        "n_target": n_target,
        "n_sampled": len(picked),
        "sampling_applied": True,
        "strata": strata,
        "uids": picked,
    }


def sample_index_path(work_dir: str | Path, dataset: str) -> Path:
    return Path(work_dir) / "sample_index" / f"{dataset}_sample.json"


def write_sample_index(work_dir: str | Path, payload: dict) -> Path:
    path = sample_index_path(work_dir, payload["dataset"])
    write_json(path, payload)
    return path


# ------------------------------------------------------------------ O1-C pool
def build_o1c_pool(sampled: dict[str, list[Item]], n_target: int, seed: int) -> dict:
    """~n_target four-option items, drawn across all three benchmarks in
    proportion to their eligible counts -- never letting whichever file is
    listed first fill the quota."""
    eligible = {ds: sorted(i.uid for i in items if o1c_eligible(i))
                for ds, items in sampled.items()}
    counts = {ds: len(v) for ds, v in eligible.items()}
    alloc = allocate({(ds,): n for ds, n in counts.items()}, n_target)

    uids: list[str] = []
    per_dataset = {}
    for ds in sorted(eligible):
        take = alloc.get((ds,), 0)
        pool = eligible[ds]
        rng = random.Random(f"o1c|{seed}|{ds}")
        chosen = sorted(rng.sample(pool, take)) if take < len(pool) else list(pool)
        uids.extend(chosen)
        per_dataset[ds] = {"eligible": counts[ds], "sampled": len(chosen)}
    return {
        "pool": "o1c",
        "seed": seed,
        "n_target": n_target,
        "n_sampled": len(uids),
        "eligibility_rule": "four-option items only (E is null)",
        "per_dataset": per_dataset,
        "uids": sorted(uids),
    }


# ------------------------------------------------------------------ NQ probe pool
def build_nq_pool(sampled: dict[str, list[Item]], n_per_dataset: int, seed: int) -> dict:
    uids: list[str] = []
    per_dataset = {}
    for ds in sorted(sampled):
        pool = sorted(i.uid for i in sampled[ds]
                      if i.relevant_noise and i.irrelevant_noise)
        rng = random.Random(f"nq|{seed}|{ds}")
        take = min(n_per_dataset, len(pool))
        chosen = sorted(rng.sample(pool, take)) if take < len(pool) else list(pool)
        uids.extend(chosen)
        per_dataset[ds] = {"available": len(pool), "sampled": len(chosen)}
    return {
        "pool": "nq",
        "seed": seed,
        "n_per_dataset": n_per_dataset,
        "n_sampled": len(uids),
        "per_dataset": per_dataset,
        "uids": sorted(uids),
    }
