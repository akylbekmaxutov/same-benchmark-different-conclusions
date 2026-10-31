"""Uncertainty: bootstrap CIs, paired bootstrap, McNemar, binomial tests."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

DEFAULT_RESAMPLES = 10_000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_ci(values, n_resamples: int = DEFAULT_RESAMPLES, seed: int = 20260815,
                 alpha: float = 0.05, statistic=np.mean) -> dict:
    """Item-level bootstrap CI for a mean-like statistic."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if v.size == 1:
        p = float(statistic(v))
        return {"point": p, "lo": p, "hi": p, "n": 1}
    rng = _rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    draws = statistic(v[idx], axis=1)
    return {"point": float(statistic(v)),
            "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)),
            "n": int(v.size)}


def paired_bootstrap_diff(a, b, n_resamples: int = DEFAULT_RESAMPLES,
                          seed: int = 20260815, alpha: float = 0.05) -> dict:
    """Paired over items: resample item indices once and apply to both arms."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    if a.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "separable": False}
    rng = _rng(seed)
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    draws = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))
    return {"diff": float(a.mean() - b.mean()), "lo": lo, "hi": hi,
            "n": int(a.size), "separable": bool(lo > 0 or hi < 0)}


def mcnemar(b: int, c: int, exact: bool = True) -> dict:
    """Paired binary contrast.  b = a-only successes, c = b-only successes."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": float("nan"), "p_value": 1.0, "test": "degenerate"}
    if exact and n < 25:
        p = float(sps.binomtest(b, n, 0.5).pvalue)
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": p, "test": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "statistic": float(stat),
            "p_value": float(sps.chi2.sf(stat, 1)), "test": "chi2_cc"}


def binom_vs_chance(n_correct: int, n: int, p_chance: float, alpha: float = 0.05) -> dict:
    """Observed accuracy with a binomial CI, against a per-cell chance rate, plus
    the exact one-sided p for `above chance`."""
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "p_chance": p_chance,
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_greater": float("nan"), "above_chance": False,
                "near_chance": True}
    res = sps.binomtest(int(n_correct), int(n), p_chance, alternative="greater")
    ci = sps.binomtest(int(n_correct), int(n), p_chance).proportion_ci(
        confidence_level=1 - alpha, method="wilson")
    acc = n_correct / n
    return {"n": int(n), "accuracy": float(acc), "p_chance": float(p_chance),
            "ci_lo": float(ci.low), "ci_hi": float(ci.high),
            "p_greater": float(res.pvalue),
            "above_chance": bool(res.pvalue < alpha),
            "near_chance": bool(ci.low <= p_chance <= ci.high)}
