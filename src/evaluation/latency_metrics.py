"""Latency summaries and bootstrap statistics."""
from __future__ import annotations

import numpy as np


def summarize_latency(values: list[float]) -> dict:
    v = np.array([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    out = {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v))}
    out["p95"] = float(np.percentile(v, 95)) if v.size >= 20 else None  # None = sample too small
    return out


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    v = np.array([x for x in values if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap on per-question differences (a - b) over the SAME examples."""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    return {"n": int(d.size), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))], "p_two_sided": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))}
