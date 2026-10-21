"""Evidence fusion (paper Eq. 4), RRF alternative (ablation A4), de-duplication and token budget.

Score vocabulary kept separate in Evidence.meta:
  vec_score / graph_score = raw retriever scores
  vec_norm / graph_norm   = min-max normalised scores
  fused                   = fused score (Eq. 4) or RRF score
"""
from __future__ import annotations

import numpy as np

from src.retrieval.types import Evidence


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in values}  # single / constant list -> all equally top
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def eq4_fuse(vec: list[Evidence], gr: list[Evidence], alpha: float, beta: float, gamma: float = 0.0) -> list[Evidence]:
    """s(d|q) = alpha*vec_norm + beta*graph_norm + gamma*recency(=0). vec_norm=0 if d not from the vector path."""
    if abs(alpha + beta + gamma - 1.0) > 1e-6:
        raise ValueError("alpha + beta + gamma must equal 1")
    vn = minmax({e.pid: e.meta["vec_score"] for e in vec})
    gn = minmax({e.pid: e.meta["graph_score"] for e in gr})
    by_pid: dict[str, Evidence] = {}
    for e in vec + gr:
        if e.pid not in by_pid:
            by_pid[e.pid] = Evidence(e.pid, e.title, e.text, 0.0, "hybrid", dict(e.meta))
        else:
            by_pid[e.pid].meta.update(e.meta)
    for pid, e in by_pid.items():
        e.meta["vec_norm"], e.meta["graph_norm"] = vn.get(pid, 0.0), gn.get(pid, 0.0)
        e.score = e.meta["fused"] = alpha * e.meta["vec_norm"] + beta * e.meta["graph_norm"]
    return sorted(by_pid.values(), key=lambda e: (-e.score, e.pid))


def rrf_fuse(vec: list[Evidence], gr: list[Evidence], k: int = 60) -> list[Evidence]:
    """Reciprocal Rank Fusion: sum 1/(k+rank). Uses ranks only, so no normalisation needed."""
    by_pid: dict[str, Evidence] = {}
    for lst in (vec, gr):
        for rank, e in enumerate(lst, start=1):
            if e.pid not in by_pid:
                by_pid[e.pid] = Evidence(e.pid, e.title, e.text, 0.0, "hybrid", dict(e.meta))
            else:
                by_pid[e.pid].meta.update(e.meta)
            by_pid[e.pid].score += 1.0 / (k + rank)
    for e in by_pid.values():
        e.meta["fused"] = e.score
    return sorted(by_pid.values(), key=lambda e: (-e.score, e.pid))


def dedupe_and_budget(ranked: list[Evidence], emb_lookup, k: int, theta_dup: float = 0.95,
                      budget_tokens: int | None = None) -> list[Evidence]:
    """Walk in score order; skip near-duplicates (cosine > theta_dup); stop at k passages / token budget."""
    chosen: list[Evidence] = []
    chosen_vecs: list[np.ndarray] = []
    used = 0
    for e in ranked:
        v = emb_lookup(e.pid)
        if any(float(v @ c) > theta_dup for c in chosen_vecs):
            continue
        n_tok = len(e.text.split())
        if budget_tokens is not None and chosen and used + n_tok > budget_tokens:
            break
        chosen.append(e)
        chosen_vecs.append(v)
        used += n_tok
        if len(chosen) >= k:
            break
    return chosen
