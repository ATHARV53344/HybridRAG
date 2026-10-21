"""Retrieval metrics. Relevance label = passage is a gold supporting passage (HotpotQA supporting-fact
titles / MuSiQue is_supporting). Similarity alone never counts as relevance."""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def all_gold_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """1 if EVERY gold passage is in the top-k (the multi-hop requirement)."""
    if not gold:
        return float("nan")
    return float(set(gold) <= set(retrieved[:k]))


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & set(gold)) / len(top)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    for i, p in enumerate(retrieved, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Binary relevance nDCG."""
    if not gold:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(retrieved[:k]) if p in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0
