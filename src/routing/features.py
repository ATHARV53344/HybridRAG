"""Inference-time query features phi(q). Uses ONLY the question text (and optionally the indexes).

Never touches answers, supporting facts or metrics. Estimated hop count is a transparent heuristic
(the paper proposes a small classifier; no hop labels exist in our data - see IMPLEMENTATION_DECISIONS).
"""
from __future__ import annotations

import re
from typing import Callable

from src.indexing.entities import extract_entities

COMPARISON = re.compile(r"\b(both|older|younger|same|differ\w*|more|less|larger|smaller|before|after|than|either|neither|compare\w*)\b", re.I)
BRIDGE = re.compile(r"\b(of the|that|who|whose|where|which|whom)\b", re.I)
WH = re.compile(r"\b(who|what|which|when|where|why|how|whose|whom)\b", re.I)

FEATURE_NAMES = ["n_entities", "n_linked", "length", "has_comparison", "n_bridge_cues", "n_clauses",
                 "est_hops", "top1_sim"]


def estimate_hops(question: str) -> int:
    """1 + number of nested-reference cues ('of the X', relative clauses), capped at 4."""
    nested = len(re.findall(r"\b(of the|of a|that|whose|where|who)\b", question, flags=re.I))
    return max(1, min(4, 1 + nested))


def extract_features(question: str, linker: Callable[[str], list[str]] | None = None,
                     top1_fn: Callable[[str], float] | None = None) -> dict[str, float]:
    ents = extract_entities(question)
    return {
        "n_entities": float(len(ents)),
        "n_linked": float(len(linker(question))) if linker else float(len(ents)),
        "length": float(len(question.split())),
        "has_comparison": 1.0 if COMPARISON.search(question) else 0.0,
        "n_bridge_cues": float(len(BRIDGE.findall(question))),
        "n_clauses": float(len(WH.findall(question)) + question.lower().count(" and ")),
        "est_hops": float(estimate_hops(question)),
        "top1_sim": float(top1_fn(question)) if top1_fn else 0.0,
    }
