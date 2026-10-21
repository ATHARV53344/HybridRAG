"""Atomic-claim extraction for short-answer QA (paper Sec. 3.5).

Claims = the model's stated supporting facts (<=3) plus the final answer itself. No extra LLM call.
"""
from __future__ import annotations

from src.generation.answer_generator import GenResult


def extract_claims(question: str, gen: GenResult) -> list[str]:
    claims = [f.strip() for f in gen.facts if f and f.strip()]
    ans = gen.answer.strip()
    if ans:
        claims.append(ans)  # the final answer is always a claim
    seen, out = set(), []
    for c in claims:
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out
