"""Prompt templates. Identical across all retrieval methods (only the evidence differs)."""
from __future__ import annotations

from src.retrieval.types import Evidence

GEN_SYSTEM = (
    "You answer questions using ONLY the labelled passages provided. "
    "If the passages do not contain enough information, say so. "
    "Respond with a single JSON object and nothing else."
)

GEN_USER = """Question: {question}

Passages:
{context}
{triples}
Instructions:
- Answer with the shortest possible phrase (a few words, a name, a date or yes/no).
- Cite the IDs of the passages that support your answer.
- If the passages are insufficient, set "sufficient" to false and "answer" to "insufficient evidence".
- "facts" lists at most 3 short factual statements, taken from the passages, that your answer relies on.

JSON format:
{{"answer": "...", "citations": ["p_..."], "sufficient": true, "facts": ["..."]}}"""

NO_RETRIEVAL_USER = """Question: {question}

Answer from your own knowledge with the shortest possible phrase.
JSON format:
{{"answer": "...", "citations": [], "sufficient": true, "facts": []}}"""

VERIFY_SYSTEM = ("You are a strict fact-checking assistant. Judge ONLY from the passage given. "
                 "Respond with a single JSON object and nothing else.")

VERIFY_USER = """Passage [{pid}]: {text}

Claim: {claim}

Does the passage support the claim?
JSON format: {{"label": "supported" | "contradicted" | "not_enough_info", "score": 0.0-1.0}}
where score is the probability that the passage entails the claim."""


def format_context(evidence: list[Evidence]) -> str:
    return "\n".join(f"[{e.pid}] {e.title}: {e.text}" for e in evidence)


def format_triples(evidence: list[Evidence], max_triples: int = 8) -> str:
    seen, lines = set(), []
    for e in evidence:
        for a, r, b in e.meta.get("triples", []) or []:
            if (a, r, b) not in seen:
                seen.add((a, r, b))
                lines.append(f"{a} - {r} - {b}")
    if not lines:
        return ""
    return "\nRelations linking the passages:\n" + "\n".join(lines[:max_triples]) + "\n"
