"""Rule-based entity extraction (auditable, no LLM, no network).

Capitalised-phrase heuristic. Known weaknesses (document in the paper): misses lower-case entities,
splits some names, treats some sentence-initial words as entities. The same extractor is used for
passages (graph building) and queries (entity linking) so both sides are consistent.
"""
from __future__ import annotations

import re

_TOKEN = r"[A-Z][A-Za-z0-9'’\-]*"
_PHRASE = re.compile(rf"{_TOKEN}(?:\s+(?:(?:of|the|de|von|van|and|for)\s+)?{_TOKEN})*")
_STOP_SINGLE = {
    "the", "a", "an", "who", "what", "which", "when", "where", "why", "how", "in", "on", "at", "of", "is",
    "are", "was", "were", "do", "does", "did", "dr", "mr", "mrs", "ms", "prof", "this", "that", "it", "he",
    "she", "they", "his", "her", "their", "its", "name", "both", "are", "has", "have", "had", "from", "and",
    "or", "for", "to", "by", "with", "as", "if", "after", "before", "during", "also", "one", "two",
}
_LEAD = re.compile(r"^(the|a|an|dr|mr|mrs|ms|prof)\s+", re.I)


def normalize_entity(name: str) -> str:
    name = name.strip()
    prev = None
    while prev != name:
        prev, name = name, _LEAD.sub("", name)
    return re.sub(r"\s+", " ", name.lower()).strip(" .,'’")


def extract_entities(text: str) -> list[str]:
    """Return de-duplicated normalised entity names in order of appearance."""
    out: list[str] = []
    for m in _PHRASE.finditer(text):
        raw = m.group(0)
        norm = normalize_entity(raw)
        if not norm or norm in _STOP_SINGLE:
            continue
        # drop leading stop-words that survived (e.g. "Which Country")
        toks = norm.split()
        while toks and toks[0] in _STOP_SINGLE:
            toks = toks[1:]
        norm = " ".join(toks)
        if norm and norm not in _STOP_SINGLE and norm not in out:
            out.append(norm)
    return out


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
