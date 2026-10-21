"""Answer generation with structured-output parsing. Failures are recorded, never hidden."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.generation.llm_client import LLMClient
from src.generation.prompts import (GEN_SYSTEM, GEN_USER, NO_RETRIEVAL_USER, format_context, format_triples)
from src.retrieval.types import Evidence

INSUFFICIENT = "insufficient evidence"


@dataclass
class GenResult:
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    sufficient: bool = False
    facts: list[str] = field(default_factory=list)
    raw: str = ""
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    parse_ok: bool = False
    error: str | None = None  # API/parse failure -> counted in the failure rate


def parse_json_answer(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and "answer" in d else None


class AnswerGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, question: str, evidence: list[Evidence]) -> GenResult:
        if evidence:
            user = GEN_USER.format(question=question, context=format_context(evidence),
                                   triples=format_triples(evidence))
        else:
            user = NO_RETRIEVAL_USER.format(question=question)
        resp = self.llm.complete(GEN_SYSTEM, user)
        g = GenResult(raw=resp.text, latency_s=resp.latency_s, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out)
        if resp.error:
            g.error = f"llm_error: {resp.error}"  # answer stays empty: a failure is never a correct answer
            return g
        d = parse_json_answer(resp.text)
        if d is None:
            g.error = "parse_error: malformed model output"
            return g
        g.parse_ok = True
        g.answer = str(d.get("answer", "")).strip()
        g.citations = [str(c) for c in d.get("citations", []) if isinstance(c, (str, int))]
        g.sufficient = bool(d.get("sufficient", True)) and g.answer.lower() != INSUFFICIENT
        g.facts = [str(f) for f in d.get("facts", [])][:3]
        return g
