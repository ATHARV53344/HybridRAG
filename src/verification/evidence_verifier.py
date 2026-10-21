"""Claim-level evidence verification (paper Eq. 6-7).

u_i  = max_{d in C} ENT(d, c_i)            (Eq. 6)
kappa = (1/m) * sum_i 1[u_i >= tau_u]      (Eq. 7)
decision: accept if kappa >= tau_a; qualify if tau_b <= kappa < tau_a; else abstain.

Entailment backends:
  * `lexical`: local token-containment proxy (no API). Weak; for offline tests/smoke runs.
  * `llm`: prompted judge returning JSON. If this is the SAME model as the generator, verification
    is NOT independent - state this in the paper. kappa measures support by context, not truth.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from src.generation.answer_generator import GenResult
from src.generation.llm_client import LLMClient
from src.generation.prompts import VERIFY_SYSTEM, VERIFY_USER
from src.retrieval.types import Evidence
from src.verification.claim_extractor import extract_claims

_STOP = {"the", "a", "an", "of", "in", "on", "is", "was", "by", "and", "to", "at", "for", "it", "as", "with"}


def lexical_entailment(passage: str, claim: str) -> float:
    ct = [t for t in re.findall(r"\w+", claim.lower()) if t not in _STOP]
    if not ct:
        return 0.0
    pt = set(re.findall(r"\w+", passage.lower()))
    return sum(t in pt for t in ct) / len(ct)


@dataclass
class ClaimResult:
    claim: str
    support: float  # u_i
    label: str  # supported | contradicted | not_enough_info
    evidence_ids: list[str]
    supported: bool


@dataclass
class Verification:
    decision: str  # accept | qualify | abstain | skipped
    kappa: float
    claims: list[ClaimResult] = field(default_factory=list)
    latency_s: float = 0.0
    n_llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None


class EvidenceVerifier:
    def __init__(self, cfg: dict, llm: LLMClient | None = None, backend: str = "lexical"):
        self.tau_u, self.tau_a, self.tau_b = cfg["tau_u"], cfg["tau_a"], cfg["tau_b"]
        if not (self.tau_b < self.tau_a):
            raise ValueError("require tau_b < tau_a")
        if backend == "llm" and llm is None:
            raise ValueError("llm backend requires an LLM client")
        self.llm, self.backend = llm, backend

    def _entail(self, ev: Evidence, claim: str, v: Verification) -> tuple[float, str]:
        if self.backend == "lexical":
            s = lexical_entailment(ev.text, claim)
            return s, ("supported" if s >= self.tau_u else "not_enough_info")
        resp = self.llm.complete(VERIFY_SYSTEM, VERIFY_USER.format(pid=ev.pid, text=ev.text, claim=claim), max_tokens=60)
        v.n_llm_calls += 1
        v.tokens_in += resp.tokens_in
        v.tokens_out += resp.tokens_out
        if resp.error:
            v.error = f"llm_error: {resp.error}"
            return 0.0, "not_enough_info"
        m = re.search(r"\{.*\}", resp.text, flags=re.S)
        try:
            d = json.loads(m.group(0)) if m else None
            label = d["label"] if d["label"] in ("supported", "contradicted", "not_enough_info") else "not_enough_info"
            return float(d.get("score", 1.0 if label == "supported" else 0.0)), label
        except Exception:  # malformed verifier output -> treated as NOT supported (conservative)
            v.error = "parse_error: malformed verifier output"
            return 0.0, "not_enough_info"

    def verify(self, question: str, gen: GenResult, evidence: list[Evidence]) -> Verification:
        t0 = time.perf_counter()
        v = Verification("skipped", 0.0)
        if gen.error or not gen.answer or not gen.sufficient:
            v.decision, v.latency_s = "abstain", time.perf_counter() - t0  # nothing to verify / model abstained
            return v
        claims = extract_claims(question, gen)
        if not evidence:  # no context -> nothing can be supported
            v.claims = [ClaimResult(c, 0.0, "not_enough_info", [], False) for c in claims]
            v.decision, v.kappa, v.latency_s = "abstain", 0.0, time.perf_counter() - t0
            return v
        for c in claims:
            best, best_ev, best_label = 0.0, [], "not_enough_info"
            for ev in evidence:
                s, label = self._entail(ev, c, v)
                if s > best:
                    best, best_ev, best_label = s, [ev.pid], label
            v.claims.append(ClaimResult(c, best, best_label, best_ev, best >= self.tau_u))
        v.kappa = sum(c.supported for c in v.claims) / max(len(v.claims), 1)
        v.decision = "accept" if v.kappa >= self.tau_a else "qualify" if v.kappa >= self.tau_b else "abstain"
        v.latency_s = time.perf_counter() - t0
        return v
