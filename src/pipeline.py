"""End-to-end pipeline for one example: route -> retrieve -> generate -> (verify) -> metrics record."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from src.data.schema import Example
from src.evaluation.answer_metrics import exact_match, token_f1
from src.evaluation.retrieval_metrics import all_gold_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.dispatcher import Retrievers
from src.retrieval.types import RetrievalResult
from src.verification.evidence_verifier import EvidenceVerifier


class Pipeline:
    def __init__(self, retrievers: Retrievers, generator: AnswerGenerator, verifier: EvidenceVerifier | None, top_k: int):
        self.R, self.gen, self.ver, self.k = retrievers, generator, verifier, top_k

    def run(self, method: str, ex: Example, router: Any | None, verify: bool = False) -> dict:
        """`router=None` => no retrieval (B0). The router sees ONLY the question (oracle gets eid)."""
        rec: dict = {"method": method, "eid": ex.eid, "question": ex.question, "gold_answer": ex.answer,
                     "qtype": ex.qtype, "split": ex.split, "gold_pids": ex.gold_pids, "error": None}
        t_all = time.perf_counter()
        if router is None:
            res = RetrievalResult("none", [], 0.0)
            rec.update(route="none", route_reason="no retrieval", route_features={}, routing_s=0.0)
        else:
            dec = router.route(ex.question, eid=ex.eid)
            res = self.R.run(dec.route, ex.question)
            rec.update(route=dec.route, route_reason=dec.reason, route_features=dec.features,
                       routing_s=dec.overhead_s, route_probs=dec.probs, oracle=dec.uses_ground_truth)
        rec.update(retrieved_pids=res.pids, retrieval_scores=[round(e.score, 4) for e in res.evidence],
                   retrieval_s=res.latency_s, fallback=res.fallback,
                   retrieval_paths=[e.meta.get("path") for e in res.evidence])
        g = self.gen.generate(ex.question, res.evidence)
        rec.update(original_answer=g.answer, citations=g.citations, sufficient=g.sufficient,
                   generation_s=g.latency_s, tokens_in=g.tokens_in, tokens_out=g.tokens_out, gen_raw=g.raw)
        if g.error:
            rec["error"] = g.error
        final = g.answer
        rec.update(verification=None, verification_s=0.0)
        if verify and self.ver is not None:
            v = self.ver.verify(ex.question, g, res.evidence)
            rec["verification"] = {"decision": v.decision, "kappa": v.kappa, "error": v.error,
                                   "claims": [asdict(c) for c in v.claims], "n_llm_calls": v.n_llm_calls}
            rec["verification_s"] = v.latency_s
            rec["tokens_in"] += v.tokens_in
            rec["tokens_out"] += v.tokens_out
            if v.decision == "abstain":
                final = ""  # withheld; original preserved in original_answer
        rec["predicted_answer"] = final
        rec["abstained"] = bool(verify and final == "" and g.answer != "")
        # ---- metrics (against gold; computed AFTER the system has produced its output) ----
        ret, gold = res.pids, ex.gold_pids
        k = self.k
        rec.update(em=exact_match(final, ex.answer), f1=token_f1(final, ex.answer),
                   f1_original=token_f1(g.answer, ex.answer), em_original=exact_match(g.answer, ex.answer),
                   recall_at_k=recall_at_k(ret, gold, k), all_gold_at_k=all_gold_at_k(ret, gold, k),
                   precision_at_k=precision_at_k(ret, gold, k), mrr=mrr(ret, gold), ndcg_at_k=ndcg_at_k(ret, gold, k))
        rec["total_s"] = time.perf_counter() - t_all
        return rec
