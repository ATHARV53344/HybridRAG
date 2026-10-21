"""Hybrid retrieval: vector + graph candidates -> fuse -> de-duplicate -> top-k."""
from __future__ import annotations

import time

from src.indexing.store import Store
from src.retrieval.fusion import dedupe_and_budget, eq4_fuse, rrf_fuse
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.types import RetrievalResult
from src.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    def __init__(self, store: Store, vr: VectorRetriever, gr: GraphRetriever, rcfg: dict):
        self.store, self.vr, self.gr, self.cfg = store, vr, gr, rcfg

    def retrieve(self, question: str, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        kv = self.cfg.get("kv", k)
        vres = self.vr.retrieve(question, kv)
        gres = self.gr.retrieve(question, kv)
        fallback = None
        if not gres.evidence:  # no recognised entities -> vector-only evidence
            fallback = f"{gres.fallback or 'graph_empty'}->vector"
            fused = vres.evidence
        elif self.cfg.get("fusion", "eq4") == "rrf":
            fused = rrf_fuse(vres.evidence, gres.evidence)
        else:
            fused = eq4_fuse(vres.evidence, gres.evidence, self.cfg["alpha"], self.cfg["beta"],
                             self.cfg.get("gamma", 0.0))
        final = dedupe_and_budget(fused, self.store.vindex.embedding_of, k,
                                  self.cfg.get("theta_dup", 0.95), self.cfg.get("context_budget_tokens"))
        for e in final:
            e.source = "hybrid"
        return RetrievalResult("hybrid", final, time.perf_counter() - t0, fallback=fallback,
                               details={"n_vec": len(vres.evidence), "n_graph": len(gres.evidence),
                                        "n_union": len(fused), "seeds": gres.details.get("seeds", [])})
