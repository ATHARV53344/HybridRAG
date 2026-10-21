"""Single entry point that runs a named route; handles graph-only fallback to vector."""
from __future__ import annotations

from src.indexing.store import Store
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.types import RetrievalResult
from src.retrieval.vector_retriever import VectorRetriever

ROUTES = ("vector", "graph", "hybrid")


class Retrievers:
    def __init__(self, store: Store, rcfg: dict):
        self.store, self.cfg = store, rcfg
        self.vector = VectorRetriever(store)
        self.graph = GraphRetriever(store, h_max=rcfg.get("h_max", 2), proximity=rcfg.get("proximity", "hop"))
        self.hybrid = HybridRetriever(store, self.vector, self.graph, rcfg)

    def run(self, route: str, question: str) -> RetrievalResult:
        k = self.cfg["top_k"]
        if route == "vector":
            return self.vector.retrieve(question, k)
        if route == "graph":
            res = self.graph.retrieve(question, k)
            if not res.evidence:  # documented fallback: no entity match -> vector evidence
                vres = self.vector.retrieve(question, k)
                vres.route, vres.fallback = "graph", f"{res.fallback or 'graph_empty'}->vector"
                vres.latency_s += res.latency_s
                return vres
            return res
        if route == "hybrid":
            return self.hybrid.retrieve(question, k)
        raise ValueError(f"Unknown route '{route}'. Expected one of {ROUTES}")
