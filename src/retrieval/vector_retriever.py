"""Dense vector retrieval (cosine similarity over FAISS)."""
from __future__ import annotations

import time

from src.indexing.store import Store
from src.retrieval.types import Evidence, RetrievalResult


class VectorRetriever:
    def __init__(self, store: Store):
        self.store = store

    def retrieve(self, question: str, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        hits = self.store.vindex.search(question, k)
        ev = []
        for pid, s in hits:
            p = self.store.passages[pid]
            ev.append(Evidence(pid, p.title, p.text, s, "vector", {"vec_score": s}))
        return RetrievalResult("vector", ev, time.perf_counter() - t0)
