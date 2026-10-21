"""Shared retrieval data types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evidence:
    pid: str
    title: str
    text: str
    score: float  # final score used for ranking in the producing retriever
    source: str  # vector | graph | hybrid
    meta: dict = field(default_factory=dict)  # vec_score, graph_score, norm scores, path, triples


@dataclass
class RetrievalResult:
    route: str
    evidence: list[Evidence]
    latency_s: float = 0.0
    fallback: str | None = None  # e.g. "no_entities->vector"
    details: dict = field(default_factory=dict)

    @property
    def pids(self) -> list[str]:
        return [e.pid for e in self.evidence]
