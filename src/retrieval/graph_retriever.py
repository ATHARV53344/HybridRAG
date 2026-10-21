"""Graph retrieval: entity linking -> bounded BFS -> provenance passages ranked by Eq. (5) proximity.

g(q, d) = max_{e in E(d)} 1 / (1 + dist_G(E_q, e)),  dist = hops from the nearest query entity.
Ties (many passages share g) are broken by proximity mass sum_e 1/(1+dist(e)), documented as an
implementation decision. Personalized PageRank (ablation A5) is available via `proximity="ppr"`.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import networkx as nx

from src.indexing.entities import extract_entities, normalize_entity
from src.indexing.store import Store
from src.retrieval.types import Evidence, RetrievalResult

_GENERIC = {"country", "city", "author", "company", "university", "novel", "river", "capital", "year",
            "born", "located"}


class GraphRetriever:
    def __init__(self, store: Store, h_max: int = 2, max_neighbors: int = 25, proximity: str = "hop"):
        self.store, self.h_max, self.max_neighbors, self.proximity = store, h_max, max_neighbors, proximity
        self.g: nx.Graph = store.kg.g
        # inverted index token -> nodes, and pid -> entities (E(d))
        self._tok2nodes: dict[str, set[str]] = defaultdict(set)
        self._pid2ents: dict[str, set[str]] = defaultdict(set)
        for n, data in self.g.nodes(data=True):
            for t in n.split():
                self._tok2nodes[t].add(n)
            for pid in data.get("passages", ()):
                self._pid2ents[pid].add(n)

    # ---- entity linking -----------------------------------------------------------
    def link_entities(self, question: str) -> list[str]:
        seeds: list[str] = []
        for qe in extract_entities(question):
            if qe in self.g:
                seeds.append(qe)
                continue
            qt = set(qe.split())
            best, best_j = None, 0.0
            for n in {n for t in qt for n in self._tok2nodes.get(t, ())}:
                nt = set(n.split())
                j = len(qt & nt) / len(qt | nt)
                if j > best_j:
                    best, best_j = n, j
            if best is not None and best_j >= 0.5:
                seeds.append(best)
        # secondary: node names appearing verbatim in the (lower-cased) question
        ql = " " + normalize_entity(question) + " "
        qtoks = set(ql.split())
        for n in {n for t in qtoks for n in self._tok2nodes.get(t, ())}:
            if n not in seeds and " " + n + " " in ql and n not in _GENERIC and len(n) > 3:
                seeds.append(n)
        return list(dict.fromkeys(seeds))

    # ---- traversal -----------------------------------------------------------------
    def _bfs(self, seeds: list[str]) -> tuple[dict[str, int], dict[str, str | None]]:
        dist = {s: 0 for s in seeds}
        parent: dict[str, str | None] = {s: None for s in seeds}
        dq = deque(seeds)
        while dq:
            u = dq.popleft()
            if dist[u] >= self.h_max:
                continue
            nbrs = sorted(self.g[u].items(), key=lambda kv: -kv[1].get("weight", 1))[: self.max_neighbors]
            for v, _ in nbrs:
                if v not in dist:
                    dist[v], parent[v] = dist[u] + 1, u
                    dq.append(v)
        return dist, parent

    def _path(self, node: str, parent: dict[str, str | None]) -> list[str]:
        path = [node]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])  # type: ignore[arg-type]
        return path[::-1]

    def retrieve(self, question: str, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        seeds = self.link_entities(question)
        if not seeds:
            return RetrievalResult("graph", [], time.perf_counter() - t0, fallback="no_entities",
                                   details={"seeds": []})
        dist, parent = self._bfs(seeds)
        ppr = None
        if self.proximity == "ppr":
            sub = self.g.subgraph(dist.keys())
            ppr = nx.pagerank(sub, personalization={s: 1.0 for s in seeds if s in sub}, weight="weight")
        cand: dict[str, dict] = {}
        for node, dn in dist.items():
            prox = (ppr[node] if ppr is not None else 1.0 / (1 + dn))
            for pid in self.g.nodes[node].get("passages", ()):
                c = cand.setdefault(pid, {"g": 0.0, "mass": 0.0, "best": node})
                if prox > c["g"]:
                    c["g"], c["best"] = prox, node
                c["mass"] += prox
        ranked = sorted(cand.items(), key=lambda kv: (-kv[1]["g"], -kv[1]["mass"], kv[0]))[:k]
        ev = []
        for pid, c in ranked:
            p = self.store.passages[pid]
            path = self._path(c["best"], parent)
            triples = [(a, self.g[a][b].get("relation", ""), b) for a, b in zip(path, path[1:])]
            ev.append(Evidence(pid, p.title, p.text, c["g"], "graph",
                               {"graph_score": c["g"], "path": path, "triples": triples,
                                "hops": dist[c["best"]]}))
        return RetrievalResult("graph", ev, time.perf_counter() - t0,
                               details={"seeds": seeds, "n_candidates": len(cand), "n_visited": len(dist)})

    def query_entity_count(self, question: str) -> int:
        return len(self.link_entities(question))
