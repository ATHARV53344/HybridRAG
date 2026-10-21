"""Knowledge-graph construction from real passages (NetworkX).

Method (simple, auditable; documented in docs/IMPLEMENTATION_DECISIONS.md):
  * nodes  = normalised entities extracted by `entities.extract_entities` from passage title + text
  * edges  = sentence-level co-occurrence between two entities; edge stores the connecting text as
             the relation label and the set of passage IDs it came from
  * every node and edge stores `passages` (provenance) so graph retrieval returns source passages.
No ground-truth answers or supporting facts are used.
"""
from __future__ import annotations

import pickle
import time
from itertools import combinations
from pathlib import Path

import networkx as nx

from src.data.schema import Passage
from src.indexing.entities import extract_entities, normalize_entity, split_sentences


def _relation_text(sentence: str, a: str, b: str) -> str:
    s = sentence.lower()
    ia, ib = s.find(a), s.find(b)
    if ia < 0 or ib < 0:
        return "co-mentioned"
    lo, hi = (ia + len(a), ib) if ia < ib else (ib + len(b), ia)
    between = s[lo:hi].strip(" ,.;")
    return between if 0 < len(between) <= 40 else "co-mentioned"


class KnowledgeGraph:
    def __init__(self, graph: nx.Graph, build_time_s: float = 0.0):
        self.g = graph
        self.build_time_s = build_time_s

    @classmethod
    def build(cls, passages: list[Passage]) -> "KnowledgeGraph":
        t0 = time.perf_counter()
        g = nx.Graph()
        for p in passages:
            title_ent = normalize_entity(p.title)
            all_ents: set[str] = set()
            if title_ent:
                all_ents.add(title_ent)
            for sent in split_sentences(p.text):
                ents = extract_entities(sent)
                all_ents.update(ents)
                for a, b in combinations(ents, 2):
                    if a == b:
                        continue
                    rel = _relation_text(sent, a, b)
                    if g.has_edge(a, b):
                        g[a][b]["passages"].add(p.pid)
                        g[a][b]["weight"] += 1
                    else:
                        g.add_edge(a, b, passages={p.pid}, relation=rel, weight=1)
            # title entity connects to everything in its own passage (passage subject)
            for e in all_ents:
                if title_ent and e != title_ent:
                    if g.has_edge(title_ent, e):
                        g[title_ent][e]["passages"].add(p.pid)
                    else:
                        g.add_edge(title_ent, e, passages={p.pid}, relation="mentioned in article", weight=1)
            for e in all_ents:
                if e not in g:
                    g.add_node(e, passages=set())
                g.nodes[e].setdefault("passages", set()).add(p.pid)
        return cls(g, time.perf_counter() - t0)

    # -- persistence -------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"g": self.g, "t": self.build_time_s}, f)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["g"], d["t"])

    def stats(self) -> dict:
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "build_time_s": round(self.build_time_s, 4)}

    def size_bytes(self) -> int:
        return len(pickle.dumps(self.g))
