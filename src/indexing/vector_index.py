"""FAISS dense index (exact inner product on normalised vectors == cosine similarity)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np

from src.data.schema import Passage
from src.indexing.embeddings import Embedder


class VectorIndex:
    def __init__(self, index: faiss.Index, pids: list[str], embedder: Embedder, build_time_s: float = 0.0):
        self.index, self.pids, self.embedder, self.build_time_s = index, pids, embedder, build_time_s
        self.emb: np.ndarray | None = None  # kept for de-duplication

    @classmethod
    def build(cls, passages: list[Passage], embedder: Embedder) -> "VectorIndex":
        t0 = time.perf_counter()
        texts = [f"{p.title}. {p.text}" for p in passages]
        emb = embedder.encode(texts)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        vi = cls(index, [p.pid for p in passages], embedder, time.perf_counter() - t0)
        vi.emb = emb
        return vi

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self.embedder.encode([query])
        k = min(k, len(self.pids))
        scores, idx = self.index.search(q, k)
        return [(self.pids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def embedding_of(self, pid: str) -> np.ndarray:
        return self.emb[self.pids.index(pid)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        np.save(directory / "embeddings.npy", self.emb)
        (directory / "meta.json").write_text(json.dumps(
            {"pids": self.pids, "embedder": self.embedder.name, "dim": self.embedder.dim,
             "build_time_s": self.build_time_s}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "VectorIndex":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(f"Index built with '{meta['embedder']}' but embedder is '{embedder.name}'. Rebuild the index.")
        vi = cls(faiss.read_index(str(directory / "faiss.index")), meta["pids"], embedder, meta["build_time_s"])
        vi.emb = np.load(directory / "embeddings.npy")
        return vi

    def size_bytes(self) -> int:
        return int(self.emb.nbytes) if self.emb is not None else 0
