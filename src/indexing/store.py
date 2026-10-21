"""Bundles passages + vector index + graph; handles build/save/load of artifacts."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.config import ROOT, get_logger
from src.data.schema import Dataset, Passage
from src.indexing.embeddings import Embedder, get_embedder
from src.indexing.graph_builder import KnowledgeGraph
from src.indexing.vector_index import VectorIndex

log = get_logger("store")


@dataclass
class Store:
    passages: dict[str, Passage]
    vindex: VectorIndex
    kg: KnowledgeGraph
    embedder: Embedder


def corpus_fingerprint(passages: list[Passage]) -> str:
    """Hash of the sorted passage IDs: cached artifacts are only valid for exactly this corpus."""
    h = hashlib.sha1("|".join(sorted(p.pid for p in passages)).encode("utf-8"))
    return h.hexdigest()[:10]


def artifact_dirs(dataset: str, embedder_name: str, fingerprint: str) -> tuple[Path, Path]:
    safe = embedder_name.replace("/", "_")
    return (ROOT / "artifacts" / "indexes" / f"{dataset}__{safe}__{fingerprint}",
            ROOT / "artifacts" / "graphs" / f"{dataset}__{fingerprint}.pkl")


def build_store(ds: Dataset, emb_cfg: dict, save: bool = True) -> Store:
    embedder = get_embedder(emb_cfg)
    vi = VectorIndex.build(ds.passages, embedder)
    kg = KnowledgeGraph.build(ds.passages)
    if save:
        idx_dir, g_path = artifact_dirs(ds.name, embedder.name, corpus_fingerprint(ds.passages))
        vi.save(idx_dir)
        kg.save(g_path)
        log.info("Saved index -> %s, graph -> %s", idx_dir, g_path)
    return Store({p.pid: p for p in ds.passages}, vi, kg, embedder)


def load_store(ds: Dataset, emb_cfg: dict) -> Store:
    """Load cached index+graph (no re-embedding); build if missing."""
    embedder = get_embedder(emb_cfg)
    idx_dir, g_path = artifact_dirs(ds.name, embedder.name, corpus_fingerprint(ds.passages))
    if (idx_dir / "faiss.index").exists() and g_path.exists():
        vi = VectorIndex.load(idx_dir, embedder)
        kg = KnowledgeGraph.load(g_path)
        log.info("Loaded cached index and graph for %s", ds.name)
        return Store({p.pid: p for p in ds.passages}, vi, kg, embedder)
    log.info("No cached artifacts for %s - building", ds.name)
    return build_store(ds, emb_cfg, save=True)
