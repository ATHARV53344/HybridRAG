"""CLI: python -m src.indexing.build --dataset hotpotqa [--config configs/experiments.yaml]"""
from __future__ import annotations

import argparse
import json

from src.config import ROOT, get_logger, load_config
from src.data.prepare import processed_dir
from src.data.schema import Dataset
from src.indexing.store import build_store

log = get_logger("build")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--embedding-backend", default=None, help="override: sbert | tfidf")
    a = ap.parse_args()
    cfg = load_config(a.config)
    emb_cfg = dict(cfg["embedding"])
    if a.embedding_backend:
        emb_cfg["backend"] = a.embedding_backend
    d = processed_dir(a.dataset, cfg["dataset"].get("data_dir", "data"))
    if not (d / "manifest.json").exists():
        raise SystemExit(f"Processed dataset not found at {d}. Run: python -m src.data.prepare --dataset {a.dataset}")
    ds = Dataset.load(d)
    store = build_store(ds, emb_cfg, save=True)
    report = {"dataset": ds.name, "n_passages": len(ds.passages), "embedder": store.embedder.name,
              "vector_index_build_s": round(store.vindex.build_time_s, 4),
              "vector_index_bytes": store.vindex.size_bytes(), "graph": store.kg.stats(),
              "graph_bytes": store.kg.size_bytes()}
    out = ROOT / "results" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"index_report_{ds.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
