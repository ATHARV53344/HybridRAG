"""Shared experiment machinery used by run_baselines and run_ablations."""
from __future__ import annotations

import copy
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import ROOT, get_logger, git_commit
from src.data.loaders import load_dataset_by_name
from src.data.prepare import processed_dir
from src.data.schema import Dataset
from src.generation.answer_generator import AnswerGenerator
from src.generation.llm_client import get_llm
from src.indexing.store import load_store
from src.pipeline import Pipeline
from src.retrieval.dispatcher import Retrievers
from src.routing.adaptive_router import FixedRouter, LearnedRouter, OracleRouter, RuleRouter, derive_labels, ROUTES
from src.routing.train import measured_costs, train_router
from src.verification.evidence_verifier import EvidenceVerifier

log = get_logger("experiment")

FIXED = {"vector_only": "vector", "graph_only": "graph", "hybrid_fixed": "hybrid"}


def load_or_prepare(cfg: dict) -> Dataset:
    d = cfg["dataset"]
    pdir = processed_dir(d["name"], d.get("data_dir", "data"))
    if (pdir / "manifest.json").exists():
        ds = Dataset.load(pdir)
    else:
        log.info("Processed dataset missing; preparing '%s'", d["name"])
        ds = load_dataset_by_name(d["name"], d.get("max_samples", 200), cfg.get("seed", 42))
        ds.save(pdir, cfg.get("seed", 42), d.get("max_samples"))
    return ds


def make_run_id(tag: str) -> str:
    return f"{tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def run_experiment(cfg: dict, split: str = "test", max_samples: int | None = None, methods: list[str] | None = None,
                   tag: str = "baselines", progress=None) -> tuple[str, Path]:
    cfg = copy.deepcopy(cfg)
    methods = methods or cfg["methods"]
    ds = load_or_prepare(cfg)
    store = load_store(ds, cfg["embedding"])
    R = Retrievers(store, cfg["retrieval"])
    llm = get_llm(cfg["generation"])
    backend = "lexical" if llm.provider == "mock" else "llm"
    verifier = EvidenceVerifier(cfg["verification"], llm, backend)
    pipe = Pipeline(R, AnswerGenerator(llm), verifier, cfg["retrieval"]["top_k"])
    linker = R.graph.link_entities

    by_split = {s: [e for e in ds.examples if e.split == s] for s in ("train", "dev", "test")}
    eval_ex = by_split[split][:max_samples] if max_samples else by_split[split]
    if not eval_ex:
        raise SystemExit(f"No examples in split '{split}'. Increase --max-samples at prepare time.")

    run_id = make_run_id(tag)
    out = ROOT / cfg.get("output_dir", "results") / "raw" / run_id
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    router_info = None
    needs_learned = any(m in methods for m in ("adaptive", "adaptive_verified")) and cfg["router"]["mode"] == "learned"
    learned = None
    if needs_learned:
        learned, router_info = train_router(pipe, linker, by_split["train"], by_split["dev"], cfg["router"])
    rule = RuleRouter(linker)
    adaptive_router = learned if learned is not None else rule
    t_start = time.perf_counter()
    total = len(methods) * len(eval_ex)
    done = 0

    def tick(m):
        nonlocal done
        done += 1
        if progress:
            progress(done, total, m)

    best_route: dict[str, str] = {}
    ordered = sorted(methods, key=lambda m: (m == "oracle", m))  # oracle needs fixed-route runs first
    if "oracle" in ordered:
        for m in FIXED:
            if m not in ordered:
                ordered.insert(0, m)  # needed for oracle labels, flagged as auxiliary below
    aux = set(ordered) - set(methods)
    fixed_recs: dict[str, list[dict]] = {}
    for m in ordered:
        if m == "oracle":  # Eq. 3 on the evaluated questions themselves: ground truth, analysis only
            costs = measured_costs({FIXED[k]: v for k, v in fixed_recs.items()})
            for i, ex in enumerate(eval_ex):
                f1s = {FIXED[k]: fixed_recs[k][i]["f1_original"] for k in FIXED}
                best_route[ex.eid] = derive_labels(f1s, costs, cfg["router"].get("epsilon", 0.02))
        for ex in eval_ex:
            if m == "no_retrieval":
                r = pipe.run(m, ex, None)
            elif m in FIXED:
                r = pipe.run(m, ex, FixedRouter(FIXED[m]))
            elif m == "rule_router":
                r = pipe.run(m, ex, rule)
            elif m == "adaptive":
                r = pipe.run(m, ex, adaptive_router)
            elif m == "adaptive_verified":
                r = pipe.run(m, ex, adaptive_router, verify=True)
            elif m == "oracle":
                r = pipe.run(m, ex, OracleRouter(best_route))
            else:
                raise ValueError(f"Unknown method '{m}'")
            if m in aux:
                r["method"] = m  # auxiliary run kept for oracle labelling only
                r["auxiliary"] = True
            else:
                tick(m)
            records.append(r)
            if m in FIXED:
                fixed_recs.setdefault(m, []).append(r)

    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    meta = {"run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"), "split": split,
            "dataset": ds.manifest(cfg.get("seed", 42), cfg["dataset"].get("max_samples")),
            "n_eval_examples": len(eval_ex), "methods": methods, "auxiliary_methods": sorted(aux),
            "seed": cfg.get("seed", 42), "config": cfg, "git_commit": git_commit(),
            "embedder": store.embedder.name, "llm_provider": llm.provider, "llm_model": llm.model,
            "verifier_backend": backend, "router_info": router_info,
            "oracle_note": "oracle uses ground truth; upper bound only" if "oracle" in methods else None,
            "wall_clock_s": time.perf_counter() - t_start,
            "latency_scope": "per-query retrieval/generation/verification; EXCLUDES index build and model load",
            "env": {"python": sys.version.split()[0], "platform": platform.platform()},
            "synthetic_data": ds.synthetic,
            "index": {"vector_build_s": store.vindex.build_time_s, "vector_bytes": store.vindex.size_bytes(),
                      "graph": store.kg.stats(), "graph_bytes": store.kg.size_bytes()}}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("Saved %d records to %s", len(records), out)
    return run_id, out
