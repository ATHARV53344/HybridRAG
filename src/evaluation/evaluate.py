"""Aggregate per-question records into machine-readable summaries (JSON + CSV)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT
from src.evaluation.latency_metrics import bootstrap_ci, paired_bootstrap, summarize_latency
from src.evaluation.verification_metrics import verification_summary
from src.routing.adaptive_router import ROUTES, derive_labels
from src.routing.train import measured_costs

FIXED = {"vector_only": "vector", "graph_only": "graph", "hybrid_fixed": "hybrid"}


def load_run(run_dir: Path) -> tuple[list[dict], dict]:
    recs = [json.loads(l) for l in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l]
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    return recs, meta


def latest_run(prefix: str | None = None, root: str = "results") -> Path:
    base = ROOT / root / "raw"
    runs = sorted([p for p in base.glob("*") if p.is_dir() and (not prefix or p.name.startswith(prefix))],
                  key=lambda p: p.stat().st_mtime) if base.exists() else []
    if not runs:
        raise SystemExit(f"No runs found in {base}. Run: python -m src.experiments.run_baselines --config configs/default.yaml")
    return runs[-1]


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else None


def summarize_method(recs: list[dict], k_label: str = "k") -> dict:
    n = len(recs)
    failed = [r for r in recs if r.get("error")]
    f1s = [r["f1"] for r in recs]
    ems = [r["em"] for r in recs]
    out = {
        "n": n, "n_failed": len(failed), "failure_rate": len(failed) / n if n else None,
        "metric_denominator": n,  # failed examples are INCLUDED (scored 0), never silently dropped
        "em": _mean(ems), "em_ci95": bootstrap_ci(ems), "f1": _mean(f1s), "f1_ci95": bootstrap_ci(f1s),
        "f1_original": _mean([r["f1_original"] for r in recs]),
        "recall_at_k": _mean([r["recall_at_k"] for r in recs]), "all_gold_at_k": _mean([r["all_gold_at_k"] for r in recs]),
        "precision_at_k": _mean([r["precision_at_k"] for r in recs]), "mrr": _mean([r["mrr"] for r in recs]),
        "ndcg_at_k": _mean([r["ndcg_at_k"] for r in recs]),
        "lat_routing": summarize_latency([r["routing_s"] for r in recs]),
        "lat_retrieval": summarize_latency([r["retrieval_s"] for r in recs]),
        "lat_generation": summarize_latency([r["generation_s"] for r in recs]),
        "lat_verification": summarize_latency([r["verification_s"] for r in recs]),
        "lat_total": summarize_latency([r["total_s"] for r in recs]),
        "tokens_in_mean": _mean([r["tokens_in"] for r in recs]), "tokens_out_mean": _mean([r["tokens_out"] for r in recs]),
        "route_distribution": dict(Counter(r["route"] for r in recs)),
        "fallback_rate": _mean([1.0 if r.get("fallback") else 0.0 for r in recs]),
        "abstain_rate": _mean([1.0 if r.get("abstained") else 0.0 for r in recs]),
    }
    vrecs = [{"decision": r["verification"]["decision"], "kappa": r["verification"]["kappa"], "f1": r["f1_original"]}
             for r in recs if r.get("verification")]
    out["verification"] = verification_summary(vrecs) if vrecs else None
    return out


def routing_accuracy(by_method: dict[str, list[dict]], eps: float) -> dict:
    """Agreement of chosen route with the Eq.3-derived label computed from the fixed-route runs (evaluation only)."""
    if not all(m in by_method for m in FIXED):
        return {}
    costs = measured_costs({FIXED[m]: by_method[m] for m in FIXED})
    eids = [r["eid"] for r in by_method["vector_only"]]
    idx = {m: {r["eid"]: r for r in by_method[m]} for m in FIXED}
    label = {e: derive_labels({FIXED[m]: idx[m][e]["f1_original"] for m in FIXED}, costs, eps) for e in eids}
    out = {"label_distribution": dict(Counter(label.values())), "majority_class_acc":
           max(Counter(label.values()).values()) / len(eids), "costs": costs}
    for m in ("rule_router", "adaptive", "adaptive_verified"):
        if m in by_method:
            hits = [r["route"] == label[r["eid"]] for r in by_method[m]]
            out[m] = {"accuracy": float(np.mean(hits)), "n": len(hits)}
    return out


def paired_comparisons(by_method: dict[str, list[dict]]) -> list[dict]:
    rows = []
    ref_methods = [m for m in ("vector_only", "graph_only", "hybrid_fixed") if m in by_method]
    for m in ("adaptive", "adaptive_verified", "rule_router"):
        if m not in by_method:
            continue
        a = {r["eid"]: r for r in by_method[m]}
        for ref in ref_methods:
            b = {r["eid"]: r for r in by_method[ref]}
            common = [e for e in a if e in b]
            for metric in ("f1", "total_s"):
                rows.append({"a": m, "b": ref, "metric": metric,
                             **paired_bootstrap([a[e][metric] for e in common], [b[e][metric] for e in common])})
    return rows


def aggregate_run(recs: list[dict], meta: dict) -> dict:
    eval_recs = [r for r in recs if not r.get("auxiliary")]
    by_method: dict[str, list[dict]] = {}
    for r in recs:
        by_method.setdefault(r["method"], []).append(r)
    summary = {"run_id": meta["run_id"], "split": meta["split"], "n_eval_examples": meta["n_eval_examples"],
               "synthetic_data": meta.get("synthetic_data"), "llm": f"{meta['llm_provider']}/{meta['llm_model']}",
               "embedder": meta["embedder"], "methods": {}}
    for m in meta["methods"]:
        summary["methods"][m] = summarize_method([r for r in eval_recs if r["method"] == m])
    summary["routing"] = routing_accuracy(by_method, meta["config"]["router"].get("epsilon", 0.02))
    summary["paired"] = paired_comparisons(by_method)
    summary["router_training"] = meta.get("router_info")
    # by question type for each method
    summary["by_qtype"] = {m: {t: _mean([r["f1"] for r in eval_recs if r["method"] == m and r["qtype"] == t])
                               for t in sorted({r["qtype"] for r in eval_recs})} for m in meta["methods"]}
    return summary


def summary_to_frame(summary: dict) -> pd.DataFrame:
    rows = []
    for m, s in summary["methods"].items():
        rows.append({"method": m, "n": s["n"], "failure_rate": s["failure_rate"], "EM": s["em"], "F1": s["f1"],
                     "F1_ci_lo": s["f1_ci95"][0], "F1_ci_hi": s["f1_ci95"][1], "Recall@k": s["recall_at_k"],
                     "AllGold@k": s["all_gold_at_k"], "P@k": s["precision_at_k"], "MRR": s["mrr"], "nDCG@k": s["ndcg_at_k"],
                     "lat_total_mean": s["lat_total"]["mean"], "lat_total_median": s["lat_total"]["median"],
                     "lat_retrieval_median": s["lat_retrieval"]["median"], "lat_generation_median": s["lat_generation"]["median"],
                     "lat_verification_median": s["lat_verification"]["median"], "tokens_in_mean": s["tokens_in_mean"],
                     "abstain_rate": s["abstain_rate"]})
    return pd.DataFrame(rows)


def save_summary(summary: dict, run_id: str, root: str = "results") -> Path:
    out = ROOT / root / "processed" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary_to_frame(summary).to_csv(out / "summary.csv", index=False)
    return out
