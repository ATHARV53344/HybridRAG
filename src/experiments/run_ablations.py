"""CLI: python -m src.experiments.run_ablations --config configs/experiments.yaml

Each ablation changes ONE thing relative to the base config and is run on the same examples.
Mapped to the paper's Table 4.3 (only the ones this prototype can run):
  A1  router type: rule vs learned vs oracle        -> methods rule_router / adaptive / oracle (base run)
  A4  fusion: Eq.4 vs RRF                            -> retrieval.fusion = rrf
  A5  graph proximity: hop distance vs PPR           -> retrieval.proximity = ppr
  A6  de-duplication off                             -> retrieval.theta_dup = 1.01
  A7  verifier off vs on                             -> adaptive vs adaptive_verified
  A3  lambda sweep                                   -> router.lambda_cost in {0, 0.3, 1.0}
  graph retrieval disabled                           -> methods vector_only only (reference)
"""
from __future__ import annotations

import argparse
import copy
import json

from rich.console import Console
from rich.table import Table

from src.config import ROOT, load_config
from src.evaluation.evaluate import aggregate_run, load_run
from src.experiments.common import run_experiment

ABLATIONS = {
    "base": {},
    "A4_fusion_rrf": {"retrieval": {"fusion": "rrf"}},
    "A5_proximity_ppr": {"retrieval": {"proximity": "ppr"}},
    "A6_no_dedup": {"retrieval": {"theta_dup": 1.01}},
    "A3_lambda_0.3": {"router": {"lambda_cost": 0.3}},
    "A3_lambda_1.0": {"router": {"lambda_cost": 1.0}},
}
METHODS = ["vector_only", "hybrid_fixed", "adaptive", "adaptive_verified"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="subset of ablation ids")
    a = ap.parse_args()
    base = load_config(a.config)
    con = Console()
    rows = []
    for name, patch in ABLATIONS.items():
        if a.only and name not in a.only:
            continue
        cfg = copy.deepcopy(base)
        for sect, kv in patch.items():
            cfg[sect].update(kv)
        con.print(f"[cyan]running ablation {name}[/cyan]")
        run_id, out = run_experiment(cfg, a.split, a.max_samples, METHODS, f"ablation-{name}")
        recs, meta = load_run(out)
        s = aggregate_run(recs, meta)
        for m, ms in s["methods"].items():
            rows.append({"ablation": name, "method": m, "n": ms["n"], "f1": ms["f1"], "em": ms["em"],
                         "recall_at_k": ms["recall_at_k"], "all_gold_at_k": ms["all_gold_at_k"],
                         "median_total_s": ms["lat_total"]["median"], "abstain_rate": ms["abstain_rate"], "run_id": run_id})
    out_dir = ROOT / "results" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_dir / "ablations.csv", index=False)
    t = Table(title="Ablations (same examples; each row changes one factor vs base)")
    for col in ("ablation", "method", "n", "F1", "EM", "Recall@k", "AllGold@k", "median ms"):
        t.add_column(col)
    for r in rows:
        t.add_row(r["ablation"], r["method"], str(r["n"]), f"{r['f1']:.3f}", f"{r['em']:.3f}",
                  f"{r['recall_at_k']:.3f}" if r["recall_at_k"] is not None else "n/a",
                  f"{r['all_gold_at_k']:.3f}" if r["all_gold_at_k"] is not None else "n/a",
                  f"{r['median_total_s'] * 1000:.1f}")
    con.print(t)
    con.print(f"Saved {out_dir/'ablations.csv'}")


if __name__ == "__main__":
    main()
