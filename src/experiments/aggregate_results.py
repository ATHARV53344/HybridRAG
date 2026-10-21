"""CLI: python -m src.experiments.aggregate_results [--run RUN_ID]  (default: latest baselines run)"""
from __future__ import annotations

import argparse

from src.config import ROOT
from src.evaluation.evaluate import aggregate_run, latest_run, load_run, save_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run id (folder in results/raw); default latest baselines run")
    a = ap.parse_args()
    run_dir = (ROOT / "results" / "raw" / a.run) if a.run else latest_run("baselines")
    recs, meta = load_run(run_dir)
    out = save_summary(aggregate_run(recs, meta), meta["run_id"])
    print(f"Wrote {out/'summary.json'} and {out/'summary.csv'}")


if __name__ == "__main__":
    main()
