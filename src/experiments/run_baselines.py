"""CLI: python -m src.experiments.run_baselines --config configs/default.yaml --max-samples 5

Runs the main systems on one split with identical examples/config, saves records + metadata, then
prints the evaluator report to the terminal.
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from src.config import load_config
from src.evaluation.evaluate import aggregate_run, load_run, save_summary
from src.evaluation.report import render
from src.experiments.common import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--max-samples", type=int, default=None, help="cap on evaluated examples")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    con = Console()
    with Progress(TextColumn("[cyan]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=con) as prog:
        task = prog.add_task("running", total=None)

        def cb(done, total, method):
            prog.update(task, completed=done, total=total, description=f"{method:<18}")

        run_id, out = run_experiment(cfg, a.split, a.max_samples, a.methods, "baselines", cb)
    recs, meta = load_run(out)
    summary = aggregate_run(recs, meta)
    save_summary(summary, run_id)
    if not a.no_report:
        render(summary, meta, con)
    con.print(f"\nRaw records: {out}\nRe-show anytime: python -m src.experiments.report --run {run_id}")


if __name__ == "__main__":
    main()
