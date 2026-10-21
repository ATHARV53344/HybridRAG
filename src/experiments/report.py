"""CLI: python -m src.experiments.report [--run RUN_ID] [--compact] [--examples N]"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from src.config import ROOT
from src.evaluation.evaluate import aggregate_run, latest_run, load_run, save_summary
from src.evaluation.report import PRETTY, render


def show_examples(recs: list[dict], n: int, console: Console) -> None:
    """Per-question traces so evaluators can see routing decisions and evidence."""
    pick = [r for r in recs if r["method"] == "adaptive_verified"] or [r for r in recs if r["method"] == "adaptive"]
    for r in pick[:n]:
        body = (f"[bold]Q:[/bold] {r['question']}\n[bold]Gold:[/bold] {r['gold_answer']}   "
                f"[bold]Pred:[/bold] {r['predicted_answer'] or '(withheld)'}  (F1={r['f1']:.2f})\n"
                f"[bold]Route:[/bold] {r['route']} — {r['route_reason']}\n"
                f"[bold]Retrieved:[/bold] {r['retrieved_pids']}\n[bold]Gold passages:[/bold] {r['gold_pids']}")
        v = r.get("verification")
        if v:
            body += f"\n[bold]Verification:[/bold] {v['decision']} (κ={v['kappa']:.2f})"
        console.print(Panel(body, title=f"example {r['eid']}", border_style="blue"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--examples", type=int, default=0, help="also print N per-question traces")
    a = ap.parse_args()
    run_dir = (ROOT / "results" / "raw" / a.run) if a.run else latest_run("baselines")
    recs, meta = load_run(run_dir)
    summary = aggregate_run(recs, meta)
    save_summary(summary, meta["run_id"])
    con = Console()
    render(summary, meta, con, compact=a.compact)
    if a.examples:
        show_examples(recs, a.examples, con)


if __name__ == "__main__":
    main()
