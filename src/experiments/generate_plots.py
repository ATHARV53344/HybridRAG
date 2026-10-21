"""CLI: python -m src.experiments.generate_plots [--run RUN_ID]
Plots/tables are generated from saved summary files only (no hard-coded values)."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import ROOT  # noqa: E402
from src.evaluation.evaluate import aggregate_run, latest_run, load_run, save_summary, summary_to_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    run_dir = (ROOT / "results" / "raw" / a.run) if a.run else latest_run("baselines")
    recs, meta = load_run(run_dir)
    summary = aggregate_run(recs, meta)
    save_summary(summary, meta["run_id"])
    df = summary_to_frame(summary)
    fig_dir, tab_dir = ROOT / "results" / "figures", ROOT / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    tag = meta["run_id"]
    note = "SYNTHETIC/MOCK - pipeline check only" if meta.get("synthetic_data") or meta["llm_provider"] == "mock" else ""

    def bar(cols, title, fname, ylabel):
        ax = df.set_index("method")[cols].plot(kind="bar", figsize=(8, 4.5))
        ax.set_title(title + (f"\n{note}" if note else ""))
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{tag}_{fname}.png", dpi=150)
        plt.close()

    bar(["EM", "F1"], "Answer quality by system", "answer_quality", "score")
    bar(["Recall@k", "AllGold@k", "MRR", "nDCG@k"], "Retrieval quality by system", "retrieval_quality", "score")
    bar(["lat_retrieval_median", "lat_generation_median", "lat_verification_median"], "Median latency components",
        "latency", "seconds")
    v = df[df.method == "adaptive_verified"]
    if not v.empty:
        bar(["abstain_rate"], "Verification: abstain rate", "verification", "rate")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df.lat_total_median, df.F1)
    for _, r in df.iterrows():
        ax.annotate(r.method, (r.lat_total_median, r.F1), fontsize=8)
    ax.set_xlabel("median total latency (s)")
    ax.set_ylabel("F1")
    ax.set_title("Answer quality vs latency" + (f"\n{note}" if note else ""))
    plt.tight_layout()
    plt.savefig(fig_dir / f"{tag}_quality_vs_latency.png", dpi=150)
    plt.close()
    df.to_csv(tab_dir / f"{tag}_comparison.csv", index=False)
    (tab_dir / f"{tag}_comparison.md").write_text(df.round(4).to_markdown(index=False)
                                                   if _has_tabulate() else df.round(4).to_string(index=False), encoding="utf-8")
    abl = ROOT / "results" / "processed" / "ablations.csv"
    if abl.exists():
        import pandas as pd

        d = pd.read_csv(abl)
        piv = d.pivot_table(index="ablation", columns="method", values="f1")
        ax = piv.plot(kind="bar", figsize=(9, 4.5))
        ax.set_title("Ablations: F1")
        plt.tight_layout()
        plt.savefig(fig_dir / "ablations_f1.png", dpi=150)
        plt.close()
    print(f"Figures -> {fig_dir}\nTables  -> {tab_dir}")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
