"""CLI: python -m src.experiments.make_report_md [--run RUN_ID]
Writes results/REPORT_DRAFT.md from saved result files. Missing values -> [RESULT NOT AVAILABLE]."""
from __future__ import annotations

import argparse

from src.config import ROOT
from src.evaluation.evaluate import aggregate_run, latest_run, load_run, save_summary

NA = "[RESULT NOT AVAILABLE]"


def f(x, nd=3):
    return NA if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    run_dir = (ROOT / "results" / "raw" / a.run) if a.run else latest_run("baselines")
    recs, meta = load_run(run_dir)
    s = aggregate_run(recs, meta)
    save_summary(s, meta["run_id"])
    M = s["methods"]
    L = [f"# Draft results report — run `{meta['run_id']}`", ""]
    if meta.get("synthetic_data") or meta["llm_provider"] == "mock":
        L += ["> **WARNING:** this run used synthetic fixture data and/or the mock LLM. It validates the pipeline only; "
              "do NOT cite these numbers in the paper.", ""]
    L += ["## 1. Experimental setup", "",
          f"- Dataset: {meta['dataset']['name']} ({meta['dataset']['source']}), split `{s['split']}`, "
          f"n = {s['n_eval_examples']} evaluated examples out of {meta['dataset']['n_examples']} prepared "
          f"(subset evaluation, not full benchmark)",
          f"- Seed: {meta['seed']}; embedder: {s['embedder']}; LLM: {s['llm']}; verifier backend: {meta['verifier_backend']}",
          f"- top-k: {meta['config']['retrieval']['top_k']}; fusion: {meta['config']['retrieval']['fusion']}; "
          f"h_max: {meta['config']['retrieval']['h_max']}",
          f"- Git commit: {meta.get('git_commit') or NA}; Python {meta['env']['python']}; {meta['env']['platform']}",
          f"- Latency scope: {meta['latency_scope']}", "",
          "## 2. Baseline comparison (answer quality)", "", "| System | n | EM | F1 | F1 95% CI | failures |",
          "|---|---|---|---|---|---|"]
    for m, x in M.items():
        L.append(f"| {m} | {x['n']} | {f(x['em'])} | {f(x['f1'])} | [{f(x['f1_ci95'][0],2)}, {f(x['f1_ci95'][1],2)}] | {x['n_failed']} |")
    L += ["", "## 3. Retrieval metrics", "", "| System | Recall@k | AllGold@k | P@k | MRR | nDCG@k |", "|---|---|---|---|---|---|"]
    for m, x in M.items():
        if m != "no_retrieval":
            L.append(f"| {m} | {f(x['recall_at_k'])} | {f(x['all_gold_at_k'])} | {f(x['precision_at_k'])} | {f(x['mrr'])} | {f(x['ndcg_at_k'])} |")
    L += ["", "## 4. Latency and cost (median seconds per query; mean tokens)", "",
          "| System | route | retrieve | generate | verify | total median | total mean | tokens in |", "|---|---|---|---|---|---|---|---|"]
    for m, x in M.items():
        L.append(f"| {m} | {f(x['lat_routing']['median'],4)} | {f(x['lat_retrieval']['median'],4)} | {f(x['lat_generation']['median'],4)} | "
                 f"{f(x['lat_verification']['median'],4)} | {f(x['lat_total']['median'],4)} | {f(x['lat_total']['mean'],4)} | {f(x['tokens_in_mean'],0)} |")
    v = M.get("adaptive_verified", {}).get("verification")
    L += ["", "## 5. Verification (RQ3)", ""]
    if v:
        L += [f"- accept/qualify/abstain: {f(v['accept_rate'])}/{f(v['qualify_rate'])}/{f(v['abstain_rate'])}; mean kappa {f(v['mean_kappa'])}",
              f"- accuracy before: {f(v['accuracy_before'])}; on answered: {f(v['accuracy_on_answered'])}",
              f"- wrong-answer detection precision/recall: {f(v['detect_wrong_precision'])}/{f(v['detect_wrong_recall'])}; "
              f"FP={v['false_positives']}, FN={v['false_negatives']}"]
    else:
        L.append(NA)
    r = s.get("routing") or {}
    L += ["", "## 6. Routing (RQ2)", ""]
    if r:
        L.append(f"- derived-label distribution {r['label_distribution']}; majority-class accuracy {f(r['majority_class_acc'],2)}")
        for m in ("rule_router", "adaptive"):
            if m in r:
                L.append(f"- {m} agreement with derived label: {f(r[m]['accuracy'],2)} (n={r[m]['n']})")
    else:
        L.append(NA)
    L += ["", "## 7. Ablations", "", "Run `python -m src.experiments.run_ablations` and cite `results/processed/ablations.csv`. "
          + (NA if not (ROOT / 'results/processed/ablations.csv').exists() else "See that file."), "",
          "## 8. Observed trade-offs / error analysis / limitations", "",
          "- [AUTHOR TO WRITE after inspecting `records.jsonl`: misrouted questions, graph extraction errors, verifier FP/FN]",
          "- Subset evaluation with a single seed; CIs are bootstrap over questions; no claim of full-benchmark performance.",
          "- Verification measures support by retrieved context, not truth; generator and verifier may share one model.", "",
          "## 9. Research questions", "",
          "| RQ | Status |", "|---|---|"]
    for rq in ("RQ1 quality across strategies", "RQ2 routing quality/cost trade-off", "RQ3 verification effect", "RQ4 overheads"):
        L.append(f"| {rq} | [AUTHOR TO JUDGE from the tables above; leave 'unresolved' if CIs overlap or n is small] |")
    out = ROOT / "results" / "REPORT_DRAFT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
