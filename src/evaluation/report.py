"""Terminal report for evaluators (rich). Reads ONLY saved result files - no hard-coded numbers.

    python -m src.experiments.report                 # latest baselines run
    python -m src.experiments.report --run <run_id>  # specific run
    python -m src.experiments.report --compact       # fits a projector / small terminal
"""
from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PRETTY = {"no_retrieval": "B0 No retrieval", "vector_only": "B1 Vector-only", "graph_only": "B2 Graph-only",
          "hybrid_fixed": "B3 Fixed hybrid", "rule_router": "B4 Rule router", "adaptive": "P  Adaptive",
          "adaptive_verified": "P+ Adaptive+Verify", "oracle": "O  Oracle (upper bound)"}


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ms(x):
    return "n/a" if x is None else f"{x * 1000:.1f}"


def _bar(v, width=10):
    if v is None:
        return ""
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "█" * n + "░" * (width - n)


def _best(summary, key, higher=True, only=None):
    vals = {m: s[key] for m, s in summary["methods"].items() if s.get(key) is not None and (only is None or m in only)}
    if not vals:
        return None
    return (max if higher else min)(vals, key=vals.get)


def render(summary: dict, meta: dict, console: Console | None = None, compact: bool = False) -> None:
    c = console or Console()
    M = summary["methods"]
    deployable = [m for m in M if m not in ("oracle",)]

    # ---- banner ---------------------------------------------------------------------
    head = Text()
    head.append("Adaptive Hybrid GraphRAG — Evaluation Report\n", style="bold cyan")
    head.append(f"run {summary['run_id']}   split={summary['split']}   n={summary['n_eval_examples']}\n")
    head.append(f"dataset {meta['dataset']['name']} ({meta['dataset']['source']})   seed={meta['seed']}\n")
    head.append(f"embedder {summary['embedder']}   LLM {summary['llm']}   verifier={meta.get('verifier_backend')}\n")
    if summary.get("synthetic_data"):
        head.append("SYNTHETIC FIXTURE DATA — pipeline check only, NOT benchmark results\n", style="bold red")
    if summary["llm"].startswith("mock"):
        head.append("MOCK LLM (heuristic reader) — NOT a language model; numbers are not LLM results\n", style="bold yellow")
    if summary["n_eval_examples"] < 100:
        head.append(f"Small sample (n={summary['n_eval_examples']}): results are PRELIMINARY, CIs wide\n", style="yellow")
    head.append("Latency = per-query time; excludes index build and model load.", style="dim")
    c.print(Panel(head, box=box.DOUBLE, border_style="cyan"))

    # ---- 1. answer quality ------------------------------------------------------------
    t = Table(title="1. Answer quality (RQ1)", box=box.SIMPLE_HEAVY, header_style="bold")
    for col in ("System", "n", "EM", "F1", "F1 95% CI", "F1 bar", "fail"):
        t.add_column(col, justify="right" if col != "System" else "left")
    best_f1 = _best(summary, "f1", True, deployable)
    for m, s in M.items():
        style = "green" if m == best_f1 else ("dim" if m == "oracle" else None)
        t.add_row(PRETTY.get(m, m), str(s["n"]), _f(s["em"]), _f(s["f1"]),
                  f"[{_f(s['f1_ci95'][0], 2)}, {_f(s['f1_ci95'][1], 2)}]", _bar(s["f1"]),
                  f"{s['n_failed']}/{s['n']}", style=style)
    c.print(t)
    c.print("[dim]Failures are counted in the denominator (scored 0). Green = best deployable system. "
            "Oracle uses ground truth and is not deployable.[/dim]\n")

    # ---- 2. retrieval ------------------------------------------------------------------
    t = Table(title="2. Retrieval quality — gold = supporting passages", box=box.SIMPLE_HEAVY, header_style="bold")
    for col in ("System", "Recall@k", "AllGold@k", "P@k", "MRR", "nDCG@k", "fallback"):
        t.add_column(col, justify="right" if col != "System" else "left")
    for m, s in M.items():
        if m == "no_retrieval":
            continue
        t.add_row(PRETTY.get(m, m), _f(s["recall_at_k"]), _f(s["all_gold_at_k"]), _f(s["precision_at_k"]),
                  _f(s["mrr"]), _f(s["ndcg_at_k"]), _f(s["fallback_rate"], 2))
    c.print(t)
    c.print("[dim]AllGold@k = fraction of questions where EVERY supporting passage is in the top-k (multi-hop requirement). "
            "fallback = graph found no entity and used vector evidence.[/dim]\n")

    # ---- 3. latency / cost -------------------------------------------------------------
    t = Table(title="3. Latency & token cost (RQ4) — ms per query", box=box.SIMPLE_HEAVY, header_style="bold")
    for col in ("System", "route", "retrieve", "generate", "verify", "TOTAL mean", "TOTAL median", "p95", "tokens in"):
        t.add_column(col, justify="right" if col != "System" else "left")
    for m, s in M.items():
        p95 = s["lat_total"]["p95"]
        t.add_row(PRETTY.get(m, m), _ms(s["lat_routing"]["median"]), _ms(s["lat_retrieval"]["median"]),
                  _ms(s["lat_generation"]["median"]), _ms(s["lat_verification"]["median"]),
                  _ms(s["lat_total"]["mean"]), _ms(s["lat_total"]["median"]),
                  "n<20" if p95 is None else _ms(p95), _f(s["tokens_in_mean"], 0))
    c.print(t)
    idx = meta.get("index", {})
    if idx:
        c.print(f"[dim]Indexing (one-time): vector {idx['vector_build_s']:.2f}s / {idx['vector_bytes']/1e6:.2f} MB;  "
                f"graph {idx['graph']['build_time_s']}s, {idx['graph']['nodes']} nodes, {idx['graph']['edges']} edges, "
                f"{idx['graph_bytes']/1e6:.2f} MB[/dim]\n")

    # ---- 4. quality-vs-cost ----------------------------------------------------------------
    t = Table(title="4. Quality vs cost (RQ2) — F1 per second of median latency", box=box.SIMPLE_HEAVY, header_style="bold")
    for col in ("System", "F1", "median ms", "F1 / s"):
        t.add_column(col, justify="right" if col != "System" else "left")
    for m, s in M.items():
        lat = s["lat_total"]["median"]
        t.add_row(PRETTY.get(m, m), _f(s["f1"]), _ms(lat), _f(s["f1"] / lat, 2) if lat and s["f1"] is not None else "n/a")
    c.print(t)

    # ---- 5. routing ---------------------------------------------------------------------------
    r = summary.get("routing") or {}
    if r:
        t = Table(title="5. Routing behaviour (RQ2)", box=box.SIMPLE_HEAVY, header_style="bold")
        for col in ("System", "vector", "graph", "hybrid", "agreement w/ derived label"):
            t.add_column(col, justify="right" if col != "System" else "left")
        n_ex = summary["n_eval_examples"]
        for m in ("rule_router", "adaptive", "adaptive_verified", "oracle"):
            if m in M:
                d = M[m]["route_distribution"]
                acc = r.get(m, {}).get("accuracy")
                t.add_row(PRETTY.get(m, m), *(f"{d.get(x, 0)}/{n_ex}" for x in ("vector", "graph", "hybrid")),
                          _f(acc, 2) if acc is not None else "—")
        c.print(t)
        c.print(f"[dim]Derived-label distribution {r['label_distribution']}; majority-class baseline accuracy "
                f"{r['majority_class_acc']:.2f}. A router must beat this to claim it learned anything (H4).[/dim]\n")

    # ---- 6. verification -------------------------------------------------------------------------
    v = M.get("adaptive_verified", {}).get("verification")
    if v:
        t = Table(title="6. Evidence verification (RQ3)", box=box.SIMPLE_HEAVY, header_style="bold")
        t.add_column("Measure"), t.add_column("Value", justify="right")
        for label, key in (("accept rate", "accept_rate"), ("qualify rate", "qualify_rate"), ("abstain rate", "abstain_rate"),
                           ("mean support score κ", "mean_kappa"), ("accuracy before verification", "accuracy_before"),
                           ("accuracy on answered questions", "accuracy_on_answered"),
                           ("wrong-answer detection precision", "detect_wrong_precision"),
                           ("wrong-answer detection recall", "detect_wrong_recall")):
            t.add_row(label, _f(v.get(key), 3))
        t.add_row("false positives (correct answers withheld)", str(v["false_positives"]))
        t.add_row("false negatives (wrong answers let through)", str(v["false_negatives"]))
        lv = M["adaptive_verified"]["lat_verification"]["median"]
        t.add_row("verification overhead (median ms)", _ms(lv))
        c.print(t)
        c.print("[dim]κ measures support by retrieved context, NOT truth. Verifier and generator may share a model → "
                "not independent. 'wrong' = F1 < 0.5 of the original answer.[/dim]\n")

    # ---- 7. significance -----------------------------------------------------------------------------
    pr = [p for p in summary.get("paired", []) if p.get("n")]
    if pr and not compact:
        t = Table(title="7. Paired comparisons on the same questions (bootstrap 95% CI of mean difference a−b)",
                  box=box.SIMPLE_HEAVY, header_style="bold")
        for col in ("a", "b", "metric", "mean diff", "95% CI", "p", "CI excludes 0?"):
            t.add_column(col, justify="left" if col in ("a", "b", "metric") else "right")
        for p in pr:
            lo, hi = p["ci95"]
            sig = "yes" if (lo > 0 or hi < 0) else "no"
            t.add_row(PRETTY.get(p["a"], p["a"]), PRETTY.get(p["b"], p["b"]), p["metric"],
                      f"{p['mean_diff']:+.4f}", f"[{lo:+.3f}, {hi:+.3f}]", f"{p['p_two_sided']:.3f}", sig)
        c.print(t)
        c.print("[dim]Single run, small n: treat as indicative, not conclusive. total_s differences are in seconds.[/dim]\n")

    # ---- 8. by question type ---------------------------------------------------------------------------
    bq = summary.get("by_qtype", {})
    if bq and not compact:
        types = sorted({t_ for d in bq.values() for t_ in d})
        t = Table(title="8. F1 by question type", box=box.SIMPLE_HEAVY, header_style="bold")
        t.add_column("System")
        for ty in types:
            t.add_column(ty, justify="right")
        for m, d in bq.items():
            t.add_row(PRETTY.get(m, m), *(_f(d.get(ty), 3) for ty in types))
        c.print(t)

    # ---- takeaways (derived from the numbers; no invented claims) ------------------------------------------
    notes = []
    bf = _best(summary, "f1", True, deployable)
    if bf:
        notes.append(f"Highest F1 among deployable systems: {PRETTY.get(bf, bf)} ({_f(M[bf]['f1'])}).")
    if "vector_only" in M and "adaptive" in M:
        d = M["adaptive"]["f1"] - M["vector_only"]["f1"]
        notes.append(f"Adaptive vs vector-only F1 difference: {d:+.3f} (see paired CI above before reading anything into it).")
    if "hybrid_fixed" in M and "adaptive" in M and M["hybrid_fixed"]["lat_total"]["median"]:
        ratio = M["adaptive"]["lat_total"]["median"] / M["hybrid_fixed"]["lat_total"]["median"]
        notes.append(f"Adaptive median latency is {ratio:.2f}× that of fixed hybrid.")
    c.print(Panel("\n".join("• " + n for n in notes) or "—", title="Observations (auto-generated from this run's numbers)",
                  border_style="green"))
