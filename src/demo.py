"""Live demo for evaluators: watch one question go through route -> retrieve -> generate -> verify.

    python -m src.demo "Which country was the author of The Silent Orchard born in?" --config configs/default.yaml
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import load_config
from src.experiments.common import load_or_prepare
from src.generation.answer_generator import AnswerGenerator
from src.generation.llm_client import get_llm
from src.indexing.store import load_store
from src.retrieval.dispatcher import Retrievers
from src.routing.adaptive_router import RuleRouter
from src.verification.evidence_verifier import EvidenceVerifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg, con = load_config(a.config), Console()
    ds = load_or_prepare(cfg)
    store = load_store(ds, cfg["embedding"])
    R = Retrievers(store, cfg["retrieval"])
    llm = get_llm(cfg["generation"])
    dec = RuleRouter(R.graph.link_entities).route(a.question)
    con.print(Panel(f"[bold]{a.question}[/bold]", title="Question"))
    con.print(Panel(f"route = [bold green]{dec.route}[/bold green]\nreason: {dec.reason}\n"
                    f"features: { {k: round(v, 2) for k, v in dec.features.items()} }  (router overhead {dec.overhead_s*1000:.2f} ms)",
                    title="1. Query-aware router"))
    res = R.run(dec.route, a.question)
    t = Table(title=f"2. Retrieved evidence ({dec.route}, {res.latency_s*1000:.1f} ms)")
    for col in ("rank", "passage", "score", "graph path"):
        t.add_column(col)
    for i, e in enumerate(res.evidence, 1):
        t.add_row(str(i), f"[{e.pid}] {e.title}", f"{e.score:.3f}", " → ".join(e.meta.get("path") or []) or "—")
    con.print(t)
    g = AnswerGenerator(llm).generate(a.question, res.evidence)
    con.print(Panel(f"answer: [bold]{g.answer or '(none)'}[/bold]   cites: {g.citations}\n"
                    f"{('ERROR: ' + g.error) if g.error else ''}latency {g.latency_s*1000:.1f} ms, tokens in/out {g.tokens_in}/{g.tokens_out}",
                    title=f"3. Generation ({llm.provider}/{llm.model})"))
    v = EvidenceVerifier(cfg["verification"], llm, "lexical" if llm.provider == "mock" else "llm").verify(a.question, g, res.evidence)
    t = Table(title=f"4. Verification: [bold]{v.decision.upper()}[/bold]  (κ = {v.kappa:.2f}, {v.latency_s*1000:.1f} ms)")
    for col in ("claim", "support u_i", "supported", "evidence"):
        t.add_column(col)
    for c in v.claims:
        t.add_row(c.claim, f"{c.support:.2f}", "yes" if c.supported else "NO", ", ".join(c.evidence_ids))
    con.print(t)


if __name__ == "__main__":
    main()
