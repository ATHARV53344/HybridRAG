import json

from src.experiments.common import run_experiment
from src.evaluation.evaluate import aggregate_run, load_run
from src.evaluation.report import render
from rich.console import Console
import io

CFG = {
    "seed": 42, "output_dir": "results_test",
    "dataset": {"name": "fixture", "max_samples": 12, "data_dir": "data_test"},
    "embedding": {"backend": "tfidf", "model": "x"},
    "retrieval": {"top_k": 3, "kv": 5, "h_max": 2, "alpha": 0.6, "beta": 0.4, "gamma": 0.0, "theta_dup": 0.95,
                  "fusion": "eq4", "context_budget_tokens": 600},
    "router": {"mode": "learned", "lambda_cost": 0.0, "tau_r": 0.5, "epsilon": 0.02},
    "generation": {"provider": "mock", "model": "mock", "temperature": 0.0},
    "verification": {"tau_u": 0.5, "tau_a": 0.8, "tau_b": 0.4},
    "methods": ["no_retrieval", "vector_only", "graph_only", "hybrid_fixed", "rule_router", "adaptive",
                "adaptive_verified", "oracle"],
}


def test_full_pipeline_on_fixture(tmp_path, monkeypatch):
    import shutil
    from src.config import ROOT
    try:
        run_id, out = run_experiment(CFG, split="test", tag="pytest")
        recs, meta = load_run(out)
        assert meta["synthetic_data"] is True and meta["llm_provider"] == "mock"
        methods = {r["method"] for r in recs if not r.get("auxiliary")}
        assert methods == set(CFG["methods"])
        n = meta["n_eval_examples"]
        assert all(sum(1 for r in recs if r["method"] == m and not r.get("auxiliary")) == n for m in methods)
        required = {"eid", "question", "gold_answer", "predicted_answer", "route", "retrieved_pids", "gold_pids",
                    "retrieval_scores", "f1", "em", "verification", "retrieval_s", "generation_s",
                    "verification_s", "error"}
        assert required <= set(recs[0])
        summ = aggregate_run(recs, meta)
        assert summ["methods"]["no_retrieval"]["f1"] == 0.0  # mock has no parametric knowledge
        assert all(summ["methods"][m]["metric_denominator"] == n for m in methods)
        assert meta["router_info"]["label_counts"]
        # oracle must be flagged and never worse than any fixed route on the F1 it optimises
        orc = [r for r in recs if r["method"] == "oracle"]
        assert all(r["oracle"] for r in orc)
        # Eq.3 picks the cheapest route within epsilon of the best F1 per question, and the mock is
        # deterministic, so oracle mean F1 must be within epsilon of the best fixed route.
        eps = CFG["router"]["epsilon"]
        best_fixed = max(summ["methods"][m]["f1_original"] for m in ("vector_only", "graph_only", "hybrid_fixed"))
        assert summ["methods"]["oracle"]["f1_original"] >= best_fixed - eps - 1e-9
        buf = io.StringIO()
        render(summ, meta, Console(file=buf, width=120))
        assert "SYNTHETIC FIXTURE" in buf.getvalue() and "MOCK LLM" in buf.getvalue()
        json.dumps(summ, default=str)
    finally:
        for d in ("results_test", "data_test", "artifacts"):
            p = ROOT / d
            if d != "artifacts" and p.exists():
                shutil.rmtree(p, ignore_errors=True)


def test_failed_api_calls_not_scored_correct(tmp_path, monkeypatch):
    from src.generation.answer_generator import AnswerGenerator
    from src.generation.llm_client import LLMClient, LLMResponse
    from src.pipeline import Pipeline
    from src.routing.adaptive_router import FixedRouter
    from src.data.fixture import load_fixture
    from src.indexing.store import build_store
    from src.retrieval.dispatcher import Retrievers

    class Down(LLMClient):
        provider, model = "down", "down"

        def complete(self, s, u, max_tokens=300):
            return LLMResponse(error="connection refused")

    ds = load_fixture()
    st = build_store(ds, {"backend": "tfidf"}, save=False)
    R = Retrievers(st, CFG["retrieval"])
    pipe = Pipeline(R, AnswerGenerator(Down()), None, 3)
    rec = pipe.run("vector_only", ds.examples[0], FixedRouter("vector"))
    assert rec["error"] and rec["f1"] == 0.0 and rec["em"] == 0.0 and rec["predicted_answer"] == ""
