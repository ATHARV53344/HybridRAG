import inspect

import pytest

from src.routing import adaptive_router, features
from src.routing.adaptive_router import (ROUTES, FixedRouter, LearnedRouter, OracleRouter, RuleRouter,
                                         derive_labels)
from src.routing.features import extract_features


def test_rule_router_returns_valid_route_with_reason(R, ds):
    rr = RuleRouter(R.graph.link_entities)
    for ex in ds.examples:
        d = rr.route(ex.question)
        assert d.route in ROUTES and d.reason and d.features and not d.uses_ground_truth


def test_rule_router_fallback_when_no_entities(R):
    d = RuleRouter(R.graph.link_entities).route("what is the meaning of life?")
    assert d.route == "vector" and "no linkable entity" in d.reason


def test_simple_goes_vector_complex_goes_graph(R):
    rr = RuleRouter(R.graph.link_entities)
    assert rr.route("When was Aerowing Corporation founded?").route == "vector"
    assert rr.route("Who founded the company that acquired the supplier of the part that Zephyr used where Orwick tested?"
                    ).route in ("graph", "hybrid")


def test_router_signature_has_no_ground_truth_inputs():
    # routers receive question text (oracle gets only an id) - no answer / gold arguments exist
    for cls in (RuleRouter, LearnedRouter, FixedRouter):
        params = inspect.signature(cls.route).parameters
        assert not {"answer", "gold", "gold_pids", "supporting"} & set(params)
    src = inspect.getsource(features) + inspect.getsource(RuleRouter) + inspect.getsource(LearnedRouter)
    for banned in ("gold_pids", ".answer", "gold_answer", "supporting_facts"):
        assert banned not in src


def test_oracle_is_flagged():
    d = OracleRouter({"e1": "graph"}).route("q", eid="e1")
    assert d.route == "graph" and d.uses_ground_truth


def test_fixed_router_validates():
    assert FixedRouter("hybrid").route("x").route == "hybrid"
    with pytest.raises(ValueError):
        FixedRouter("bogus")


def test_derive_labels_picks_cheapest_within_eps():
    cost = {"vector": 0.0, "graph": 0.5, "hybrid": 1.0}
    assert derive_labels({"vector": 0.8, "graph": 0.8, "hybrid": 0.81}, cost, 0.02) == "vector"
    assert derive_labels({"vector": 0.2, "graph": 0.9, "hybrid": 0.9}, cost, 0.02) == "graph"


def test_learned_router_requires_fit_and_low_confidence_falls_to_hybrid(R):
    lr = LearnedRouter(R.graph.link_entities)
    with pytest.raises(RuntimeError):
        lr.route("x")
    qs = ["When was Aerowing Corporation founded?", "Which country was the author of The Silent Orchard born in?",
          "What is the capital of Taloria?", "Who wrote Harbor Lights?"]
    lr.fit([extract_features(q, R.graph.link_entities) for q in qs], ["vector", "graph", "vector", "vector"])
    lr.tau_r = 1.01  # impossible confidence -> always hybrid fallback
    assert lr.route(qs[0]).route == "hybrid"
    lr.tau_r = 0.0
    assert lr.route(qs[0]).route in ROUTES


def test_features_are_question_only():
    f = extract_features("Who is older, Mira Valdoren or Lena Ostrow?")
    assert f["has_comparison"] == 1.0 and f["n_entities"] >= 2
