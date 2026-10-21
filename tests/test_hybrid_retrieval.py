import numpy as np
import pytest

from src.retrieval.fusion import dedupe_and_budget, eq4_fuse, minmax, rrf_fuse
from src.retrieval.types import Evidence


def ev(pid, **meta):
    return Evidence(pid, "t", f"text {pid}", 0.0, "x", meta)


def test_minmax_constant_and_empty():
    assert minmax({}) == {}
    assert minmax({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}
    assert minmax({"a": 0.0, "b": 2.0}) == {"a": 0.0, "b": 1.0}


def test_eq4_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        eq4_fuse([ev("a", vec_score=1.0)], [], 0.5, 0.2)


def test_eq4_dedupes_union_and_zero_for_missing_path():
    vec = [ev("a", vec_score=0.9), ev("b", vec_score=0.1)]
    gr = [ev("b", graph_score=1.0), ev("c", graph_score=0.5)]
    out = eq4_fuse(vec, gr, 0.6, 0.4)
    assert sorted(e.pid for e in out) == ["a", "b", "c"]  # b appears once
    by = {e.pid: e for e in out}
    assert by["c"].meta["vec_norm"] == 0.0 and by["a"].meta["graph_norm"] == 0.0
    assert by["b"].score == pytest.approx(0.6 * 0.0 + 0.4 * 1.0)


def test_rrf_merges_and_dedupes():
    out = rrf_fuse([ev("a"), ev("b")], [ev("b"), ev("c")])
    assert [e.pid for e in out][0] == "b" and len({e.pid for e in out}) == 3


def test_dedupe_by_embedding_and_k():
    vecs = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "c": np.array([0.0, 1.0])}
    ranked = [ev("a"), ev("b"), ev("c")]
    out = dedupe_and_budget(ranked, vecs.get, k=5, theta_dup=0.95)
    assert [e.pid for e in out] == ["a", "c"]
    assert len(dedupe_and_budget(ranked, vecs.get, k=1)) == 1


def test_token_budget_keeps_at_least_one():
    vecs = {"a": np.array([1.0, 0.0]), "c": np.array([0.0, 1.0])}
    out = dedupe_and_budget([ev("a"), ev("c")], vecs.get, k=5, budget_tokens=1)
    assert [e.pid for e in out] == ["a"]


def test_hybrid_end_to_end_no_duplicate_pids(R, ds):
    for ex in ds.examples:
        res = R.run("hybrid", ex.question)
        assert len(res.pids) == len(set(res.pids)) <= 3


def test_hybrid_falls_back_without_entities(R):
    res = R.run("hybrid", "what is the meaning of life?")
    assert res.fallback and res.evidence


def test_unknown_route_raises(R):
    with pytest.raises(ValueError):
        R.run("nonsense", "q")
