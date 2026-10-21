import math

import pytest

from src.evaluation.answer_metrics import exact_match, normalize_answer, token_f1
from src.evaluation.latency_metrics import bootstrap_ci, paired_bootstrap, summarize_latency
from src.evaluation.retrieval_metrics import all_gold_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from src.evaluation.verification_metrics import verification_summary


def test_normalization():
    assert normalize_answer("The  Eiffel, Tower!") == "eiffel tower"


def test_em_f1_basic():
    assert exact_match("The Eiffel Tower", "eiffel tower") == 1.0
    assert token_f1("big red car", "red car") == pytest.approx(0.8)
    assert token_f1("yes", "no") == 0.0


@pytest.mark.parametrize("pred", [None, "", "   "])
def test_empty_predictions_safe(pred):
    assert exact_match(pred, "x") == 0.0 and token_f1(pred, "x") == 0.0
    assert exact_match(pred, None) == 0.0


def test_retrieval_metrics():
    ret, gold = ["a", "b", "c", "d"], ["b", "d"]
    assert recall_at_k(ret, gold, 2) == 0.5 and recall_at_k(ret, gold, 4) == 1.0
    assert all_gold_at_k(ret, gold, 2) == 0.0 and all_gold_at_k(ret, gold, 4) == 1.0
    assert precision_at_k(ret, gold, 2) == 0.5
    assert mrr(ret, gold) == 0.5
    assert 0 < ndcg_at_k(ret, gold, 4) < 1
    assert ndcg_at_k(["b", "d"], gold, 2) == pytest.approx(1.0)


def test_retrieval_metrics_empty():
    assert math.isnan(recall_at_k(["a"], [], 3))
    assert precision_at_k([], ["a"], 3) == 0.0 and mrr([], ["a"]) == 0.0


def test_latency_summary_small_sample_has_no_p95():
    s = summarize_latency([0.1, 0.2, 0.3])
    assert s["median"] == pytest.approx(0.2) and s["p95"] is None
    assert summarize_latency([])["n"] == 0
    assert summarize_latency(list(range(30)))["p95"] is not None


def test_bootstrap_and_paired():
    lo, hi = bootstrap_ci([1.0] * 10)
    assert lo == hi == 1.0
    r = paired_bootstrap([1, 1, 1, 1], [0, 0, 0, 0])
    assert r["mean_diff"] == 1.0 and r["ci95"][0] > 0


def test_verification_summary_counts():
    recs = [{"decision": "accept", "kappa": 1.0, "f1": 1.0}, {"decision": "accept", "kappa": 1.0, "f1": 0.0},
            {"decision": "abstain", "kappa": 0.0, "f1": 0.0}, {"decision": "abstain", "kappa": 0.0, "f1": 1.0}]
    s = verification_summary(recs)
    assert s["abstain_rate"] == 0.5 and s["false_positives"] == 1 and s["false_negatives"] == 1
    assert s["detect_wrong_precision"] == 0.5 and s["detect_wrong_recall"] == 0.5
    assert verification_summary([])["n"] == 0
