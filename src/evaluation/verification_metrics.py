"""Verification metrics computed against ANSWER correctness (F1 >= wrong_thr counts as correct).

Caveat to state in the paper: claim-level ground truth is unavailable, so these measure how well the
verifier's accept/abstain decision tracks answer correctness, not claim-level truth.
"""
from __future__ import annotations


def verification_summary(records: list[dict], correct_f1: float = 0.5) -> dict:
    """records: dicts with keys decision, kappa, f1 (of the ORIGINAL answer), has_answer."""
    recs = [r for r in records if r.get("decision") not in (None, "skipped")]
    n = len(recs)
    if n == 0:
        return {"n": 0}
    accepted = [r for r in recs if r["decision"] in ("accept", "qualify")]
    abstained = [r for r in recs if r["decision"] == "abstain"]
    wrong = [r for r in recs if r["f1"] < correct_f1]
    flagged_wrong = [r for r in abstained if r["f1"] < correct_f1]  # abstain on a wrong answer = true positive
    tp, fp = len(flagged_wrong), len(abstained) - len(flagged_wrong)  # fp: abstained on a correct answer
    fn = len(wrong) - tp  # wrong answers that were accepted/qualified
    return {
        "n": n,
        "abstain_rate": len(abstained) / n,
        "qualify_rate": sum(r["decision"] == "qualify" for r in recs) / n,
        "accept_rate": sum(r["decision"] == "accept" for r in recs) / n,
        "mean_kappa": sum(r["kappa"] for r in recs) / n,
        "accuracy_before": sum(r["f1"] >= correct_f1 for r in recs) / n,
        "accuracy_on_answered": (sum(r["f1"] >= correct_f1 for r in accepted) / len(accepted)) if accepted else None,
        "detect_wrong_precision": tp / (tp + fp) if (tp + fp) else None,
        "detect_wrong_recall": tp / (tp + fn) if (tp + fn) else None,
        "false_positives": fp,  # correct answers withheld
        "false_negatives": fn,  # wrong answers let through
    }
