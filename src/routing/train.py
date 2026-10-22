"""Router training protocol (no leakage).

1. Run V, G, H (with generation) on the TRAIN split -> per-question F1 and measured cost.
2. Labels y(q) = cheapest route within eps of best F1 (Eq. 3). Only train-split gold answers are used.
3. Fit softmax/logistic router on train features (Eq. 1).
4. Tune tau_r on the DEV split by looking up per-route F1 from the dev runs (no extra generation).
5. Freeze. The test split is never touched here.
"""
from __future__ import annotations

import numpy as np

from src.config import get_logger
from src.data.schema import Example
from src.pipeline import Pipeline
from src.routing.adaptive_router import ROUTES, LearnedRouter, derive_labels, FixedRouter
from src.routing.features import extract_features

log = get_logger("router-train")


def measured_costs(records_by_route: dict[str, list[dict]]) -> dict[str, float]:
    """Ĉ(r) in [0,1]: mean of min-max normalised median latency and median prompt tokens across routes."""
    lat = {r: float(np.median([x["retrieval_s"] + x["generation_s"] for x in recs])) if recs else 0.0
           for r, recs in records_by_route.items()}
    tok = {r: float(np.median([x["tokens_in"] for x in recs])) if recs else 0.0 for r, recs in records_by_route.items()}

    def mm(d):
        lo, hi = min(d.values()), max(d.values())
        return {k: (v - lo) / (hi - lo) if hi > lo else 0.0 for k, v in d.items()}

    ln, tn = mm(lat), mm(tok)
    return {r: 0.5 * ln[r] + 0.5 * tn[r] for r in ROUTES}


def _run_routes(pipe: Pipeline, examples: list[Example]) -> dict[str, list[dict]]:
    return {r: pipe.run_batch(f"train_{r}", examples, FixedRouter(r), verify=False) for r in ROUTES}


def train_router(pipe: Pipeline, linker, train: list[Example], dev: list[Example], rcfg: dict) -> tuple[LearnedRouter, dict]:
    eps = rcfg.get("epsilon", 0.02)
    tr = _run_routes(pipe, train)
    costs = measured_costs(tr)
    feats, labels = [], []
    for i, ex in enumerate(train):
        f1s = {r: tr[r][i]["f1_original"] for r in ROUTES}
        labels.append(derive_labels(f1s, costs, eps))
        feats.append(extract_features(ex.question, linker))
    router = LearnedRouter(linker, rcfg.get("lambda_cost", 0.0), rcfg.get("tau_r", 0.5))
    router.fit(feats, labels, costs)
    info = {"n_train": len(train), "label_counts": {r: labels.count(r) for r in ROUTES}, "costs": costs,
            "epsilon": eps, "lambda": router.lam}
    if dev:
        dv = _run_routes(pipe, dev)
        best_tau, best_score = router.tau_r, -1.0
        for tau in (0.34, 0.4, 0.5, 0.6, 0.7):
            router.tau_r = tau
            sc = np.mean([dv[router.route(ex.question).route][i]["f1_original"] for i, ex in enumerate(dev)])
            if sc > best_score + 1e-9:
                best_tau, best_score = tau, float(sc)
        router.tau_r = best_tau
        info.update(tuned_tau_r=best_tau, dev_f1_at_tau=best_score, n_dev=len(dev))
    log.info("Router trained: %s", info)
    return router, info
