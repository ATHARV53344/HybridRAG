"""All routing policies live here: fixed, rule-based, learned (Eq. 1-3) and ORACLE (analysis only).

Every router returns a RouteDecision with the chosen route, the features used and a textual reason.
Routers receive only the question string (plus indexes for entity linking) - never answers/gold labels.
The oracle is the single exception and is explicitly flagged `uses_ground_truth=True`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.routing.features import FEATURE_NAMES, extract_features

log = logging.getLogger("router")
ROUTES = ("vector", "graph", "hybrid")


@dataclass
class RouteDecision:
    route: str
    features: dict
    reason: str
    overhead_s: float = 0.0
    probs: dict | None = None
    uses_ground_truth: bool = False


class FixedRouter:
    def __init__(self, route: str):
        if route not in ROUTES:
            raise ValueError(f"Invalid fixed route '{route}'")
        self.fixed = route

    def route(self, question: str, **_) -> RouteDecision:
        return RouteDecision(self.fixed, {}, f"fixed route '{self.fixed}'")


class RuleRouter:
    """Paper's initial rule baseline: simple -> vector, medium -> hybrid, complex -> graph.

    complexity: complex if est_hops >= hop_complex; medium if comparison cue or est_hops >= 2;
    else simple. No linked entities -> vector (graph cannot start without a seed entity).
    """

    def __init__(self, linker=None, hop_complex: int = 3, hop_medium: int = 2):
        self.linker, self.hop_complex, self.hop_medium = linker, hop_complex, hop_medium

    def route(self, question: str, **_) -> RouteDecision:
        t0 = time.perf_counter()
        f = extract_features(question, self.linker)
        if f["n_linked"] == 0:
            r, why = "vector", "no linkable entity -> vector fallback"
        elif f["est_hops"] >= self.hop_complex:
            r, why = "graph", f"complex (est_hops={f['est_hops']:.0f} >= {self.hop_complex})"
        elif f["has_comparison"] or f["est_hops"] >= self.hop_medium:
            r, why = "hybrid", f"medium (comparison={f['has_comparison']:.0f}, est_hops={f['est_hops']:.0f})"
        else:
            r, why = "vector", "simple (single hop, no comparison cue)"
        d = RouteDecision(r, f, why, time.perf_counter() - t0)
        log.info("route=%s reason=%s", r, why)
        return d


class LearnedRouter:
    """p(r|q) = softmax(W phi(q) + b) via multinomial logistic regression (Eq. 1) with
    cost-aware selection (Eq. 2) and low-confidence hybrid fallback (tau_r)."""

    def __init__(self, linker=None, lam: float = 0.0, tau_r: float = 0.5, top1_fn=None):
        self.linker, self.lam, self.tau_r, self.top1_fn = linker, lam, tau_r, top1_fn
        self.model = None
        self.mu = self.sd = None
        self.classes_: list[str] = []
        self.cost = {"vector": 0.0, "graph": 0.5, "hybrid": 1.0}  # replaced by measured costs

    def _vec(self, f: dict) -> np.ndarray:
        return np.array([f[n] for n in FEATURE_NAMES], dtype=float)

    def fit(self, feats: list[dict], labels: list[str], costs: dict[str, float] | None = None) -> None:
        from sklearn.linear_model import LogisticRegression

        if costs:
            self.cost = costs
        X = np.stack([self._vec(f) for f in feats])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        if len(set(labels)) < 2:  # degenerate: constant label -> remember majority
            self.model, self.classes_ = None, [labels[0]]
            return
        self.model = LogisticRegression(max_iter=1000).fit((X - self.mu) / self.sd, labels)
        self.classes_ = list(self.model.classes_)

    def predict_proba(self, f: dict) -> dict[str, float]:
        if self.model is None:
            return {r: (1.0 if r == self.classes_[0] else 0.0) for r in ROUTES}
        p = self.model.predict_proba(((self._vec(f) - self.mu) / self.sd)[None])[0]
        out = {r: 0.0 for r in ROUTES}
        out.update(dict(zip(self.classes_, map(float, p))))
        return out

    def route(self, question: str, **_) -> RouteDecision:
        if not self.classes_:
            raise RuntimeError("LearnedRouter used before fit(). Train on the train split first.")
        t0 = time.perf_counter()
        f = extract_features(question, self.linker, self.top1_fn)
        p = self.predict_proba(f)
        if max(p.values()) < self.tau_r:
            r, why = "hybrid", f"low confidence (max p={max(p.values()):.2f} < tau_r={self.tau_r})"
        else:
            r = max(ROUTES, key=lambda x: p[x] - self.lam * self.cost[x])  # Eq. 2
            why = f"argmax p - lambda*C (lambda={self.lam}), p={ {k: round(v, 2) for k, v in p.items()} }"
        if f["n_linked"] == 0 and r in ("graph",):
            r, why = "vector", why + "; no linkable entity -> vector"
        d = RouteDecision(r, f, why, time.perf_counter() - t0, probs=p)
        log.info("route=%s reason=%s", r, why)
        return d


class OracleRouter:
    """UPPER BOUND ONLY. Chooses the cheapest route within epsilon of the best F1 using ground truth
    (Eq. 3 applied to the evaluated question). Not deployable; never report as the adaptive method."""

    def __init__(self, best_route: dict[str, str]):
        self.best_route = best_route

    def route(self, question: str, eid: str | None = None, **_) -> RouteDecision:
        r = self.best_route.get(eid, "hybrid")
        return RouteDecision(r, {}, "ORACLE (uses ground truth)", 0.0, uses_ground_truth=True)


def derive_labels(f1_by_route: dict[str, float], cost: dict[str, float], eps: float) -> str:
    """Eq. 3: cheapest route whose F1 is within eps of the best F1 among routes."""
    best = max(f1_by_route.values())
    ok = [r for r in ROUTES if r in f1_by_route and f1_by_route[r] >= best - eps]
    return min(ok, key=lambda r: (cost[r], ROUTES.index(r)))
