"""Answer EM / token-F1 following the official HotpotQA normalisation (lower-case, strip punctuation
and articles, collapse whitespace). yes/no/noanswer special-case follows the official script."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str | None, gold: str | None) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold)) if (pred or "").strip() else 0.0


def token_f1(pred: str | None, gold: str | None) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return 0.0
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        if p != g:
            return 0.0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(pt), n / len(gt)
    return 2 * prec * rec / (prec + rec)
