"""Passage normalization, dedup, chunking and deterministic split assignment."""
from __future__ import annotations

import random
import re

from .schema import Example, Passage, make_passage_id


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 0, overlap: int = 0) -> list[str]:
    """Word-based chunking. chunk_size=0 disables chunking (HotpotQA paragraphs are already short)."""
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, max(len(words), 1), step)]


def dedupe_passages(passages: list[Passage]) -> list[Passage]:
    seen: dict[str, Passage] = {}
    for p in passages:
        seen.setdefault(p.pid, p)
    return list(seen.values())


def assign_splits(examples: list[Example], seed: int, frac=(0.5, 0.2, 0.3)) -> None:
    """Deterministic train/dev/test assignment. Test is used only for final reporting."""
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_train, n_dev = int(frac[0] * n), int(frac[1] * n)
    for rank, i in enumerate(idx):
        examples[i].split = "train" if rank < n_train else "dev" if rank < n_train + n_dev else "test"


def build_passage(title: str, text: str, meta: dict | None = None) -> Passage:
    title, text = normalize_text(title), normalize_text(text)
    return Passage(pid=make_passage_id(title, text), title=title, text=text, meta=meta or {})
