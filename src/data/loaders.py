"""Dataset adapters. All return a common `Dataset`.

HotpotQA is the primary adapter. MuSiQue and Natural Questions adapters are provided but
UNTESTED in the development sandbox (no Hugging Face access there) - check field names on first run.

Split policy: official HotpotQA test labels are not public, so we sample from the official
*validation* (dev, distractor setting) split and carve our own train/dev/test partitions by seed
(see preprocessing.assign_splits). This is a limitation to state in the paper.
"""
from __future__ import annotations

import random

from .fixture import load_fixture
from .preprocessing import assign_splits, build_passage, dedupe_passages
from .schema import Dataset, Example


def load_hotpotqa(max_samples: int, seed: int) -> Dataset:
    from datasets import load_dataset  # lazy import

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:max_samples])
    examples, passages = [], []
    for i in idx:
        r = ds[i]
        sup_titles = set(r["supporting_facts"]["title"])
        gold = []
        ctx = r["context"]  # HF schema: {"title": [..], "sentences": [[..], ..]}
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            p = build_passage(title, " ".join(sents))
            passages.append(p)
            if title in sup_titles:
                gold.append(p.pid)
        examples.append(Example(eid=r["id"], question=r["question"], answer=r["answer"],
                                qtype=r.get("type", "unknown"), gold_pids=sorted(set(gold))))
    assign_splits(examples, seed)
    return Dataset("hotpotqa", "hf:hotpotqa/hotpot_qa (distractor, validation)", "hf-default",
                   examples, dedupe_passages(passages))


def load_musique(max_samples: int, seed: int) -> Dataset:
    """MuSiQue-Ans via HF 'dgslibisey/MuSiQue' (validation). Fields: paragraphs[{title, paragraph_text, is_supporting}]."""
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:max_samples])
    examples, passages = [], []
    for i in idx:
        r = ds[i]
        gold = []
        for para in r["paragraphs"]:
            p = build_passage(para["title"], para["paragraph_text"])
            passages.append(p)
            if para.get("is_supporting"):
                gold.append(p.pid)
        examples.append(Example(eid=r["id"], question=r["question"], answer=r["answer"],
                                qtype="multi_hop", gold_pids=sorted(set(gold))))
    assign_splits(examples, seed)
    return Dataset("musique", "hf:dgslibisey/MuSiQue (validation)", "hf-default", examples, dedupe_passages(passages))


def load_nq(max_samples: int, seed: int) -> Dataset:
    """Natural Questions adapter - SIMPLIFIED.

    Uses the HF 'google-research-datasets/natural_questions' (streamed). Each example's gold
    passage is its annotated long answer. Supporting-fact metrics are NOT comparable to HotpotQA.
    Implement/verify field mapping before relying on it.
    """
    raise NotImplementedError(
        "NQ adapter is a documented stub: the NQ HF schema (token-level document + annotations) needs a "
        "dedicated preprocessing step. See docs/DATASET_NOTES.md.")


def load_dataset_by_name(name: str, max_samples: int, seed: int) -> Dataset:
    name = name.lower()
    if name == "fixture":
        return load_fixture(seed, max_samples)
    if name == "hotpotqa":
        return load_hotpotqa(max_samples, seed)
    if name == "musique":
        return load_musique(max_samples, seed)
    if name in ("nq", "naturalquestions"):
        return load_nq(max_samples, seed)
    raise ValueError(f"Unknown dataset '{name}'")
