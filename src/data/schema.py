"""Common dataset schema shared by every dataset adapter."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def make_passage_id(title: str, text: str) -> str:
    """Stable ID: hash of normalized title + text, so identical passages dedupe across examples."""
    h = hashlib.sha1(f"{title.strip().lower()}||{' '.join(text.split()).lower()}".encode("utf-8"))
    return "p_" + h.hexdigest()[:12]


@dataclass
class Passage:
    pid: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Example:
    eid: str
    question: str
    answer: str
    qtype: str  # bridge | comparison | single_hop | unknown
    gold_pids: list[str]  # supporting passage IDs (evaluation ONLY)
    split: str = "test"  # train | dev | test (router split, assigned by seed)


@dataclass
class Dataset:
    name: str
    source: str
    version: str
    examples: list[Example]
    passages: list[Passage]
    synthetic: bool = False

    def manifest(self, seed: int, max_samples: int | None) -> dict:
        splits: dict[str, int] = {}
        for e in self.examples:
            splits[e.split] = splits.get(e.split, 0) + 1
        return {"name": self.name, "source": self.source, "version": self.version,
                "synthetic": self.synthetic, "n_examples": len(self.examples),
                "n_passages": len(self.passages), "splits": splits, "seed": seed,
                "max_samples": max_samples}

    def save(self, out_dir: Path, seed: int, max_samples: int | None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "examples.jsonl", "w", encoding="utf-8") as f:
            for e in self.examples:
                f.write(json.dumps(asdict(e)) + "\n")
        with open(out_dir / "passages.jsonl", "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(asdict(p)) + "\n")
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest(seed, max_samples), f, indent=2)

    @staticmethod
    def load(out_dir: Path) -> "Dataset":
        man = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        ex = [Example(**json.loads(l)) for l in (out_dir / "examples.jsonl").read_text(encoding="utf-8").splitlines() if l]
        ps = [Passage(**json.loads(l)) for l in (out_dir / "passages.jsonl").read_text(encoding="utf-8").splitlines() if l]
        return Dataset(man["name"], man["source"], man["version"], ex, ps, man.get("synthetic", False))
