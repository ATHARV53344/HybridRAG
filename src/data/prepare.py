"""CLI: python -m src.data.prepare --dataset hotpotqa --max-samples 200"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ROOT, get_logger
from src.data.loaders import load_dataset_by_name

log = get_logger("prepare")


def processed_dir(dataset: str, data_dir: str = "data") -> Path:
    return ROOT / data_dir / "processed" / dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    try:
        ds = load_dataset_by_name(a.dataset, a.max_samples, a.seed)
    except Exception as e:  # report honestly, do not fake data
        log.error("Could not load dataset '%s': %s", a.dataset, e)
        raise SystemExit(1)
    out = processed_dir(ds.name, a.data_dir)
    ds.save(out, a.seed, a.max_samples)
    log.info("Saved %d examples / %d passages to %s", len(ds.examples), len(ds.passages), out)


if __name__ == "__main__":
    main()
