"""Configuration loading and shared logging setup."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:  # optional .env support
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return cfg


def get_logger(name: str) -> logging.Logger:
    # Console stays quiet by default so the evaluator report is readable; every routing reason is saved in
    # records.jsonl. Set LOG_LEVEL=INFO to see per-decision logs.
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
