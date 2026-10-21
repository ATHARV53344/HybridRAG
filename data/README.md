# data/

- `raw/` — downloads (git-ignored). `processed/<dataset>/` — `examples.jsonl`, `passages.jsonl`, `manifest.json` written by `python -m src.data.prepare`.
- `fixtures/` — reserved; the synthetic fixture lives in `src/data/fixture.py` and is generated in code. It is **synthetic** and must never be mixed into benchmark results.
