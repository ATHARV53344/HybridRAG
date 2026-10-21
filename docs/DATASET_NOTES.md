# Dataset notes

- **HotpotQA** (primary): `hotpotqa/hotpot_qa`, config `distractor`, split `validation`, via Hugging Face `datasets`. Each example has 10 paragraphs (2 gold + 8 distractors) and supporting-fact titles; a passage is gold if its title is a supporting-fact title. Test answers are not public, so we partition the validation sample ourselves. **Not yet run against the real download** (blocked in the dev sandbox) — on first run, check `data/processed/hotpotqa/manifest.json` and that `gold_pids` are non-empty.
- **MuSiQue**: adapter for HF `dgslibisey/MuSiQue` (validation) using `paragraphs[].is_supporting`. Untested; verify field names.
- **Natural Questions**: documented stub (raises `NotImplementedError`). NQ needs HTML/token-span preprocessing to build passages and has no multi-hop supporting-fact annotation, so supporting-evidence metrics are not comparable to HotpotQA.
- **Fixture** (`src/data/fixture.py`): 12 invented questions for tests/smoke runs; manifest has `synthetic: true`; reports print a red banner. Never mix with benchmark results.
- Pooled corpus: all paragraphs from all sampled questions are indexed together, so retrieval must find the right paragraphs among other questions' paragraphs (harder than the per-question distractor setting; document when comparing to published numbers).
