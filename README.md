# Adaptive Hybrid GraphRAG — research prototype

Implementation of the design in the paper *"Adaptive Hybrid GraphRAG: Query-Aware Selection of Vector, Graph, and Hybrid Retrieval with Evidence Verification for Knowledge-Intensive Question Answering"*.

> **Status honesty box.** All code paths run and are unit-tested on a **synthetic fixture** with a **mock LLM** (pipeline validation only). **No benchmark result exists yet.** Real HotpotQA download, Sentence-Transformer embeddings and a real LLM were *not* exercised in the development sandbox (no Hugging Face / API access). Run them on your machine (below) before writing the Results section. Never quote fixture/mock numbers in the paper.

## What is implemented

| Paper component | Module |
|---|---|
| Ingestion, passage IDs, dedup, splits | `src/data/` |
| Dense index (Sentence-Transformers + FAISS, cosine) | `src/indexing/embeddings.py`, `vector_index.py` |
| Knowledge graph with provenance (NetworkX) | `src/indexing/graph_builder.py`, `entities.py` |
| Vector / graph / hybrid retrieval | `src/retrieval/` |
| Eq. 4 fusion, Eq. 5 proximity, RRF, dedup + budget | `fusion.py`, `graph_retriever.py` |
| Query features, rule router, learned router (Eq. 1–3), oracle | `src/routing/` |
| Generation (+ mock/Anthropic/OpenAI adapters) | `src/generation/` |
| Claim-level verification (Eq. 6–7, accept/qualify/abstain) | `src/verification/` |
| Metrics, aggregation, bootstrap CIs | `src/evaluation/` |
| Baselines, ablations, plots, evaluator report | `src/experiments/` |

Systems compared: `no_retrieval` (B0), `vector_only` (B1), `graph_only` (B2), `hybrid_fixed` (B3), `rule_router` (B4), `adaptive` (P), `adaptive_verified` (P+), `oracle` (O, upper bound only).

## 1. Setup (Windows 11, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env      # then put your API key in .env (only needed for a real LLM)
```

Troubleshooting
- *"running scripts is disabled"*: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen PowerShell.
- *FAISS install fails*: use Python 3.11/3.12 (not 3.13+ on older wheels), then `pip install faiss-cpu`. If it still fails, `conda install -c conda-forge faiss-cpu`.
- *torch / sentence-transformers is huge or fails*: `pip install torch --index-url https://download.pytorch.org/whl/cpu` first, or set `embedding.backend: tfidf` (weaker offline fallback, **not** the paper's setting).
- *Dependency conflicts*: delete `.venv` and reinstall in a fresh environment.
- *Hugging Face blocked/offline*: `prepare` exits with an error; it never fabricates data.

## 2. Workflow

```powershell
python -m pytest -q                                                        # 59 offline tests
python -m src.experiments.run_baselines --config configs/default.yaml --max-samples 5   # smoke test (fixture + mock LLM)

python -m src.experiments.run_baselines --config configs/openai_smoke.yaml --max-samples 5  # real-LLM check: 5 HotpotQA questions, OPENAI_API_KEY, ~$0.014 on gpt-4.1-nano

python -m src.data.prepare --dataset hotpotqa --max-samples 200            # needs internet
python -m src.indexing.build --dataset hotpotqa --config configs/experiments.yaml
python -m src.experiments.run_baselines --config configs/experiments.yaml  # real run (set provider/model + API key)
python -m src.experiments.run_ablations --config configs/experiments.yaml
python -m src.experiments.aggregate_results
python -m src.experiments.generate_plots
python -m src.experiments.make_report_md                                   # results/REPORT_DRAFT.md
```

Run order matters: **dev run first** (`--split dev --max-samples 50`), freeze configs, **then** the test split. `--split test` is the default for reporting.

### First real-LLM smoke run (2026-10-21, gpt-4.1-nano, 5 HotpotQA test questions)

Pipeline check only — n=5, CIs span most of [0, 1]; **do not report these numbers.**

| System | EM | F1 | Recall@k | AllGold@k | median ms | input tokens/q |
|---|---|---|---|---|---|---|
| B0 No retrieval | 0.00 | 0.07 | — | — | 743 | 109 |
| B1 Vector-only | 0.40 | 0.53 | 0.80 | 0.80 | 930 | 800 |
| B2 Graph-only | 0.20 | 0.33 | 0.40 | 0.00 | 987 | 1132 |
| B3 Fixed hybrid | 0.40 | 0.53 | 0.90 | 0.80 | 1168 | 842 |
| B4 Rule router | 0.40 | 0.57 | 0.80 | 0.60 | 1145 | 938 |
| P Adaptive | 0.40 | 0.53 | 0.90 | 0.80 | 1283 | 842 |
| P+ Adaptive+Verify | 0.40 | 0.53 | 0.90 | 0.80 | 16372 | 4708 |

What it showed: every stage runs against a real API with 0 failed calls; graph-only is weakest
(rule-based entity extraction); the learned router sent all 5 to hybrid (12 of its 15 training labels were
hybrid); verification costs ~14x the latency and ~5.6x the tokens of answering and changed no answer here.
Two of the "wrong" answers are scoring artefacts ("Filmmaker" vs gold "director", "British" vs "a British
composer"). Raw records: `results/raw/openai-smoke_20261021-234142/`.

### 100-question run (2026-10-22, gpt-4.1-nano, sbert embeddings)

`configs/openai_test100.yaml --max-samples 100`: 334 sampled questions → 167 train / 66 dev for the router,
**100 test questions** scored (83 bridge, 17 comparison). 2,804 API calls, 0 failures, **$0.22**, 5.4 min
with `concurrency: 8`. Single run, one seed, one cheap model: preliminary, not final paper numbers.

| System | EM | F1 (95% CI) | Recall@k | AllGold@k | median ms | input tokens/q |
|---|---|---|---|---|---|---|
| B0 No retrieval | 0.23 | 0.32 (0.23–0.40) | — | — | 754 | 100 |
| B1 Vector-only | 0.40 | 0.51 (0.42–0.61) | 0.78 | 0.58 | 944 | 833 |
| B2 Graph-only | 0.21 | 0.28 (0.21–0.36) | 0.26 | 0.05 | 963 | 1312 |
| B3 Fixed hybrid | 0.32 | 0.45 (0.36–0.54) | 0.67 | 0.40 | 974 | 1016 |
| B4 Rule router | 0.36 | 0.48 (0.39–0.57) | 0.68 | 0.46 | 938 | 974 |
| P Adaptive (learned) | 0.39 | 0.52 (0.43–0.61) | 0.78 | 0.58 | 925 | 833 |
| P+ Adaptive+Verify | 0.38 | 0.49 (0.40–0.58) | 0.78 | 0.58 | 11835 | 3868 |
| O Oracle (upper bound) | 0.42 | 0.58 (0.49–0.67) | 0.77 | 0.59 | 954 | 860 |

What it shows:
- **Vector-only is the best fixed route.** Graph-only is far behind (it finds every supporting passage for
  only 5% of questions — the capitalised-phrase entity extractor is the bottleneck), and adding graph
  evidence (hybrid) *lowers* F1 vs vector (−0.06, CI crosses 0).
- **The learned router collapsed to "always vector".** 147 of its 167 training labels were vector (Eq. 3 picks
  the cheapest route within ε), so it routed all 100 test questions to vector. Adaptive − vector F1 = +0.003
  (CI −0.01…+0.02): the tiny gap is API nondeterminism, not routing. The rule router does worse than vector
  because it sends 57% of questions to the weaker graph/hybrid routes.
- **There is headroom.** Oracle routing reaches F1 0.58 (+0.06 over vector), mostly from sending ~13% of
  questions to hybrid/graph. A router that finds those questions would help; the current features don't.
- **Verification costs more than it gives here.** It withheld 12% of answers; when it flags an answer as
  wrong it is right 86% of the time, but it catches only 26% of wrong answers, so F1 drops 0.52 → 0.49
  (CI −0.06…−0.003 vs vector) at ~12× latency and ~4.6× tokens.
- Latencies were measured with 8 questions in flight; compare them only with runs at the same `concurrency`.

Raw records: `results/raw/baselines-test100_20261022-130834/`.

## 3. Showing metrics to evaluators in the terminal

```powershell
python -m src.experiments.report                    # full report of the latest run
python -m src.experiments.report --compact          # smaller, for a projector
python -m src.experiments.report --examples 3       # + per-question traces (route, evidence, verdict)
python -m src.experiments.report --run <run_id>     # a specific saved run
python -m src.demo "Which country was the author of The Silent Orchard born in?" --config configs/default.yaml   # live walk-through
```

`run_baselines` prints the same report automatically when it finishes. The report reads only saved files in `results/`, shows every system side by side (EM/F1 with CIs, Recall@k / AllGold@k / MRR / nDCG, latency split into routing / retrieval / generation / verification, tokens, routing behaviour vs the majority-class baseline, verification outcomes, paired comparisons) and prints warning banners for synthetic data, mock LLM and small *n*.

Suggested 5-minute evaluator flow: `pytest` (credibility) → `report` (tables) → `demo` on one multi-hop and one simple question (mechanism) → open `results/figures/`.

## 4. Where things are saved
`data/processed/<dataset>/` (examples, passages, manifest) · `artifacts/` (FAISS index, graph) · `results/raw/<run_id>/` (per-question `records.jsonl` + `run_meta.json`) · `results/processed/<run_id>/` (`summary.json/csv`) · `results/figures/`, `results/tables/`.

## 5. Docs
`docs/ARCHITECTURE.md` · `docs/IMPLEMENTATION_DECISIONS.md` (every deviation/assumption) · `docs/EXPERIMENT_PROTOCOL.md` · `docs/DATASET_NOTES.md` · `docs/RESULTS_INTERPRETATION.md`.

## 6. Known limitations
Rule-based entity extraction (capitalised phrases) — weaker than the paper's LLM-assisted extraction; NQ adapter is a documented stub; MuSiQue adapter untested; the learned router is a logistic regression over heuristic features; verification uses the generator's LLM (not independent) or a lexical proxy offline; single-run, subset evaluation.
