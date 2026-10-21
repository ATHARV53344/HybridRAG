# Experiment protocol

Fill the **[FILL]** fields from `results/raw/<run_id>/run_meta.json` after the real run; do not leave defaults implied.

| Item | Value |
|---|---|
| Dataset / split | HotpotQA distractor, official *validation*, own 50/20/30 train/dev/test partition by seed (test labels are not public) |
| Sample selection | uniform random sample, `--max-samples N`, seed 42; N = **[FILL]** |
| Corpus | union of the context paragraphs of the sampled questions, deduplicated by stable passage ID; no gold answers indexed |
| Preprocessing | whitespace normalisation, no chunking (paragraphs are short), `chunk_text` available |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalised, inner product = cosine, FAISS `IndexFlatIP` |
| Graph | rule-based extraction (see IMPLEMENTATION_DECISIONS #1), h_max = 2 |
| top-k / kv | 5 / 10 (config) |
| Fusion | Eq. 4, α = 0.6, β = 0.4, γ = 0; RRF in ablation |
| Generator | **[FILL model id + run dates]**, temperature 0, identical prompt for every system |
| Verifier | `llm` backend, same model unless a separate one is configured → not independent |
| Router | learned: trained on train split, τ_r tuned on dev, frozen before test |
| Metrics | EM, token-F1 (HotpotQA normalisation); Recall@k, AllGold@k, P@k, MRR, nDCG@k vs gold supporting passages; latency mean/median/p95 (p95 only if n ≥ 20); tokens; verification summary; bootstrap 95% CIs; paired bootstrap vs fixed baselines |
| Failures | kept in the denominator and scored 0; `failure_rate` reported per method; errors stored in each record |
| Latency scope | per-query retrieval/generation/verification after warm start; index build and model load **excluded** (index build time reported separately) |
| Hardware/software | **[FILL CPU/RAM/OS/Python]**; `run_meta.json` records Python/OS/commit |
| Repeats | single run unless repeated; hosted-model nondeterminism is not removed by temperature 0 |

Order of work: smoke test (fixture, mock) → dev run (`--split dev`) to choose α, β, ε, λ, τ → freeze configs, record them → one test-split run → ablations on the same test examples. Never tune on test. Subset results must be described as preliminary and not as full-benchmark performance.
