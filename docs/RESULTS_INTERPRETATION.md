# How to interpret and report results

**Metrics**
- *EM / F1*: answer correctness vs gold. F1 gives partial credit. An abstention or API failure scores 0, so abstaining lowers F1 — always read it next to coverage (`abstain_rate`).
- *Recall@k / AllGold@k*: Recall = share of gold passages in the top-k; AllGold = share of questions with **all** gold passages retrieved — the multi-hop requirement, usually the headline retrieval number. P@k is bounded by k vs the number of gold passages (2 for HotpotQA), so low P@k is expected.
- *MRR / nDCG@k*: ranking quality with binary relevance (gold passage = relevant). Similarity alone is never relevance.
- *κ (support score)*: fraction of claims entailed by retrieved context. It is **not** truth: right answer from memory can score low, wrong context can score high.
- *Detect-wrong precision/recall*: how well "abstain" aligns with answers whose F1 < 0.5. Claim-level truth is unavailable, so this is answer-level.
- *Latency*: per-query, excluding index build/model load. Report mean, median, and p95 only when n ≥ 20. API latency varies run to run.

**Meaningful comparisons**: same examples, same generator/prompt, same k. Use the paired bootstrap rows, not eyeballed means. "CI excludes 0" on a few dozen questions is weak evidence; with overlapping CIs write "no measurable difference".

**Quality vs cost (RQ2/RQ4)**: compare each system to *fixed hybrid* (is quality retained?) and to *vector-only* (what did you pay?). Include router and verifier overhead in the totals. Report tokens as an API-cost proxy. The oracle shows headroom; if it barely beats fixed hybrid, adaptive routing cannot help much under this design — report that as a legitimate negative result.

**Negative / inconclusive results**: write them plainly ("adaptive did not improve F1 over fixed hybrid; difference −0.01, 95% CI [−0.05, +0.03], n=…"). Check misrouted questions and graph extraction errors in `records.jsonl` for explanations, labelled as hypotheses.

**Needs further validation before claiming**: any gain on a subset generalising to the full benchmark or other corpora; verifier effectiveness (needs manual audit and an independent judge); router accuracy beyond the majority-class baseline (H4); anything measured with the mock LLM or fixture; graph quality (audit a sample of edges by hand).

**Do not** write "improves", "outperforms" or "reduces hallucination" unless a paired CI supports it on a real run, and never edit the paper to imply experiments were done before they ran.
