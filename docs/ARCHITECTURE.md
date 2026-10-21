# Architecture

```
question ─► Router (features φ(q): entities, length, cues, ĥ) ─► route ∈ {vector, graph, hybrid}
              │ fixed | rule | learned (Eq.1-3) | oracle (analysis only)
              ▼
 ┌─────────────── Retrievers ────────────────────────────────────────────┐
 │ vector : embed q → FAISS top-k (cosine)                               │
 │ graph  : link entities → bounded BFS (h_max) → provenance passages    │
 │          ranked by g = 1/(1+dist)   [fallback → vector if no entity]  │
 │ hybrid : both → Eq.4 (or RRF) → dedup (θ_dup) → budget B → top-k      │
 └───────────────────────────────────────────────────────────────────────┘
              ▼ Evidence(pid, text, score, meta{vec/graph/norm/fused, path, triples})
 AnswerGenerator (LLM adapter: mock | anthropic | openai) → JSON {answer, citations, sufficient, facts}
              ▼
 EvidenceVerifier: claims → u_i = max_d ENT(d,c_i) → κ → accept | qualify | abstain
              ▼
 record (JSONL): route+reason, retrieved ids/scores, answer, metrics, verification, per-stage latency, tokens, error
```

Layers (paper Sec. 3.1) map to packages: ingestion `src/data`, dual index `src/indexing`, retrieval control `src/routing` + `src/retrieval`, generation/verification `src/generation` + `src/verification`. Retrieval code never imports the LLM module. Ground truth (`gold_pids`, answers) is read only in `Pipeline.run` *after* the system output is produced, and in `evaluation/`.

Artifacts: FAISS index + embeddings + meta (`artifacts/indexes/`), pickled NetworkX graph (`artifacts/graphs/`).
