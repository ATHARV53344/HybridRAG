"""Embedding backends.

- `sbert`: Sentence Transformers (the paper's setting). Needs `pip install sentence-transformers`.
- `tfidf`: stateless hashed bag-of-words fallback (sklearn HashingVectorizer). Offline, deterministic,
  weaker than neural embeddings. For smoke tests / machines without torch. NOT the reported setting.

All embeddings are L2-normalised float32, so inner product == cosine similarity.
"""
from __future__ import annotations

import threading

import numpy as np


class Embedder:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.name = f"tfidf-hash-{dim}"
        self._vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2",
                                      stop_words="english", lowercase=True, ngram_range=(1, 2))

    def encode(self, texts: list[str]) -> np.ndarray:
        x = self._vec.transform(texts).astype(np.float32).toarray()
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (x / norms).astype(np.float32)


class SbertEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # explicit, actionable error
            raise ImportError("sentence-transformers is not installed. Run `pip install sentence-transformers` "
                              "or set embedding.backend: tfidf in the config.") from e
        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._lock = threading.Lock()  # the torch model segfaults under concurrent calls (pipeline workers > 1)

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=False).astype(np.float32)


def get_embedder(cfg: dict) -> Embedder:
    backend = cfg.get("backend", "sbert")
    if backend == "tfidf":
        return HashingEmbedder()
    if backend == "sbert":
        return SbertEmbedder(cfg["model"])
    raise ValueError(f"Unknown embedding backend '{backend}'")
