from src.indexing.embeddings import get_embedder
from src.indexing.vector_index import VectorIndex
from src.retrieval.vector_retriever import VectorRetriever


def test_embedding_dimensions():
    emb = get_embedder({"backend": "tfidf"})
    x = emb.encode(["hello world", "another passage"])
    assert x.shape == (2, emb.dim)


def test_returns_valid_pids_and_sorted_scores(store, ds):
    res = VectorRetriever(store).retrieve(ds.examples[0].question, 4)
    valid = {p.pid for p in ds.passages}
    assert res.pids and set(res.pids) <= valid
    scores = [e.score for e in res.evidence]
    assert scores == sorted(scores, reverse=True)


def test_cached_index_loads_without_reembedding(store, ds, tmp_path):
    store.vindex.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, store.embedder)
    q = ds.examples[0].question
    assert loaded.search(q, 3) == store.vindex.search(q, 3)


def test_stale_cache_is_not_reused_for_a_different_corpus(monkeypatch, tmp_path):
    """Regression: re-preparing with a different sample size must not load an old index/graph."""
    import src.indexing.store as st
    from src.data.fixture import load_fixture

    monkeypatch.setattr(st, "ROOT", tmp_path)
    big, small = load_fixture(42, 12), load_fixture(42, 5)
    st.build_store(big, {"backend": "tfidf"}, save=True)
    loaded = st.load_store(small, {"backend": "tfidf"})  # must rebuild, not reuse
    assert set(loaded.vindex.pids) == {p.pid for p in small.passages}
    assert st.corpus_fingerprint(big.passages) != st.corpus_fingerprint(small.passages)


def test_index_embedder_mismatch_raises(store, tmp_path):
    import pytest

    store.vindex.save(tmp_path)

    class Other:
        name, dim = "different", 3

    with pytest.raises(ValueError):
        VectorIndex.load(tmp_path, Other())
