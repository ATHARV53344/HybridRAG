import pytest

from src.data.fixture import load_fixture
from src.indexing.store import build_store
from src.retrieval.dispatcher import Retrievers

RCFG = {"top_k": 3, "kv": 5, "h_max": 2, "alpha": 0.6, "beta": 0.4, "gamma": 0.0, "theta_dup": 0.95, "fusion": "eq4"}


@pytest.fixture(scope="session")
def ds():
    return load_fixture(seed=42)


@pytest.fixture(scope="session")
def store(ds):
    return build_store(ds, {"backend": "tfidf"}, save=False)


@pytest.fixture(scope="session")
def R(store):
    return Retrievers(store, RCFG)
