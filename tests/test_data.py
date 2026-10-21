from src.data.fixture import load_fixture
from src.data.preprocessing import chunk_text
from src.data.schema import Dataset, make_passage_id


def test_records_parsed(ds):
    assert ds.synthetic and len(ds.examples) == 12
    e = ds.examples[0]
    assert e.question and e.answer and e.gold_pids
    pids = {p.pid for p in ds.passages}
    assert set(e.gold_pids) <= pids


def test_passage_ids_stable():
    a = make_passage_id("T", "Some  text.")
    assert a == make_passage_id(" t ", "some text.")  # whitespace/case-insensitive
    assert a != make_passage_id("T", "Other text.")
    assert [p.pid for p in load_fixture(1).passages] == [p.pid for p in load_fixture(99).passages]


def test_no_gold_flag_leaks_into_passages(ds):
    assert all(p.meta == {} for p in ds.passages)  # retrievers must not see 'supporting' markers


def test_splits_disjoint_and_deterministic():
    a, b = load_fixture(7), load_fixture(7)
    assert [e.split for e in a.examples] == [e.split for e in b.examples]
    assert {e.split for e in a.examples} == {"train", "dev", "test"}


def test_chunking():
    assert chunk_text("a b c d e f", 0) == ["a b c d e f"]
    assert chunk_text("a b c d e f", 4, 2) == ["a b c d", "c d e f", "e f"]


def test_save_load_roundtrip(ds, tmp_path):
    ds.save(tmp_path, 42, None)
    d2 = Dataset.load(tmp_path)
    assert len(d2.examples) == len(ds.examples) and d2.passages[0].pid == ds.passages[0].pid
