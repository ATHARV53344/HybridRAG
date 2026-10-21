from src.indexing.graph_builder import KnowledgeGraph


def test_graph_has_nodes_edges_and_provenance(store, ds):
    g = store.kg.g
    assert g.number_of_nodes() > 0 and g.number_of_edges() > 0
    valid = {p.pid for p in ds.passages}
    for _, d in g.nodes(data=True):
        assert d["passages"] and d["passages"] <= valid
    for _, _, d in g.edges(data=True):
        assert d["passages"] and d["passages"] <= valid


def test_graph_built_from_passages_not_gold(ds):
    kg = KnowledgeGraph.build([])
    assert kg.g.number_of_nodes() == 0  # nothing is invented without passages


def test_bridge_passage_reached_via_traversal(R, ds):
    ex = ds.examples[0]  # Silent Orchard -> Mira Valdoren -> Eldoria
    res = R.graph.retrieve(ex.question, 5)
    assert set(ex.gold_pids) <= set(res.pids)
    assert any(len(e.meta["path"]) > 1 for e in res.evidence)  # at least one multi-hop path


def test_missing_entities_handled(R):
    res = R.graph.retrieve("what is the meaning of life?", 3)
    assert res.evidence == [] and res.fallback == "no_entities"


def test_graph_route_falls_back_to_vector(R):
    res = R.run("graph", "what is the meaning of life?")
    assert res.evidence and res.fallback == "no_entities->vector"


def test_hop_limit_respected(store):
    from src.retrieval.graph_retriever import GraphRetriever

    r = GraphRetriever(store, h_max=0)
    res = r.retrieve("Which country was the author of The Silent Orchard born in?", 10)
    assert all(e.meta["hops"] == 0 for e in res.evidence)


def test_graph_save_load(store, tmp_path):
    p = tmp_path / "g.pkl"
    store.kg.save(p)
    assert KnowledgeGraph.load(p).g.number_of_nodes() == store.kg.g.number_of_nodes()
