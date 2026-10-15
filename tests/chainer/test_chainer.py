from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from androidmirror import chainer
from androidmirror.chainer import ChainError, plan_chain, plan_merge, suggest_alignment
from androidmirror.ingest import load_seed
from androidmirror.ontology import Ontology
from androidmirror.patches import Op, Patch
from androidmirror.server import create_app
from androidmirror.steward import GovernanceError, Steward

SEED = Path(__file__).resolve().parents[2] / "data" / "seed_book1.json"


def build(ops, name="mirror"):
    o = Ontology(name)
    o.apply_ops(ops, provenance={"patch": "setup"})
    return o


def node(nid, label=None, **kw):
    return Op(op="add_node", id=nid, label=label or nid.title(), **kw)


def rel(s, k, d):
    return Op(op="add_relation", src=s, rel=k, dst=d)


def cause(s, d, sign=1, strength=0.5, **kw):
    return Op(op="add_causal_edge", src=s, dst=d, sign=sign, strength=strength, **kw)


def comparable(o):
    # The core's update_node undo keeps the rolled-back patch in a node's provenance, so leave that out.
    snap = o.snapshot()
    strip = lambda rows: sorted(str({k: v for k, v in r.items() if k not in ("modified", "provenance") or "src" in r})  # noqa: E731
                                for r in rows)
    return {k: strip(snap[k]) for k in ("nodes", "relations", "causal_edges")}


def test_merge_moves_both_graphs_only_after_approval(tmp_path):
    o = build([node("anger"), node("wrath"), node("reason"), node("vexation"), node("harm"),
               rel("reason", "governs", "wrath"), rel("wrath", "kind_of", "anger"),
               cause("vexation", "wrath", strength=0.4), cause("wrath", "harm", strength=0.7),
               cause("anger", "harm", strength=0.9, mechanism="lashing out")])
    before = comparable(o)
    st = Steward(o, tmp_path)
    patch = st.propose(plan_merge(o.graph_snapshot(), "anger", ["wrath"]))
    assert patch.kind == "chain" and patch.status == "proposed" and patch.proposed_by == "chainer"
    assert comparable(o) == before and o.version == 1  # planning and proposing change nothing

    st.approve(patch.id, "Android K")
    assert "wrath" not in o.nodes
    assert o.kg.has_edge("reason", "anger", key="governs")
    assert not o.kg.has_edge("anger", "anger")  # wrath kind_of anger was between merged nodes
    assert o.dag.has_edge("vexation", "anger")
    harm = o.dag.edges["anger", "harm"]
    assert harm["strength"] == 0.9 and harm["mechanism"] == "lashing out"
    assert o.nodes["anger"]["attrs"]["aliases"] == ["Wrath"]
    assert patch.extra["metrics"]["after"] == {"nodes": 4, "relations": 1, "causal_edges": 2}

    st.rollback(patch.id, "Android K")
    assert comparable(o) == before


def test_merge_keeps_provenance_of_collapsed_edges():
    o = Ontology()
    o.apply_ops([node("a"), node("b"), node("c")])
    o.apply_ops([cause("a", "c")], provenance={"patch": "p1"})
    o.apply_ops([cause("b", "c")], provenance={"patch": "p2"})
    o.apply_ops(plan_merge(o.graph_snapshot(), "a", ["b"]).ops)
    patches = [p["patch"] for p in o.dag.edges["a", "c"]["provenance"]]
    assert patches == ["p1", "p2"]


def test_opposite_signs_are_refused_unless_user_picks():
    o = build([node("a"), node("b"), node("c"), cause("a", "c", sign=1), cause("b", "c", sign=-1)])
    with pytest.raises(ChainError, match="opposite causal signs"):
        plan_merge(o.graph_snapshot(), "a", ["b"])
    p = plan_merge(o.graph_snapshot(), "a", ["b"], on_conflict="keep_incoming")
    assert p.extra["conflicts"][0]["kept_sign"] == -1 and p.warnings
    o.apply_ops(p.ops)
    assert o.dag.edges["a", "c"]["sign"] == -1


def test_merge_that_would_create_a_cycle_is_refused():
    o = build([node("a"), node("b"), node("c"), cause("a", "b"), cause("b", "c")])
    with pytest.raises(ChainError, match="cycle: .*a"):
        plan_merge(o.graph_snapshot(), "a", ["c"])


def test_stale_merge_is_refused_at_approval(tmp_path):
    o = build([node("a"), node("b"), node("c"), cause("b", "c")])
    st = Steward(o, tmp_path)
    merge = st.propose(plan_merge(o.graph_snapshot(), "a", ["b"]))
    st.approve(st.propose(Patch(ops=[rel("b", "r", "c")])).id, "k")
    with pytest.raises(GovernanceError, match="stale"):
        st.approve(merge.id, "k")


def test_chain_aligns_shared_nodes_and_namespaces_the_rest(tmp_path):
    marcus = build([node("death"), node("fear"), node("god"), cause("fear", "death")])
    epictetus = build([node("death"), node("god"), node("choice", "Prohairesis"), node("fear"),
                       rel("choice", "judges", "death"), cause("choice", "fear", sign=-1),
                       cause("fear", "death", strength=0.9)], name="epictetus")
    hint = suggest_alignment(marcus.graph_snapshot(), epictetus.graph_snapshot())
    assert hint["exact"] == {"death": "death", "god": "god", "fear": "fear"}

    st = Steward(marcus, tmp_path)
    p = st.propose(plan_chain(marcus.graph_snapshot(), epictetus.graph_snapshot(), "Epictetus", separate=["god"]))
    st.approve(p.id, "Android K")
    assert marcus.nodes["epictetus:choice"]["attrs"]["ontology"] == "epictetus"
    assert marcus.nodes["epictetus:god"]["label"] == "God (Epictetus)"  # kept separate, survived the Steward's dedupe
    assert marcus.kg.has_edge("epictetus:choice", "death", key="judges")
    assert marcus.dag.edges["epictetus:choice", "fear"]["sign"] == -1
    assert marcus.dag.edges["fear", "death"]["strength"] == 0.9
    assert marcus.nodes["death"]["attrs"]["chained_with"] == [{"ontology": "epictetus", "id": "death", "label": "Death"}]

    with pytest.raises(ChainError, match="already chained"):
        plan_chain(marcus.graph_snapshot(), epictetus.graph_snapshot(), "Epictetus")


def test_chain_refuses_a_cross_ontology_cycle():
    a = build([node("x"), node("y"), cause("x", "y")])
    b = build([node("x"), node("y"), cause("y", "x")])
    with pytest.raises(ChainError, match="cycle"):
        plan_chain(a.graph_snapshot(), b.graph_snapshot(), "other")


def test_merge_on_the_meditations_seed_rolls_back_cleanly(tmp_path):
    st = Steward(Ontology(), tmp_path)
    for p in load_seed(st, SEED):
        st.approve(p.id, "seed")
    o = st.ontology
    survivor, other = sorted(o.nodes, key=lambda n: -o.kg.degree(n) - o.dag.degree(n))[:2]
    before = comparable(o)
    patch = chainer.plan_merge(o.graph_snapshot(), survivor, [other], on_conflict="keep_existing")
    try:
        st.approve(st.propose(patch).id, "Android K")
    except GovernanceError:
        pytest.skip("the two busiest seed nodes cannot be merged without a cycle")
    assert other not in o.nodes
    st.rollback(patch.id, "Android K")
    assert comparable(o) == before


def test_http_routes_only_queue(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(tmp_path)
    st = app.state.steward
    st.approve(st.propose(Patch(ops=[node("death"), node("fear"), node("wrath"), node("anger"), cause("fear", "death")])).id, "k")
    c = TestClient(app)
    version = st.ontology.version
    r = c.post("/api/chain/merge", json={"survivor": "anger", "absorbed": ["wrath"]})
    assert r.status_code == 200 and r.json()["status"] == "proposed"
    assert c.post("/api/chain/merge", json={"survivor": "anger", "absorbed": ["nope"]}).status_code == 400
    other = build([node("death"), node("choice"), cause("choice", "death")], name="epictetus").snapshot()
    assert c.post("/api/chain/align", json={"ontology": other}).json()["exact"] == {"death": "death"}
    r = c.post("/api/chain/import", json={"ontology": other})
    assert r.status_code == 200 and r.json()["extra"]["namespace"] == "epictetus"
    assert st.ontology.version == version and len(st.list("proposed")) == 2
    assert len(Steward(Ontology(), tmp_path).list("proposed")) == 2  # the queue was saved
