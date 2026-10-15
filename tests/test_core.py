from pathlib import Path

import pytest

from androidmirror import analysis, query
from androidmirror.corpus import split_sections
from androidmirror.ingest import ingest_chunks, load_seed
from androidmirror.ontology import Ontology, OntologyError
from androidmirror.patches import Op, Patch, Source
from androidmirror.steward import GovernanceError, Steward

SEED = Path(__file__).resolve().parent.parent / "data" / "seed_book1.json"


def nodes(*names):
    return [Op(op="add_node", id=n, label=n.title()) for n in names]


def test_cycle_rejected_and_state_unchanged():
    o = Ontology()
    o.apply_ops(nodes("a", "b") + [Op(op="add_causal_edge", src="a", dst="b")])
    with pytest.raises(OntologyError):
        o.apply_ops([Op(op="add_causal_edge", src="b", dst="a")])
    assert list(o.dag.edges) == [("a", "b")]
    assert o.version == 1


def test_inverse_restores_everything():
    o = Ontology()
    o.apply_ops(nodes("a", "b", "c") + [Op(op="add_relation", src="a", rel="knows", dst="b"),
                                        Op(op="add_causal_edge", src="b", dst="c", sign=-1, strength=0.9)])
    before = o.snapshot()
    inv = o.apply_ops([Op(op="remove_node", id="b"), Op(op="update_node", id="a", label="Alpha", attrs={"x": 1})])
    assert "b" not in o.nodes
    o.apply_ops(inv)
    after = o.snapshot()
    for key in ("nodes", "relations", "causal_edges"):
        assert sorted(map(str, before[key])) == sorted(map(str, after[key]))


def test_steward_requires_named_approval(tmp_path):
    st = Steward(Ontology(), tmp_path)
    p = st.propose(Patch(ops=nodes("a")))
    with pytest.raises(GovernanceError):
        st.approve(p.id, "  ")
    st.approve(p.id, "Android K")
    assert st.get(p.id).approval.approver == "Android K"
    assert "a" in st.ontology.nodes
    reloaded = Steward(Ontology.load(tmp_path / "ontology.json"), tmp_path)
    assert reloaded.get(p.id).status == "approved" and "a" in reloaded.ontology.nodes


def test_dedupe_and_quote_check(tmp_path):
    st = Steward(Ontology(), tmp_path, corpus_text="Begin the morning by saying to thyself.")
    st.approve(st.propose(Patch(ops=[Op(op="add_node", id="reason", label="Reason")])).id, "k")
    p = st.propose(Patch(ops=[Op(op="add_node", id="reason_2", label="reason"), Op(op="add_node", id="b", label="B"),
                              Op(op="add_relation", src="reason_2", rel="r", dst="b")],
                         source=Source(quote="not in the book")))
    assert [o.op for o in p.ops] == ["add_node", "add_relation"] and p.ops[1].src == "reason"
    assert any("quote not found" in w for w in p.warnings)


def test_rollback(tmp_path):
    st = Steward(Ontology(), tmp_path)
    p = st.approve(st.propose(Patch(ops=nodes("a", "b") + [Op(op="add_causal_edge", src="a", dst="b")])).id, "k")
    st.rollback(p.id, "k")
    assert not st.ontology.nodes and st.get(p.id).status == "rolled_back"


def test_seed_approves_in_any_order(tmp_path):
    st = Steward(Ontology(), tmp_path)
    queued = load_seed(st, SEED)
    assert len(queued) == 9
    for p in reversed(queued):  # last first: patches must be self-contained
        st.approve(p.id, "k")
    o = st.ontology
    assert {"marcus_aurelius", "vexation", "dignity"} <= set(o.nodes)
    assert o.dag.has_edge("vexation", "cooperation")


def test_analysis_on_seed(tmp_path):
    st = Steward(Ontology(), tmp_path)
    for p in load_seed(st, SEED):
        st.approve(p.id, "k")
    o = st.ontology
    effects = {e["id"]: e["effect"] for e in analysis.intervene(o, "morning_preparation", 1.0)}
    assert effects["vexation"] < 0 < effects["cooperation"]  # less vexation, more cooperation
    assert effects["living_according_to_nature"] > 0
    br = analysis.breaker(o)
    assert any(c["id"] == "marcus_aurelius" for c in br["cut_points"])
    m = analysis.metrics(o)
    assert all(0 <= v <= 1 for v in m.values())
    assert analysis.important_nodes(o)[0]["id"]
    q = query.ask(o, "What did he learn from his mother?")
    assert "his_mother" in q["nodes"] and any("Piety" in f for f in q["facts"])


def test_split_sections():
    text = ("*** START OF THE PROJECT GUTENBERG EBOOK ***\nIntro I. not a section\n\nTHE FIRST BOOK\n\n"
            "I. From my grandfather Verus.\n\nII. From my father.\n\nTHE SECOND BOOK\n\nI. Begin the morning.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK ***")
    chunks = split_sections(text)
    assert [(c.book, c.section) for c in chunks] == [(1, "I"), (1, "II"), (2, "I")]
    assert chunks[2].locator == "Book II, I" and chunks[2].text == "Begin the morning."


def test_ingest_with_fake_extractor(tmp_path):
    st = Steward(Ontology(), tmp_path)
    chunks = split_sections("THE FIRST BOOK\n\nI. a\n\nII. b\n")

    def fake(text, locator, known):
        if text == "a":
            return [{"kind": "descriptive", "quote": "a", "rationale": "", "nodes": [
                {"id": "x", "label": "X", "type": "concept"}], "relations": [], "causal_edges": []}]
        assert "x: X" in known  # pending nodes are offered for reuse
        return [{"kind": "causal", "quote": "b", "rationale": "", "nodes": [{"id": "y", "label": "Y", "type": "concept"}],
                 "relations": [], "causal_edges": [{"src": "x", "dst": "y", "sign": 1, "strength": 0.5,
                                                    "confidence": 0.5, "mechanism": ""}]}]

    p1, p2 = ingest_chunks(st, chunks, extract=fake)
    st.approve(p2.id, "k")  # self-contained: carries node x
    st.approve(p1.id, "k")  # x now exists: its add is dropped, not an error
    assert st.ontology.dag.has_edge("x", "y")
