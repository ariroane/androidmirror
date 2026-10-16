from pathlib import Path

import networkx as nx
import pytest

from androidmirror import breaker
from androidmirror.ingest import load_seed
from androidmirror.ontology import Ontology
from androidmirror.patches import Op
from androidmirror.steward import GovernanceError, Steward

SEED = Path(__file__).resolve().parent.parent / "data" / "seed_book1.json"


def world() -> Ontology:
    """`duty` bridges a civic cluster (rome, senate) and a private one (family, grief)."""
    o = Ontology()
    o.apply_ops([Op(op="add_node", id=n, label=n.title(), type="value") for n in
                 ("duty", "rome", "senate", "family", "grief", "calm")] + [
        Op(op="add_relation", src="rome", rel="has", dst="senate"),
        Op(op="add_relation", src="family", rel="causes", dst="grief"),
        Op(op="add_relation", src="duty", rel="serves", dst="rome"),
        Op(op="add_relation", src="duty", rel="owed_to", dst="family"),
        Op(op="add_causal_edge", src="senate", dst="duty", sign=1, strength=0.8, confidence=0.9),
        Op(op="add_causal_edge", src="duty", dst="grief", sign=-1, strength=0.6, confidence=0.7),
        Op(op="add_causal_edge", src="duty", dst="calm", sign=1, strength=0.2, confidence=0.3),
        Op(op="add_causal_edge", src="grief", dst="calm", sign=-1, strength=0.9, confidence=0.9),
    ])
    return o


def steward(o: Ontology, tmp_path: Path) -> Steward:
    return Steward(o, tmp_path)


def test_breaker_never_writes(monkeypatch):
    o = world()
    before = o.snapshot()

    def boom(*a, **k):
        raise AssertionError("the Breaker must not change the ontology")

    monkeypatch.setattr(Ontology, "apply_ops", boom)
    breaker.split_candidates(o)
    breaker.sever_candidates(o)
    breaker.propose_split(o, "duty")
    breaker.propose_sever(o, "duty", "calm")
    assert breaker.propose(o.graph_snapshot())
    assert o.snapshot() == before


def test_split_candidates_find_the_bridge():
    rows = breaker.split_candidates(world())
    duty = next(r for r in rows if r["id"] == "duty")
    assert len(duty["groups"]) >= 2
    assert rows[0]["id"] == "duty"


def test_auto_split_moves_links_in_both_graphs(tmp_path):
    o = world()
    st = steward(o, tmp_path)
    patch = st.propose(breaker.propose_split(o, "duty"))
    assert patch.kind == "break" and patch.status == "proposed"
    assert o.nodes["duty"]["attrs"].get("archived") is None  # nothing applied yet

    st.approve(patch.id, "Android K")
    parts = o.nodes["duty"]["attrs"]["split_into"]
    assert o.nodes["duty"]["attrs"]["archived"] is True
    assert o.kg.degree("duty") == 0 and o.dag.degree("duty") == 0
    civic = next(p for p in parts if o.kg.has_edge(p, "rome"))
    assert o.dag.has_edge("senate", civic)  # the causal edge followed its descriptive cluster
    private = next(p for p in parts if o.kg.has_edge(p, "family"))
    assert o.dag.has_edge(private, "grief") and not o.dag.has_edge(civic, "grief")
    assert nx.is_directed_acyclic_graph(o.dag)
    assert all(o.nodes[p]["attrs"]["split_from"] == "duty" for p in parts)
    assert all(o.nodes[p]["attrs"]["aliases"] == ["Duty"] for p in parts)


def test_split_with_named_parts_and_copied_link(tmp_path):
    o = world()
    st = steward(o, tmp_path)
    parts = [{"label": "Public duty", "id": "public_duty"}, {"label": "Private duty", "id": "private_duty"}]
    assign = {breaker.edge_key("kg", "duty", "rome", "serves"): "public_duty",
              breaker.edge_key("dag", "duty", "calm"): breaker.ALL}
    patch = st.propose(breaker.propose_split(o, "duty", parts, assign))
    st.approve(patch.id, "Android K")
    assert o.kg.has_edge("public_duty", "rome", key="serves")
    assert o.dag.has_edge("public_duty", "calm") and o.dag.has_edge("private_duty", "calm")
    assert o.dag.edges["public_duty", "calm"]["strength"] == pytest.approx(0.2)


def test_split_rolls_back_exactly(tmp_path):
    o = world()
    st = steward(o, tmp_path)
    before = o.snapshot()
    patch = st.propose(breaker.propose_split(o, "duty"))
    st.approve(patch.id, "Android K")
    st.rollback(patch.id, "Android K")
    after = o.snapshot()
    strip = lambda rows: sorted(str({k: v for k, v in r.items() if k not in ("modified", "provenance")})  # noqa: E731
                                for r in rows)
    for key in ("nodes", "relations", "causal_edges"):
        assert strip(before[key]) == strip(after[key])


def test_split_rejects_unknown_part_and_too_few_parts():
    o = world()
    with pytest.raises(ValueError):
        breaker.propose_split(o, "duty", [{"label": "Only one"}])
    with pytest.raises(ValueError):
        breaker.propose_split(o, "duty", [{"label": "A"}, {"label": "B"}], {"dag:duty|calm": "nope"})
    with pytest.raises(ValueError):
        breaker.propose_split(o, "calm")  # no bridge, so the parts must be named


def test_sever_cuts_both_graphs_and_reports_cost(tmp_path):
    o = world()
    o.apply_ops([Op(op="add_relation", src="calm", rel="follows", dst="duty")])
    st = steward(o, tmp_path)
    patch = st.propose(breaker.propose_sever(o, "duty", "calm"))
    assert patch.extra["breaker"]["influence_lost"] > 0
    st.approve(patch.id, "Android K")
    assert not o.dag.has_edge("duty", "calm") and not o.kg.has_edge("calm", "duty")


def test_sever_warns_when_it_disconnects():
    o = world()
    o.apply_ops([Op(op="add_node", id="forum", label="Forum"), Op(op="add_relation", src="rome", rel="has", dst="forum")])
    patch = breaker.propose_sever(o, "rome", "forum")
    assert patch.extra["breaker"]["disconnects"] is True
    assert patch.warnings
    with pytest.raises(ValueError):
        breaker.propose_sever(world(), "rome", "calm")


def test_sever_candidates_flag_weak_links():
    rows = breaker.sever_candidates(world())
    weak = next(r for r in rows if {r["src"], r["dst"]} == {"duty", "calm"})
    assert weak["weakness"] > 0.9 and weak["tension"]


def test_stale_break_is_refused(tmp_path):
    o = world()
    st = steward(o, tmp_path)
    patch = st.propose(breaker.propose_split(o, "duty"))
    o.apply_ops([Op(op="update_node", id="duty", label="Duty to Rome")])
    with pytest.raises(GovernanceError):
        st.approve(patch.id, "Android K")


def test_works_on_the_seed(tmp_path):
    o = Ontology()
    st = Steward(o, tmp_path)
    for p in load_seed(st, SEED):
        st.approve(p.id, "test")
    for row in breaker.split_candidates(o, top=3):
        try:
            p = breaker.propose_split(o, row["id"])
        except ValueError:
            continue  # ambivalent but not a bridge: parts must be named
        o.check_ops(p.ops)
    for row in breaker.sever_candidates(o, top=3):
        o.check_ops(breaker.propose_sever(o, row["src"], row["dst"]).ops)


def test_break_api_only_queues(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from androidmirror.p1_server import create_app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(tmp_path)
    st = app.state.steward
    for p in load_seed(st, SEED):
        st.approve(p.id, "test")
    c = TestClient(app)
    cands = c.get("/api/break/candidates").json()
    assert set(cands) == {"split", "sever"} and cands["sever"]
    version = st.ontology.version
    cut = cands["sever"][0]
    patch = c.post("/api/break/sever", json={"src": cut["src"], "dst": cut["dst"]}).json()
    assert patch["kind"] == "break" and patch["status"] == "proposed"
    assert st.ontology.version == version  # queued, not applied
    named = c.post("/api/break/split", json={"node": "simplicity", "parts": [{"label": "Plain living"},
                                                                              {"label": "Plain speech"}]})
    assert named.status_code == 200
    assert c.post("/api/break/split", json={"node": "nope"}).status_code == 400
    assert c.get("/api/break/links/simplicity").json()
    assert c.post(f"/api/patches/{patch['id']}/approve", json={"approver": "Android K"}).status_code == 200


def test_cli_break_commands(tmp_path, capsys):
    from androidmirror.p1_cli import main

    main(["--data", str(tmp_path), "seed", str(SEED)])
    main(["--data", str(tmp_path), "approve", "all", "--as", "test"])
    main(["--data", str(tmp_path), "break-candidates"])
    main(["--data", str(tmp_path), "break-split", "simplicity", "--parts", "Plain living", "Plain speech"])
    assert "queued" in capsys.readouterr().out
