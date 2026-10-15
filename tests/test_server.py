from pathlib import Path

from fastapi.testclient import TestClient

from androidmirror.ingest import load_seed
from androidmirror.patches import Op
from androidmirror.server import create_app

SEED = Path(__file__).resolve().parent.parent / "data" / "seed_book1.json"


def test_review_flow(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(tmp_path)
    load_seed(app.state.steward, SEED)
    c = TestClient(app)
    assert c.get("/").status_code == 200
    pending = c.get("/api/patches").json()
    assert len(pending) == 9
    assert c.post(f"/api/patches/{pending[0]['id']}/approve", json={"approver": ""}).status_code == 400
    for p in pending:
        assert c.post(f"/api/patches/{p['id']}/approve", json={"approver": "Android K"}).status_code == 200
    s = c.get("/api/status").json()
    assert s["nodes"] > 20 and s["proposed"] == 0 and s["llm"] is False
    r = c.post("/api/intervene", json={"node": "vexation", "delta": -1}).json()
    assert r["effects"][0]["id"] == "cooperation" and r["story"] is None
    assert c.post("/api/query", json={"question": "Why does vexation matter?"}).json()["facts"]
    assert c.get("/api/investigate").json()["breaker"]["cut_points"]
    assert set(c.get("/api/metrics").json()) >= {"coherence", "tension", "valence"}
    assert c.post("/api/ingest", json={"book": 1}).status_code in (400, 502)  # no key or no network here


def test_engine_routes_only_queue(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(tmp_path)
    st = app.state.steward
    for p in load_seed(st, SEED):
        st.approve(p.id, "Android K")
    st.ontology.apply_ops([Op(op="add_node", id="stray", label="Stray thought", type="concept")])  # an orphan to tidy
    c = TestClient(app)
    version = c.get("/api/status").json()["version"]

    r = c.get("/api/refactor").json()
    assert {"patches", "findings", "metrics"} <= set(r)
    assert c.get("/api/status").json()["proposed"] == 0  # suggesting queues nothing
    key = r["patches"][0]["extra"]["key"]
    assert len(c.post("/api/refactor/queue", json={"keys": [key]}).json()["queued"]) == 1
    assert c.post("/api/refactor/queue", json={"keys": [key]}).json()["queued"] == []  # not twice
    assert c.get("/api/refactor").json()["patches"][0]["queued"] is True

    for cand in c.get("/api/chain/candidates").json():
        assert {"survivor", "absorbed", "survivor_label", "absorbed_label", "why"} <= set(cand)
    nodes = [n["id"] for n in c.get("/api/graph").json()["nodes"]]
    assert c.post("/api/chain/merge", json={"survivor": nodes[0], "absorbed": [nodes[1]]}).status_code in (200, 400)

    assert c.get("/api/status").json()["version"] == version  # nothing applied
