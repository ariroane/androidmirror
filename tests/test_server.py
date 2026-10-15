from pathlib import Path

from fastapi.testclient import TestClient

from androidmirror.ingest import load_seed
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
