from fastapi.testclient import TestClient

from androidmirror import evolver
from androidmirror.server import create_app


def test_evolve_endpoint_only_proposes(tmp_path, monkeypatch):
    monkeypatch.setattr(evolver.search, "discover", lambda: {})
    client = TestClient(create_app(tmp_path))
    assert len(client.get("/api/probes").json()) >= 1
    r = client.post("/api/evolve", json={"generations": 1}).json()
    assert r["operators"] == [] and r["designs"] == [] and r["queued"] == []
    assert client.get("/api/status").json()["version"] == 0
