import json
from pathlib import Path

from fastapi.testclient import TestClient

from androidmirror.mirror.build import build, rediscover
from androidmirror.mirror.corpus import CorpusSpec
from androidmirror.mirror.store import Mirror
from androidmirror.server import create_app

from .fake import FakeCoder, write_corpus


def make(tmp_path: Path, n: int = 800, **settings):
    write_corpus(tmp_path / "corpus" / "synthetic", n)
    spec = CorpusSpec(id="synthetic", title="Synthetic", subject="Tester", splitter="documents")
    coder = FakeCoder()
    cfg = {"bootstrap": 4, "selection": {"min_support": 5}, **settings}
    mirror = build(spec, tmp_path, coder=coder, settings=cfg, log=lambda *_: None)
    return spec, coder, mirror


def test_build_recovers_the_equivalence_class(tmp_path):
    _, coder, mirror = make(tmp_path)
    c = mirror["causal"]
    assert set(c["variables"]) == {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}
    directed = {(a, b) for a, b, _ in c["cpdag"]["directed"]}
    # the collider and everything Meek's R1 carries from it are compelled by the data alone
    assert {("alpha", "gamma"), ("beta", "gamma"), ("gamma", "delta"), ("delta", "epsilon")} <= directed
    assert c["cpdag"]["undirected"] == []
    assert c["class_size"]["data"]["count"] == 1
    # f is independent of everything, so the stated claim a -> f has no link in the data
    pin = next(p for p in c["pins"] if p["src"] == "alpha" and p["dst"] == "zeta")
    assert pin["status"] == "no link in the data"
    agree = next(p for p in c["pins"] if p["src"] == "gamma")
    assert agree["status"] == "agrees"
    # every compelled arrow traces back to a test
    why = next(w for a, b, w in c["cpdag"]["directed"] if (a, b) == ("alpha", "gamma"))
    assert why["rule"] == "v-structure" and why["test"]["p"] > 0.05
    assert coder.calls["code"] == 800 and coder.calls["measure"] == 800


def test_build_is_cached_and_causal_reruns_without_coding(tmp_path):
    spec, coder, _ = make(tmp_path, n=200)
    coder2 = FakeCoder()
    build(spec, tmp_path, coder=coder2, settings={"bootstrap": 0}, log=lambda *_: None)
    assert coder2.calls == {"code": 0, "consolidate": 0, "measure": 0}
    m = rediscover(tmp_path, "synthetic", {"alpha": 0.01, "claims_as_pins": False})
    assert m["causal"]["settings"]["alpha"] == 0.01 and m["causal"]["pins"] == []


def test_reversible_edges_and_pins(tmp_path):
    # without a, b the chain c - d - e has no collider, so the data cannot orient it
    _, _, mirror = make(tmp_path, selection={"min_support": 5, "max_vars": 3}, bootstrap=0)
    c = mirror["causal"]
    assert set(c["variables"]) >= {"gamma", "delta"}
    und = {tuple(e) for e in c["cpdag"]["undirected"]}
    assert ("delta", "gamma") in und
    assert c["class_size"]["data"]["count"] >= 2
    # the stated claim c -> d pins the edge, and the class shrinks
    assert next(p for p in c["pins"] if p["src"] == "gamma")["status"] == "applied"
    assert c["class_size"]["with_pins"]["count"] < c["class_size"]["data"]["count"]
    statuses = {e["status"] for e in c["edges"]}
    assert "pinned" in statuses


def test_store_views_and_api(tmp_path):
    make(tmp_path, n=400)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/status").json()["built"] == ["synthetic"]
    kg = client.get("/api/graph", params={"layer": "kg"}).json()
    assert any(e["rel"] == "is_a" for e in kg["edges"])
    causal = client.get("/api/graph", params={"layer": "causal"}).json()
    assert {n["id"] for n in causal["nodes"]} >= {"gamma", "delta"}
    node = client.get("/api/node/gamma").json()
    assert node["mentions"] and node["mentions"][0]["locator"].startswith("Doc")
    assert node["causal"]["selection"]["status"] == "selected"
    p = client.get(f"/api/passage/{node['mentions'][0]['passage']}").json()
    assert "gamma" in p["text"] and any(lv["id"] == "gamma" for lv in p["levels"])
    eff = client.get("/api/effect", params={"x": "gamma", "y": "delta"}).json()
    assert eff["data_only"]["sign"] == "+"
    eff = client.get("/api/effect", params={"x": "delta", "y": "epsilon"}).json()
    assert eff["data_only"]["sign"] == "-"
    edge = client.get("/api/edge", params={"a": "alpha", "b": "gamma"}).json()
    assert edge["chain"][0]["rule"] == "v-structure" and edge["n_together"] > 0
    know = client.get("/api/knowledge").json()
    types = {v["type"] for v in know["violations"]}
    assert "denied_but_follows" in types  # "justice is not a good" against justice is_a virtue is_a good
    h = client.get("/api/holds", params={"s": "self", "r": "values", "o": "justice"}).json()
    assert h["status"] == "inferred"
    face = client.get("/api/archetype").json()
    assert "archetype" in face
    r = client.post("/api/ask", json={"question": "what does gamma do?"}).json()
    assert r["nodes"] == ["gamma"] and r["facts"]


def test_no_mirror_built_yet(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/status").json()["built"] == []
    assert client.get("/api/graph").status_code == 404
