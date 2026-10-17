from pathlib import Path

from fastapi.testclient import TestClient

from androidmirror.mirror.build import build, refit
from androidmirror.mirror.corpus import CorpusSpec
from androidmirror.server import create_app

from .fake import FakeCoder, write_corpus


def make(tmp_path: Path, n: int = 80, **settings):
    write_corpus(tmp_path / "corpus" / "synthetic", n)
    spec = CorpusSpec(id="synthetic", title="Synthetic", subject="Tester", splitter="documents")
    coder = FakeCoder()
    mirror = build(spec, tmp_path, coder=coder, settings=settings or None, log=lambda *_: None)
    return spec, coder, mirror


def test_build_recovers_the_subjects_model(tmp_path):
    _, coder, mirror = make(tmp_path)
    c = mirror["causal"]
    arrows = {(e["src"], e["dst"], e["sign"]) for e in c["edges"]}
    assert arrows == {("alpha", "gamma", 1), ("beta", "gamma", 1), ("gamma", "delta", 1), ("delta", "epsilon", -1)}
    st = {x["id"]: x for x in c["constraints"]}
    # said in 30 passages (twice in 10 of them), and implied by "not alpha but gamma" in 3 more:
    # weighted by passages, not by repetitions
    gd = st["reaches|gamma|delta"]
    assert gd["weight"] == 33 and gd["mentions"] == 43 and gd["kinds"] == {"causes": 40, "screens_off": 3}
    assert st["reaches|alpha|gamma"]["kinds"] == {"intervention": 8}
    assert st["reaches|alpha|delta"]["status"] == "follows"  # an overall claim the chain explains: no shortcut
    assert st["only_through|alpha|gamma|delta"]["status"] == "holds"
    assert st["no_path|zeta|alpha"]["status"] == "holds"
    slip = st["reaches|epsilon|gamma"]
    assert slip["status"] == "rejected" and slip["reason"] == "cycle"
    # other voices and hypotheses are kept, never fitted
    assert st["reaches|beta|zeta"]["status"] == "not held" and st["reaches|zeta|delta"]["status"] == "not held"
    assert [u["missing"] for u in mirror["unresolved"]] == [["unicorns"]]
    assert {v["concept"]: v["direction"] for v in mirror["values"]} == {"zeta": 1, "epsilon": -1}
    assert c["holdout"]["folds"] == 5 and c["holdout"]["positive"]["restated"] > 0
    assert coder.calls == {"code": 80, "consolidate": 1, "judgments": 80}


def test_build_is_cached_and_causal_refits_without_reading(tmp_path):
    spec, _, _ = make(tmp_path)
    coder2 = FakeCoder()
    build(spec, tmp_path, coder=coder2, log=lambda *_: None)
    assert coder2.calls == {"code": 0, "consolidate": 0, "judgments": 0}
    m = refit(tmp_path, "synthetic", {"min_weight": 6}, log=lambda *_: None)
    assert m["causal"]["settings"]["min_weight"] == 6
    assert {(e["src"], e["dst"]) for e in m["causal"]["edges"]} == {("alpha", "gamma"), ("gamma", "delta"), ("delta", "epsilon")}


def test_store_views_and_api(tmp_path):
    make(tmp_path)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/status").json()["built"] == ["synthetic"]
    meta = client.get("/api/meta").json()
    assert meta["judgments"] > 0 and meta["causal"]["arrows"] == 4
    kg = client.get("/api/graph", params={"layer": "kg"}).json()
    assert any(e["rel"] == "is_a" for e in kg["edges"])
    causal = client.get("/api/graph", params={"layer": "causal"}).json()
    assert {n["id"] for n in causal["nodes"]} == {"alpha", "beta", "gamma", "delta", "epsilon"}
    assert next(n for n in causal["nodes"] if n["id"] == "epsilon")["value"] == -1
    e = next(x for x in causal["edges"] if (x["source"], x["target"]) == ("gamma", "delta"))
    assert e["direct"] and e["weight"] == 33 and e["sign"] == 1
    node = client.get("/api/node/gamma").json()
    assert node["mentions"] and node["causal"]["in_model"] and node["causal"]["edges"] and node["causal"]["constraints"]
    edge = client.get("/api/edge", params={"src": "gamma", "dst": "delta"}).json()
    assert edge["own"]["weight"] == 33 and edge["own"]["evidence"][0]["quote"]
    assert any(c["id"] == "reaches|alpha|delta" for c in edge["supports"])
    assert any(c["id"] == "reaches|epsilon|gamma" for c in edge["challenged_by"])
    ex = client.get("/api/explain", params={"x": "alpha", "y": "epsilon"}).json()
    assert ex["verdict"] == "lowers" and ex["paths"][0]["labels"] == ["alpha", "gamma", "delta", "epsilon"]
    assert client.get("/api/explain", params={"x": "zeta", "y": "alpha"}).json()["verdict"].startswith("makes no difference")
    cv = client.get("/api/causal").json()
    assert cv["stats"]["rejected"] == 1 and cv["n_unresolved"] == 1 and cv["values"] and cv["holdout"]
    p = client.get("/api/passage/00030").json()
    assert p["judgments"][0]["kind"] == "intervention"
    know = client.get("/api/knowledge").json()
    assert "denied_but_follows" in {v["type"] for v in know["violations"]}
    h = client.get("/api/holds", params={"s": "self", "r": "values", "o": "justice"}).json()
    assert h["status"] == "inferred"
    face = client.get("/api/archetype").json()
    assert "archetype" in face
    r = client.post("/api/ask", json={"question": "what does gamma do?"}).json()
    assert r["nodes"] == ["gamma"] and any("cause-and-effect claim" in f for f in r["facts"])
    assert "passages" not in r and not any('"' in f for f in r["facts"])  # structures only, no corpus text
    r = client.post("/api/ask", json={"question": "what does alpha do to epsilon?"}).json()
    assert any("pathway: alpha -> gamma -> delta -> epsilon, so alpha lowers epsilon" in f for f in r["facts"])


def test_ask_answers_then_proposes_uses(tmp_path, monkeypatch):
    from androidmirror import llm
    from androidmirror.mirror import ask
    from androidmirror.mirror.build import mirror_dir
    from androidmirror.mirror.store import Mirror

    spec, *_ = make(tmp_path)
    m = Mirror.load(mirror_dir(tmp_path, spec.id))
    calls = []

    def fake(system, user, schema, **_):
        calls.append((system, user))
        if "uses" in schema["properties"]:
            return {"uses": [{"lever": "alpha", "outcome": "less epsilon", "how": "via gamma and delta",
                              "assumes": "Tester's model holds", "citations": ["C1"]}] * 4}
        return {"answer": "Alpha lowers epsilon [C1].", "citations": ["C1"], "nodes_used": ["alpha"]}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "_json_call", fake)
    r = ask.ask(m, "what does alpha do to epsilon?")
    assert r["answer"].startswith("Alpha") and len(r["uses"]) == 3 and r["uses"][0]["lever"] == "alpha"
    assert len(calls) == 2 and "Answer: Alpha lowers epsilon [C1]." in calls[1][1] and "[C1]" in calls[1][1]


def test_no_mirror_built_yet(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/status").json()["built"] == []
    assert client.get("/api/graph").status_code == 404
