import json
from pathlib import Path

from fastapi.testclient import TestClient

from androidmirror.mirror.build import build, rediscover, run_step
from androidmirror.mirror.corpus import CorpusSpec
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
    assert c["n_rows"] == 800 and all(s["origin"] == "corpus" for s in mirror["situations"])
    directed = {(a, b) for a, b, _ in c["cpdag"]["directed"]}
    # the collider and everything Meek's R1 carries from it are compelled by the data alone
    assert {("alpha", "gamma"), ("beta", "gamma"), ("gamma", "delta"), ("delta", "epsilon")} <= directed
    assert c["cpdag"]["undirected"] == []
    assert c["class_size"]["data"]["count"] == 1
    # zeta is independent of everything, so the stated claim alpha -> zeta has no link in the data
    pin = next(p for p in c["pins"] if p["src"] == "alpha" and p["dst"] == "zeta")
    assert pin["status"] == "no link in the data"
    assert next(p for p in c["pins"] if p["src"] == "gamma")["status"] == "agrees"
    # every compelled arrow traces back to a test, and the collider's own signature was checked
    why = next(w for a, b, w in c["cpdag"]["directed"] if (a, b) == ("alpha", "gamma"))
    assert why["rule"] == "v-structure" and why["test"]["p"] > 0.05 and why["signature"]["p"] <= 0.05
    # the subject's stated independences are tested, not trusted
    checks = {i["kind"]: i["status"] for i in c["independences"]}
    assert checks == {"whether_or_not": "agrees", "not_x_but_y": "agrees"}
    # the three evidence views are run side by side
    assert set(c["views"]) == {"stated", "implied", "all"}
    assert coder.calls["code"] == 800 and coder.calls["situations"] == 800


def test_unmentioned_is_unknown_not_zero(tmp_path):
    _, _, mirror = make(tmp_path, n=200, bootstrap=0)
    s = next(x for x in mirror["situations"] if len(x["cells"]) < 6 and all(c["tier"] == "stated" for c in x["cells"].values()))
    text = (tmp_path / "corpus" / "synthetic" / f"{s['passage']}.txt").read_text()
    for v in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"):
        assert (v in s["cells"]) == (f"[v {v} " in text)


def test_build_is_cached_and_causal_reruns_without_coding(tmp_path):
    spec, coder, _ = make(tmp_path, n=200)
    coder2 = FakeCoder()
    build(spec, tmp_path, coder=coder2, settings={"bootstrap": 0}, log=lambda *_: None)
    assert coder2.calls == {"code": 0, "consolidate": 0, "situations": 0, "answer": 0}
    m = rediscover(tmp_path, "synthetic", {"alpha": 0.01, "claims_as_pins": False}, log=lambda *_: None)
    assert m["causal"]["settings"]["alpha"] == 0.01 and m["causal"]["pins"] == []
    m = rediscover(tmp_path, "synthetic", {"view": "stated"}, log=lambda *_: None)
    assert m["causal"]["view"] == "stated"


def test_reversible_edges_and_pins(tmp_path):
    # without beta the chain alpha - gamma - delta has no collider, so the data cannot orient it
    _, _, mirror = make(tmp_path, selection={"min_support": 5, "max_vars": 3}, bootstrap=0)
    c = mirror["causal"]
    assert set(c["variables"]) >= {"gamma", "delta"}
    und = {tuple(e) for e in c["cpdag"]["undirected"]}
    assert ("delta", "gamma") in und
    assert c["class_size"]["data"]["count"] >= 2
    # the stated claim gamma -> delta pins the edge, and the class shrinks
    assert next(p for p in c["pins"] if p["src"] == "gamma")["status"] == "applied"
    assert c["class_size"]["with_pins"]["count"] < c["class_size"]["data"]["count"]
    assert "pinned" in {e["status"] for e in c["edges"]}


def test_transformation_validate_fill_and_vignettes(tmp_path):
    make(tmp_path, n=600, selection={"min_support": 5, "max_vars": 3}, bootstrap=0, claims_as_pins=False)
    coder = FakeCoder()
    quiet = lambda *_: None  # noqa: E731
    try:
        run_step("fill", tmp_path, "synthetic", coder=coder, log=quiet)
        raise AssertionError("fill must refuse to run before validation")
    except RuntimeError as e:
        assert "validate" in str(e)
    v = run_step("validate", tmp_path, "synthetic", coder=coder, log=quiet, n=60)
    assert v["held_out"] == 60 and v["overall"]["subject"]["precision"] >= 0.6
    assert v["overall"]["generic"]["answered"] == 0 and v["trusted"]
    f = run_step("fill", tmp_path, "synthetic", coder=coder, log=quiet, leak_sample=5)
    assert f["cells"] and f["prior_leak_rate"] is not None
    g = run_step("vignettes", tmp_path, "synthetic", coder=coder, log=quiet, per_level=10, max_targets=2)
    assert g["situations"] and all(s["cells"][s["target"]]["tier"] == "set" for s in g["situations"])
    mirror = json.loads((tmp_path / "mirror" / "synthetic" / "mirror.json").read_text())
    tiers = {c.get("source") for s in mirror["situations"] for c in s["cells"].values()}
    assert "transformation" in tiers
    vig = mirror["causal"]["vignettes"]
    assert vig["n_vignettes"] == len(g["situations"]) and vig["responses"]
    # with no stated claims, the corpus leaves the chain unoriented; setting gamma in vignettes moves delta
    change = {(d["a"], d["b"]): d["change"] for d in vig["diff"]}
    assert change[("delta", "gamma")] == "oriented by vignettes"
    assert mirror["transformation"]["validation"]["trusted"] and mirror["transformation"]["vignettes"]["n"]

    client = TestClient(create_app(tmp_path))
    graph = client.get("/api/graph", params={"layer": "causal"}).json()
    e = next(x for x in graph["edges"] if {x["source"], x["target"]} == {"gamma", "delta"})
    assert graph["vignettes"] and e["change"] == "oriented by vignettes" and e["vignette_direction"] == ["gamma", "delta"]
    edge = client.get("/api/edge", params={"a": "gamma", "b": "delta"}).json()
    assert edge["n_vignettes"] > 0 and edge["vignette_change"]["change"] == "oriented by vignettes"
    eff = client.get("/api/effect", params={"x": "gamma", "y": "delta"}).json()
    assert "with_vignettes" in eff
    prof = client.get("/api/profile").json()
    assert prof["profile"]["vocabulary"] and prof["validation"]["overall"]


def test_store_views_and_api(tmp_path):
    make(tmp_path, n=800)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/status").json()["built"] == ["synthetic"]
    kg = client.get("/api/graph", params={"layer": "kg"}).json()
    assert any(e["rel"] == "is_a" for e in kg["edges"])
    causal = client.get("/api/graph", params={"layer": "causal"}).json()
    assert {n["id"] for n in causal["nodes"]} >= {"gamma", "delta"}
    node = client.get("/api/node/gamma").json()
    assert node["mentions"] and node["mentions"][0]["locator"].startswith("Doc")
    assert node["causal"]["selection"]["status"] == "selected" and node["situations"]
    sid = node["situations"][0]["id"]
    s = client.get(f"/api/situation/{sid}").json()
    assert any(c["id"] == "gamma" and c["tier"] == "stated" for c in s["cells"])
    p = client.get(f"/api/passage/{node['mentions'][0]['passage']}").json()
    assert "gamma" in p["text"] and p["situations"]
    eff = client.get("/api/effect", params={"x": "gamma", "y": "delta"}).json()
    assert eff["data_only"]["sign"] == "+"
    eff = client.get("/api/effect", params={"x": "delta", "y": "epsilon"}).json()
    assert eff["data_only"]["sign"] == "-"
    edge = client.get("/api/edge", params={"a": "alpha", "b": "gamma"}).json()
    assert edge["chain"][0]["rule"] == "v-structure" and edge["n_together"] > 0
    cv = client.get("/api/causal").json()
    assert cv["tiers"]["stated"] > 0 and cv["views"] and cv["situations"] == 800
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
