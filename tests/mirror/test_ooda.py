from fastapi.testclient import TestClient

from androidmirror.mirror import interview as I
from androidmirror.mirror import ooda
from androidmirror.mirror.build import build
from androidmirror.mirror.corpus import CorpusSpec
from androidmirror.mirror.store import Mirror
from androidmirror.server import create_app

from .fake import FakeCoder
from .test_build import make

# An interviewee who notices danger, reads it through fear, decides by what they value, and acts by running;
# running brings back calm noticing (feedback), and fear sometimes makes them run without deciding.
ANSWERS = {
    "observe-01": "[v danger 1 event] [v noise 1 event] [j causes noise danger + 1 direct]",
    "observe-02": "[v danger 1 event] [v fear 1 emotion] [j causes danger fear + 1 direct]",
    "orient-01": "[v fear 1 emotion] [v safety 1 state] [j causes fear safety - 1 direct] [w safety 1]",
    "orient-02": "[v fear 1 emotion] [s mother learned_from self] [s self learned_from mother]",
    "decide-01": "[v safety 1 state] [s safety is_a good] [s self values safety] [w safety 1]",
    "decide-02": "[v safety 1 state] [v running 1 practice] [j causes safety running + 1 direct]",
    "act-01": "[v running 1 practice] [s self practices running] [v noise 1 event] [j causes running noise - 1 direct]",
    "act-02": "[v fear 1 emotion] [v running 1 practice] [j causes fear running + 1 direct]",
}


def staged(tmp_path):
    I.start(tmp_path, "Tester")
    for qid, text in ANSWERS.items():
        I.answer(tmp_path, "tester", qid, text)
    I.write_corpus(tmp_path, "tester")
    spec = CorpusSpec.load(str(tmp_path / "corpora" / "interview-tester.json"))
    build(spec, tmp_path, coder=FakeCoder(), log=lambda *_: None)
    return tmp_path / "mirror" / spec.id


def tiers(r):
    return {c["id"]: t["id"] for t in r["tiers"] for c in t["concepts"]}


def test_concepts_land_in_their_stage(tmp_path):
    r = ooda.model(Mirror.load(staged(tmp_path)))
    t = tiers(r)
    assert t["noise"] == "observe" and t["danger"] == "observe"
    assert t["fear"] == "orient" and t["mother"] == "orient"
    assert t["safety"] == "decide" and t["running"] == "act"
    assert r["staged"]
    running = next(c for x in r["tiers"] for c in x["concepts"] if c["id"] == "running")
    reasons = {x["reason"] for x in running["reasons"]}
    assert "Tester practices it" in reasons and "it is a practice" in reasons
    assert any("Act stage" in x for x in reasons)


def test_links_are_classified_by_the_stages_they_join(tmp_path):
    r = ooda.model(Mirror.load(staged(tmp_path)))
    kinds = {(l["src"], l["dst"]): l["kind"] for l in r["links"] if l["via"] != "fact"}
    assert kinds[("noise", "danger")] == "within"
    assert kinds[("danger", "fear")] == "forward"
    assert kinds[("safety", "running")] == "forward"
    assert kinds[("fear", "running")] == "skip"  # Orient straight to Act
    # Act back to Observe: the loop closes; the acyclic causal model holds it aside, the OODA model keeps it
    assert kinds[("running", "noise")] == "feedback"
    assert next(l for l in r["links"] if l["src"] == "running")["via"] == "loop"
    assert r["flow"]["act"]["observe"] == 1
    text = " ".join(r["findings"])
    assert "The loop closes" in text and "Orient leads straight to Act" in text


def test_a_placement_by_hand_overrides_and_reverts(tmp_path):
    root = staged(tmp_path)
    ooda.place(root, "danger", "orient", "Android K")
    c = next(c for t in ooda.model(Mirror.load(root))["tiers"] for c in t["concepts"] if c["id"] == "danger")
    assert c["tier"] == "orient" and c["auto"] == "observe" and c["by"] == "Android K"
    ooda.place(root, "danger", None)
    assert tiers(ooda.model(Mirror.load(root)))["danger"] == "observe"


def test_any_corpus_gets_a_model_and_nothing_changes(tmp_path):
    make(tmp_path)
    root = tmp_path / "mirror" / "synthetic"
    before = (root / "mirror.json").read_text()
    r = ooda.model(Mirror.load(root))
    assert not r["staged"] and r["placed"] and r["findings"]
    t = tiers(r)
    assert t["virtue"] == "decide"  # called good
    assert t["alpha"] == "observe"  # a cause nothing in the model causes
    assert (root / "mirror.json").read_text() == before


def test_routes(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    staged(tmp_path)
    c = TestClient(create_app(tmp_path))
    q = {"corpus": "interview-tester"}
    g = c.get("/api/graph", params={**q, "layer": "ooda"}).json()
    assert {n["tier"] for n in g["nodes"]} == {"observe", "orient", "decide", "act"}
    assert any(e["kind"] == "feedback" for e in g["edges"])
    assert len(c.get("/api/ooda", params=q).json()["tiers"]) == 4
    r = c.post("/api/ooda/place", json={"concept": "noise", "tier": "act", "by": "Ari", **q}).json()
    assert tiers(r)["noise"] == "act"
    assert c.post("/api/ooda/place", json={"concept": "noise", "tier": "nap", **q}).status_code == 400
    assert c.post("/api/ooda/place", json={"concept": "nope", "tier": "act", **q}).status_code == 404


def test_ask_reads_the_ooda_model(tmp_path, monkeypatch):
    from androidmirror import llm
    from androidmirror.mirror import ask

    monkeypatch.setattr(llm, "available", lambda: False)
    m = Mirror.load(staged(tmp_path))
    # an idea: its stage and why, and the stage links that touch it
    items = ask.ask(m, "what does fear do?")["facts"]
    o = [x for x in items if x.split("]")[0].startswith("[O")]
    assert any("stage: fear is placed in Orient" in x and "Orient stage" in x for x in o)
    assert any("fear raises running, Orient -> Act, skipping a stage" in x for x in o)
    assert any("what the loop shows: The loop closes" in x for x in o)
    # a stage: its ideas, with no idea named in the question
    items = ask.ask(m, "what does Tester observe?")["facts"]
    assert any("stage Observe" in x and "danger" in x and "noise" in x for x in items)
    assert not any("stage Decide" in x for x in items)
    # the whole loop: every stage, the flow between stages, and the feedback link held aside by the causal model
    items = ask.ask(m, "how does the OODA loop close?")["facts"]
    assert sum("stage " in x and "(" in x and ": " in x and x.split("] ")[1].startswith("stage ") for x in items) >= 4
    assert any("passages behind the links between stages" in x for x in items)
    assert any("running raises noise, Act -> Observe, from Act back" in x and "holds it aside" in x for x in items)
