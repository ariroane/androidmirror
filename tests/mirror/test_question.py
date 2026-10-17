import json

import pytest
from fastapi.testclient import TestClient

from androidmirror import llm
from androidmirror.mirror import answers
from androidmirror.mirror import judgments as jd
from androidmirror.mirror import question as Q
from androidmirror.mirror.build import build
from androidmirror.mirror.causal import fit
from androidmirror.mirror.store import Mirror
from androidmirror.server import create_app

from .fake import FakeCoder
from .test_build import make
from .test_units import J, arrows, status


@pytest.fixture
def built(tmp_path):
    spec, _, _ = make(tmp_path)
    return tmp_path, spec, tmp_path / "mirror" / "synthetic"


def ask(root, kind, about=None):
    qs = Q.questions(Mirror.load(root), limit=None, kinds={kind})
    return next(q for q in qs if about is None or set(about) <= set(q["about"]))


def answer(root, kind, option, about=None, approve=True):
    q = ask(root, kind, about)
    p = Q.answer(Mirror.load(root), q["id"], option)
    if approve:
        Q.approve(root, p["id"], "Android K")
    return q, p


def constraint(root, cid):
    return next(c for c in json.loads((root / "mirror.json").read_text())["causal"]["constraints"] if c["id"] == cid)


def test_questions_come_from_what_the_structures_leave_open(built):
    _, _, root = built
    qs = Q.questions(Mirror.load(root), limit=None)
    kinds = {q["kind"] for q in qs}
    assert {"tension", "evaluate", "link", "conflict"} <= kinds
    top = qs[0]
    assert top["kind"] == "tension" and "epsilon raises gamma" in top["text"] and top["evidence"]
    assert [o["key"] for o in top["options"]] == ["this", "other", "neither", "skip"]
    assert all(o["changes"] for o in top["options"] if o["key"] != "skip")
    # the same Mirror asks the same questions, under the same ids
    assert [q["id"] for q in Q.questions(Mirror.load(root), limit=None)] == [q["id"] for q in qs]
    # said or connected pairs are not asked about again
    links = {frozenset(q["about"]) for q in qs if q["kind"] == "link"}
    assert frozenset(("zeta", "alpha")) not in links and frozenset(("gamma", "delta")) not in links
    assert frozenset(("alpha", "beta")) in links


def test_answering_only_proposes(built):
    _, _, root = built
    before = (root / "mirror.json").read_bytes()
    q, p = answer(root, "tension", "neither", approve=False)
    assert p["status"] == "proposed" and p["changes"] == [
        {"type": "ruling", "target": "reaches|epsilon|gamma", "ruling": "retract", "says": "not: epsilon raises gamma"}]
    assert p["preview"]["judgments"] == [{"id": "reaches|epsilon|gamma", "before": "rejected", "after": "retracted"}]
    assert not p["preview"]["arrows_added"] and not p["preview"]["arrows_removed"]
    assert (root / "mirror.json").read_bytes() == before  # nothing applied
    assert constraint(root, "reaches|epsilon|gamma")["status"] == "rejected"
    # a question with a proposal waiting is not asked again; rejecting it brings it back
    assert q["id"] not in {x["id"] for x in Q.questions(Mirror.load(root), limit=None)}
    Q.reject(root, p["id"])
    assert q["id"] in {x["id"] for x in Q.questions(Mirror.load(root), limit=None)}


def test_approval_needs_a_name_and_can_be_revoked(built):
    _, _, root = built
    q, p = answer(root, "tension", "neither", approve=False)
    with pytest.raises(ValueError):
        Q.approve(root, p["id"], "  ")
    Q.approve(root, p["id"], "Android K")
    with pytest.raises(ValueError):
        Q.approve(root, p["id"], "Android K")  # not twice
    c = constraint(root, "reaches|epsilon|gamma")
    assert c["status"] == "retracted" and c["ruling"]["ruling"] == "retract"
    m = Mirror.load(root)
    assert m.passage(f"answer:{p['id']}")["text"].startswith("Q: ")
    assert q["id"] not in {x["id"] for x in Q.questions(m, limit=None)}
    Q.revoke(root, p["id"])
    assert constraint(root, "reaches|epsilon|gamma")["status"] == "rejected"


def test_affirming_the_slip_breaks_the_loop_elsewhere(built):
    _, _, root = built
    _, p = answer(root, "tension", "this", approve=False)
    assert p["preview"]["arrows_added"] == ["epsilon raises gamma"]
    assert p["preview"]["arrows_removed"] == ["delta lowers epsilon"]  # the weakest link of the loop gives way
    Q.approve(root, p["id"], "Android K")
    c = constraint(root, "reaches|epsilon|gamma")
    assert c["status"] == "arrow" and c["pinned"]
    assert constraint(root, "reaches|delta|epsilon")["status"] == "rejected"


def test_a_link_answer_adds_a_judgment_of_its_own(built):
    _, _, root = built
    _, p = answer(root, "link", "lowers", about=["alpha", "beta"])
    c = constraint(root, "reaches|alpha|beta")
    assert c["status"] == "arrow" and c["sign"] == -1 and c["pinned"] and c["weight"] == 1
    assert c["evidence"][0]["passage"] == f"answer:{p['id']}" and c["evidence"][0]["register"] == "answers"
    m = Mirror.load(root)
    assert ("alpha", "beta") in {(e["src"], e["dst"]) for e in m.causal["edges"]}


def test_answers_survive_a_rebuild(built):
    tmp, spec, root = built
    answer(root, "link", "raises", about=["alpha", "beta"])
    build(spec, tmp, coder=FakeCoder(), log=lambda *_: None)
    assert constraint(root, "reaches|alpha|beta")["status"] == "arrow"
    assert answers.approved(answers.load(root))


def test_knowledge_graph_answers(built):
    _, _, root = built
    assert "denied_but_follows" in {v["type"] for v in Mirror.load(root).kb.violations}
    _, p = answer(root, "conflict", "holds", about=["justice"])
    m = Mirror.load(root)
    assert "denied_but_follows" not in {v["type"] for v in m.kb.violations}
    denial = next(f for f in m.facts if (f["s"], f["r"], f["o"], f["neg"]) == ("justice", "is_a", "good", True))
    assert denial["retracted"] and not denial["held"] and denial["evidence"]  # kept on record
    answer(root, "evaluate", "bad", about=["gamma"])
    assert Mirror.load(root).evaluation("gamma") == "bad"


def test_the_other_side_of_a_denial_drops_its_weakest_premise(built):
    _, _, root = built
    _, p = answer(root, "conflict", "denied", about=["justice"], approve=False)
    assert p["changes"][0]["key"] == "justice|is_a|virtue"
    assert p["preview"]["inconsistencies"]["after"] < p["preview"]["inconsistencies"]["before"]


def test_rulings_on_open_arrows_and_signs():
    # a -> c said most often but only "overall"; a -> b -> c also said: the a -> c arrow is open
    js = [J("causes", "a", "c", p=f"x{i}") for i in range(5)] + [J("causes", "a", "b", p="y"), J("causes", "b", "c", p="z")]
    m = fit(js)
    assert next(e for e in m["edges"] if (e["src"], e["dst"]) == ("a", "c"))["open"]

    def ruling(**kw):
        return {**J("ruling", "a", "c", sign=kw.pop("sign", 0), p="answer:1", held=False), "target": "reaches|a|c",
                "ruling": "affirm", "direct": False, "indirect": False, **kw}

    through = fit(js + [ruling(indirect=True)])
    assert arrows(through) == {("a", "b", 1), ("b", "c", 1)} and status(through)["reaches|a|c"] == "follows"
    direct = fit(js + [ruling(direct=True)])
    assert not next(e for e in direct["edges"] if (e["src"], e["dst"]) == ("a", "c"))["open"]
    flipped = fit([J("causes", "a", "c", p="p1"), J("causes", "a", "c", -1, p="p2")] + [ruling(sign=-1)])
    assert arrows(flipped) == {("a", "c", -1)}
    retracted = jd.constraints(js + [ruling(ruling="retract")])
    assert next(c for c in retracted if c["id"] == "reaches|a|c")["weight"] == 0


def test_a_pinned_answer_beats_a_stronger_denial_of_the_same_pair():
    js = [J("independent", "a", "b", sign=0, p=f"d{i}") for i in range(3)]
    answer_ = {**J("causes", "a", "b", p="answer:1"), "register": "answers"}
    m = fit(js + [answer_])
    assert arrows(m) == {("a", "b", 1)} and status(m)["no_path|a|b"] == "rejected"


def test_values_answer():
    mirror = {"judgments": [], "facts": [], "nodes": {"anger": {}},
              "values": [{"concept": "anger", "more": 2, "less": 2, "direction": 0, "evidence": []}]}
    book = {"proposals": [{"id": "p", "question": "q", "answer": "Less anger", "status": "approved",
                           "changes": [{"type": "value", "concept": "anger", "direction": -1}]}]}
    v = answers.overlay(mirror, book)["values"][0]
    assert v["direction"] == -1 and v["less"] == 3 and v["answered"] == -1
    assert mirror["values"][0]["less"] == 2  # the corpus-built Mirror is not changed


def test_a_written_answer_is_read_by_claude(built, monkeypatch):
    _, _, root = built
    q = ask(root, "tension")
    with pytest.raises(llm.LLMUnavailable):
        Q.answer(Mirror.load(root), q["id"], text="the loop is a slip of the pen")
    monkeypatch.setattr(llm, "available", lambda: True)
    seen = {}

    def fake(system, user, schema, **kw):
        seen["enum"] = schema["properties"]["option"]["enum"]
        return {"option": "neither", "note": "a slip of the pen"}

    monkeypatch.setattr(llm, "_json_call", fake)
    p = Q.answer(Mirror.load(root), q["id"], text="the loop is a slip of the pen")
    assert p["option"] == "neither" and p["note"] == "a slip of the pen" and p["status"] == "proposed"
    assert seen["enum"] == ["this", "other", "neither", "skip"]


def test_api(built):
    tmp, _, root = built
    c = TestClient(create_app(tmp))
    r = c.get("/api/questions", params={"limit": 3}).json()
    assert len(r["questions"]) == 3 and r["total"] > 3 and r["proposals"] == []
    q = r["questions"][0]
    assert c.post("/api/questions/answer", json={"question_id": q["id"], "option": "nope"}).status_code == 400
    assert c.post("/api/questions/answer", json={"question_id": "missing", "option": "this"}).status_code == 404
    p = c.post("/api/questions/answer", json={"question_id": q["id"], "option": q["options"][0]["key"]}).json()
    assert p["status"] == "proposed" and p["preview"]["summary"]
    assert c.post(f"/api/proposals/{p['id']}/approve", json={"approver": ""}).status_code == 400
    assert c.post(f"/api/proposals/{p['id']}/approve", json={"approver": "Android K"}).json()["status"] == "approved"
    assert c.get("/api/questions").json()["proposals"][0]["status"] == "approved"
    assert c.get("/api/causal").json()["stats"]["rejected"] == 1  # the loop now breaks at delta -> epsilon
    assert c.post(f"/api/proposals/{p['id']}/revoke", json={}).json()["status"] == "revoked"


def test_cli(built):
    _, _, root = built
    replies = iter(["3", "Android K", "", "q"])
    lines = []
    Q.cli(root, limit=5, inp=lambda _: next(replies), out=lines.append)
    assert any("approved" in x for x in lines)
    assert constraint(root, "reaches|epsilon|gamma")["status"] == "retracted"
