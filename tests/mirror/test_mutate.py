"""The mutate engine: merges and splits, re-evaluated in both graphs, applied only on approval."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from androidmirror.mirror import mutate
from androidmirror.mirror.build import build, mirror_dir
from androidmirror.mirror.corpus import CorpusSpec
from androidmirror.server import create_app

from .fake import FakeCoder


class Coder(FakeCoder):
    """The fake coder, with a codebook that folds "joy" into pleasure."""

    def consolidate(self, lines, subject):
        self.calls["consolidate"] += 1
        return [{"id": "pleasure", "label": "pleasure", "kind": "emotion", "gloss": "pleasure", "variable": True,
                 "members": [{"label": "pleasure", "polarity": 1}, {"label": "joy", "polarity": 1}]}]


def tags(i: int) -> list[str]:
    """A subject for whom wrath is another word for anger, calm is anger's opposite, and "pleasure" has two senses."""
    t = []
    if i < 2:
        t += [f"[v {c} 1 {k}]" for c, k in (("opinion", "belief"), ("anger", "emotion"), ("wrath", "emotion"),
                                             ("calm", "state"), ("misery", "state"), ("virtue", "disposition"))]
    if i < 5:
        t.append("[j causes opinion anger - 1 direct]")
    if 5 <= i < 8:
        t.append("[j causes opinion wrath - 1 direct]")
    if 8 <= i < 12:
        t.append("[j causes anger misery - 1 direct]")
    if 12 <= i < 14:
        t.append("[j causes wrath misery - 1 direct]")
    if 14 <= i < 17:
        t.append("[j causes opinion calm - -1 direct]")
    if 17 <= i < 20:
        t.append("[j causes calm misery - -1 direct]")
    if i == 20:
        t += ["[j causes anger wrath - 1]", "[s wrath same_as anger]", "[s anger is_a bad]", "[s calm is_a good]"]
    if i in (21, 22):
        t.append("[w calm 1]")
    if 23 <= i < 27:  # joy, the good kind of pleasure, makes for virtue
        t += ["[v joy 1 emotion]", "[s joy is_a good]", "[j causes joy virtue - 1 direct]"]
    if 27 <= i < 30:  # pleasure, the passion, undoes it
        t += ["[v pleasure 1 emotion]", "[s pleasure is_a bad]", "[j causes pleasure virtue - -1 direct]"]
    if i in (30, 31):
        t.append("[v pleasure 1 emotion]")
    if i == 32:
        t.append("[v joy 1 emotion]")
    if i == 33:
        t += ["[s virtue is_a good]", "[s self values virtue]"]
    return t


def make(tmp_path: Path, n: int = 36):
    folder = tmp_path / "corpus" / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{i:05d}.txt").write_text(f"# Doc {i}\npassage {i} " + " ".join(tags(i)), encoding="utf-8")
    spec = CorpusSpec(id="synthetic", title="Synthetic", subject="Tester", splitter="documents")
    build(spec, tmp_path, coder=Coder(), log=lambda *_: None)
    return spec, mirror_dir(tmp_path, "synthetic")


def mirror(root: Path) -> dict:
    return json.loads((root / "mirror.json").read_text(encoding="utf-8"))


def test_merge_combines_evidence_and_refits(tmp_path):
    _, root = make(tmp_path)
    eng = mutate.Engine(root)
    before = (root / "mirror.json").read_bytes()
    p = eng.propose_merge(["anger", "wrath"])
    assert (root / "mirror.json").read_bytes() == before  # proposing never touches the Mirror
    m, v = p["mutation"], p["preview"]
    assert m == {"type": "merge", "into": "anger", "members": [{"id": "wrath", "polarity": 1}]}
    c = {x["id"]: x for x in v["causal"]["constraints"]}
    oa = c["reaches|opinion|anger"]
    assert oa["weight"] == 8 and {x["id"] for x in oa["from"]} == {"reaches|opinion|anger", "reaches|opinion|wrath"}
    assert c["reaches|anger|misery"]["weight"] == 6
    assert v["causal"]["n_collapsed"] == 1 and v["causal"]["collapsed"][0]["text"] == "anger raises wrath"
    assert any(x["r"] == "same_as" for x in v["kg"]["collapsed"])
    assert any("say nothing once merged" in w for w in v["warnings"])
    assert {e["text"] for e in v["causal"]["edges_after"]} >= {"opinion raises anger", "anger raises misery"}
    assert {e["text"] for e in v["causal"]["edges_lost"]} == {"anger raises wrath"}  # a self-loop now

    with pytest.raises(mutate.MutationError):
        eng.approve(p["id"], "")
    r = eng.approve(p["id"], "Android K")
    assert r["status"] == "applied"
    after = mirror(root)
    assert "wrath" not in after["nodes"] and "wrath" in after["nodes"]["anger"]["aliases"]
    assert after["nodes"]["anger"]["merged_from"][0]["id"] == "wrath"
    edges = {(e["src"], e["dst"]): e for e in after["causal"]["edges"]}
    assert edges[("opinion", "anger")]["weight"] == 8 and ("anger", "wrath") not in edges
    assert after["meta"]["mutations"] == [p["id"]]
    assert eng.applied()[0]["approver"] == "Android K"

    u = eng.undo("Android K")
    assert u["how"] == "restored" and (root / "mirror.json").read_bytes() == before
    assert eng.get(p["id"])["status"] == "undone"


def test_merge_as_opposite_turns_signs(tmp_path):
    _, root = make(tmp_path)
    p = mutate.Engine(root).propose_merge(["anger", "calm"], opposite=["calm"])
    c = {x["id"]: x for x in p["preview"]["causal"]["constraints"]}
    # opinion lowers calm = opinion raises anger; calm lowers misery = anger raises misery
    assert c["reaches|opinion|anger"]["weight"] == 8 and c["reaches|opinion|anger"]["text"] == "opinion raises anger"
    assert c["reaches|anger|misery"]["weight"] == 7
    after, _ = mutate.evaluate(mirror(root), p["mutation"])
    assert after["nodes"]["anger~"]["label"] == "calm" and after["nodes"]["anger~"]["pole_of"] == "anger"
    assert any(f["s"] == "anger~" and f["r"] == "is_a" and f["o"] == "good" for f in after["facts"])
    assert {v["concept"]: v["direction"] for v in after["values"]}["anger"] == -1  # urging calm is urging less anger


def test_detectors_find_synonyms_opposites_and_senses(tmp_path):
    _, root = make(tmp_path)
    found = mutate.Engine(root).propose()
    by = {(p["source"], json.dumps(p["mutation"].get("members") or p["mutation"]["node"])): p for p in found}
    srcs = {p["source"] for p in found}
    assert {"same_as", "causal_twins", "disjoint_evaluations"} <= srcs
    same = next(p for p in found if p["source"] == "same_as")
    assert {same["mutation"]["into"], same["mutation"]["members"][0]["id"]} == {"anger", "wrath"}
    twins = [p["mutation"] for p in found if p["source"] == "causal_twins"]
    assert any(m["members"][0]["polarity"] == -1 and {m["into"], m["members"][0]["id"]} == {"anger", "calm"} for m in twins)
    sp = next(p for p in found if p["mutation"]["type"] == "split")
    # called good and bad, and said to raise and to lower virtue, in the same passages: one proposal, both reasons
    assert sp["mutation"]["node"] == "pleasure" and "both good and bad" in sp["reason"] and "lowers it in 3" in sp["reason"]
    # 4 + 3 by evidence; then "joy" marks one sense and "pleasure" the other
    assert [len(s["passages"]) for s in sp["mutation"]["senses"]] == [5, 5]
    assert sp["mutation"]["basis"] == {"evidence": 7, "wording": 3}
    assert by and all(p["preview"]["summary"] for p in found)  # every proposal carries a preview
    # the held-out check is left for later, so that finding candidates stays quick
    assert all(p["preview"]["causal"]["holdout_after"] is None for p in found)
    checked = mutate.Engine(root).check(same["id"])
    assert checked["preview"]["causal"]["holdout_after"]["claims"] > 0 and "Held-out claims" in checked["preview"]["summary"]


def test_split_divides_facts_and_resolves_the_contradiction(tmp_path):
    _, root = make(tmp_path)
    eng = mutate.Engine(root)
    p = eng.propose_split("pleasure", [{"label": "joy", "gloss": "the good kind", "aliases": ["joy"]},
                                       {"label": "pleasure as passion", "gloss": "the passion", "aliases": ["pleasure"]}])
    m, v = p["mutation"], p["preview"]
    assert [len(s["passages"]) for s in m["senses"]] == [5, 5] and m["basis"] == {"wording": 10}
    assert any(x["type"] == "disjoint_evaluations" for x in v["kg"]["violations_before"])
    assert not any(x["type"] == "disjoint_evaluations" for x in v["kg"]["violations_after"])
    c = {x["id"]: x for x in v["causal"]["constraints"]}
    assert c["reaches|joy|virtue"]["status"] == "arrow" and c["reaches|joy|virtue"]["text"] == "joy raises virtue"
    assert c["reaches|pleasure_as_passion|virtue"]["text"] == "pleasure as passion lowers virtue"
    assert v["causal"]["tensions_before"] and not v["causal"]["tensions_after"]
    eng.approve(p["id"], "Android K")
    after = mirror(root)
    assert "pleasure" not in after["nodes"] and after["nodes"]["joy"]["split_from"]["id"] == "pleasure"
    assert "pleasure" in after["nodes"]["joy"]["aliases"]


def test_stale_proposals_are_recomputed_not_forced(tmp_path):
    _, root = make(tmp_path)
    eng = mutate.Engine(root)
    a = eng.propose_merge(["anger", "wrath"])
    b = eng.propose_merge(["opinion", "misery"])
    eng.approve(a["id"], "Android K")
    r = eng.approve(b["id"], "Android K")
    assert r["status"] == "stale" and r["proposal"]["base"] != b["base"]
    assert "opinion" in mirror(root)["nodes"] and "misery" in mirror(root)["nodes"]
    assert eng.approve(b["id"], "Android K")["status"] == "applied"
    # a proposal about a concept that has since been merged away no longer applies
    c = eng.propose_merge(["virtue", "calm"])
    eng.approve(eng.propose_merge(["anger", "calm"], opposite=["calm"])["id"], "Android K")
    assert eng.approve(c["id"], "Android K")["status"] == "invalid"
    assert "calm" not in mirror(root)["nodes"]


def test_rejected_detector_proposals_stay_rejected(tmp_path):
    _, root = make(tmp_path)
    eng = mutate.Engine(root)
    first = eng.propose()
    eng.reject(first[0]["id"])
    again = {p["id"]: p for p in eng.propose()}
    assert again[first[0]["id"]]["status"] == "rejected"


def test_approved_mutations_survive_a_rebuild_and_undo_after_one(tmp_path):
    spec, root = make(tmp_path)
    eng = mutate.Engine(root)
    p = eng.propose_merge(["anger", "wrath"])
    eng.approve(p["id"], "Android K")
    build(spec, tmp_path, coder=Coder(), log=lambda *_: None)
    m = mirror(root)
    assert "wrath" not in m["nodes"] and m["meta"]["mutations"] == [p["id"]]
    build(spec, tmp_path, coder=Coder(), settings={"folds": 4}, log=lambda *_: None)  # the Mirror on disk changes
    assert eng.undo("Android K")["how"] == "rebuild"
    build(spec, tmp_path, coder=Coder(), log=lambda *_: None)
    assert "wrath" in mirror(root)["nodes"]


def test_claude_reads_the_senses(tmp_path):
    _, root = make(tmp_path)
    calls = []

    def reader(subject, label, gloss, passages, hint, fixed):
        calls.append((label, len(passages), hint))
        return {"senses": [{"label": "joy", "gloss": "gladness at acting well"},
                           {"label": "pleasure", "gloss": "the passion"}],
                "assign": [{"passage": p["id"], "sense": 0 if "joy" in p["text"] else 1} for p in passages]}

    p = mutate.Engine(root).propose_split("pleasure", reader=reader)
    assert calls and calls[0][0] == "pleasure" and "two senses" in calls[0][2]
    assert [len(s["passages"]) for s in p["mutation"]["senses"]] == [5, 5]
    assert p["reason"].startswith("senses read by Claude")

    def one(subject, label, gloss, passages, hint, fixed):
        return {"senses": [{"label": "pleasure", "gloss": ""}], "assign": []}

    with pytest.raises(mutate.MutationError):
        mutate.Engine(root).propose_split("pleasure", reader=one)


def test_bad_mutations_are_refused(tmp_path):
    _, root = make(tmp_path)
    data = mirror(root)
    for ids in (["anger"], ["anger", "nothing"], ["anger", "self"], ["anger", "good"]):
        with pytest.raises(mutate.MutationError):
            mutate.merge(data, ids)


def test_api(tmp_path):
    make(tmp_path)
    client = TestClient(create_app(tmp_path))
    found = client.post("/api/mutate/propose", json={}).json()
    assert found["proposals"] and all(p["status"] == "pending" for p in found["proposals"])
    p = client.post("/api/mutate/merge", json={"ids": ["anger", "wrath"]}).json()
    assert p["preview"]["summary"].startswith("Merge wrath into anger")
    assert client.get(f"/api/mutate/{p['id']}").json()["id"] == p["id"]
    assert client.post(f"/api/mutate/{p['id']}/approve", json={"approver": ""}).status_code == 400
    r = client.post(f"/api/mutate/{p['id']}/approve", json={"approver": "Android K"}).json()
    assert r["status"] == "applied"
    assert "wrath" not in {c["id"] for c in client.get("/api/concepts").json()}
    listing = client.get("/api/mutate").json()
    assert listing["applied"][0]["id"] == p["id"]
    assert client.post("/api/mutate/undo", json={"approver": "Android K"}).json()["how"] == "restored"
    assert "wrath" in {c["id"] for c in client.get("/api/concepts").json()}
    s = client.post("/api/mutate/split", json={"node": "pleasure"}).json()
    assert s["mutation"]["type"] == "split"
    assert client.post("/api/mutate/split", json={"node": "nothing"}).status_code == 400


def test_previews_fit_the_approved_answers_too(tmp_path):
    from androidmirror.mirror import answers

    _, root = make(tmp_path)
    answers.save(root, {"proposals": [{"id": "a1", "status": "approved", "question": "Does wrath breed opinion?",
                                       "answer": "yes", "approved_by": "Android K", "changes": [
                                           {"type": "judgment", "kind": "causes", "x": "wrath", "y": "virtue",
                                            "sign": -1, "scope": "direct"}]}]})
    eng = mutate.Engine(root)
    p = eng.propose_merge(["anger", "wrath"])
    # the preview is fitted as the Mirror will load: the answer stays on wrath, so it is flagged
    assert any("approved answer" in w for w in p["preview"]["warnings"])
    eng.approve(p["id"], "Android K")
    edges = {(e["src"], e["dst"]) for e in mirror(root)["causal"]["edges"]}
    assert ("wrath", "virtue") in edges


def test_split_without_senses_says_what_to_give(tmp_path):
    _, root = make(tmp_path)
    eng = mutate.Engine(root)
    w = eng.words("pleasure")
    assert w["passages"] == 10 and {x["word"] for x in w["words"]} == {"joy", "pleasure"}
    with pytest.raises(mutate.MutationError, match="Name at least two senses"):
        eng.propose_split("anger")  # nothing in the Mirror splits anger by itself
    p = eng.propose_split("anger", [{"label": "anger", "aliases": ["anger"]}, {"label": "fury", "aliases": ["fury"]}])
    assert any("do not tell the passages apart" in x for x in p["preview"]["warnings"])
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/mutate/words/pleasure").json()["label"] == "pleasure"
    r = client.post("/api/mutate/split", json={"node": "pleasure", "senses": [
        {"label": "joy", "aliases": ["joy"]}, {"label": "passion", "aliases": ["pleasure"]}]}).json()
    assert [len(s["passages"]) for s in r["mutation"]["senses"]] == [5, 5]
