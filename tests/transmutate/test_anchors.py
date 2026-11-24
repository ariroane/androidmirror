"""Anchors in WordNet and merging through them: a small made-up WordNet, the two synthetic Mirrors of test_transmutate."""

import gzip
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from androidmirror.cli import main
from androidmirror.mirror import anchor_eval, anchors, pivot, wordnet
from androidmirror.mirror import transmutate as tm
from androidmirror.mirror.store import Mirror
from androidmirror.server import create_app
from tests.transmutate.test_transmutate import make


def syn(n: int) -> str:
    return f"oewn-{n:08d}-n"


def entry(n, words, hyper=(), ant=(), lex="noun.state", definition=""):
    return syn(n), {"pos": "n", "lex": lex, "def": definition or f"sense {n}", "words": list(words),
                    "hyper": [syn(h) for h in hyper], "ant": [syn(a) for a in ant]}


SYNSETS = dict([
    entry(1, ["entity"], lex="noun.Tops"), entry(2, ["abstraction"], [1], lex="noun.Tops"),
    entry(3, ["state"], [2]), entry(4, ["feeling"], [3], lex="noun.feeling"), entry(5, ["emotion"], [4], lex="noun.feeling"),
    entry(10, ["alpha"], [5]), entry(11, ["gamma"], [5]), entry(12, ["eta", "subgamma"], [11]),
    entry(13, ["zeta", "calm"], [5], ant=[14]), entry(14, ["unease", "agitation"], [5], ant=[13]),
    entry(20, ["justice"], [3], lex="noun.attribute", definition="the quality of being just"),
    entry(21, ["justice", "judge"], [1], lex="noun.person", definition="a public official who decides cases"),
    entry(30, ["beta", "theta"], [5]), entry(40, ["delta"], [5]), entry(41, ["epsilon"], [5]),
])
LEMMAS: dict = {}
for sid, s in SYNSETS.items():
    for w in s["words"]:
        LEMMAS.setdefault(w, []).append([sid, "n"])
INDEX = {"edition": "test", "lemmas": LEMMAS, "synsets": SYNSETS}
A_SENSES = {"alpha": 10, "gamma": 11, "zeta": 13, "justice": 20, "beta": 30, "delta": 40, "epsilon": 41}
B_SENSES = {"alpha": 10, "gamma": 11, "eta": 12, "unease": 14, "justice": 21, "theta": 30, "delta": 40, "epsilon": 41}


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    data = make(tmp_path_factory.mktemp("anch"))
    root = data / "wordnet"
    root.mkdir()
    with gzip.open(root / f"oewn-{wordnet.EDITION}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(INDEX, f)
    return data


def anchor_all(data: Path, approver: str = "Tester") -> None:
    wn = wordnet.WordNet(INDEX)
    for name, senses in (("synthetic", A_SENSES), ("second", B_SENSES)):
        root = data / "mirror" / name
        m = json.loads((root / "mirror.json").read_text(encoding="utf-8"))
        for concept, n in senses.items():
            anchors.set_anchor(root, m, wn, concept, wordnet.curie(syn(n)), approver=approver)


def test_wordnet_index_from_xml():
    xml = b"""<?xml version="1.0"?><LexicalResource><Lexicon id="oewn">
    <LexicalEntry id="e1"><Lemma writtenForm="calm" partOfSpeech="n"/><Sense id="s1" synset="oewn-1-n">
      <SenseRelation relType="antonym" target="s2"/></Sense></LexicalEntry>
    <LexicalEntry id="e2"><Lemma writtenForm="Agitation" partOfSpeech="n"/><Sense id="s2" synset="oewn-2-n"/></LexicalEntry>
    <Synset id="oewn-1-n" partOfSpeech="n" lexfile="noun.state"><Definition>a quiet state</Definition>
      <SynsetRelation relType="hypernym" target="oewn-3-n"/></Synset>
    <Synset id="oewn-2-n" partOfSpeech="n" lexfile="noun.state"><Definition>a troubled state</Definition></Synset>
    <Synset id="oewn-3-n" partOfSpeech="n" lexfile="noun.Tops"><Definition>a condition</Definition></Synset>
    </Lexicon></LexicalResource>"""
    wn = wordnet.WordNet(wordnet.index_xml(io.BytesIO(xml)))
    assert wn.senses("Calm") == ["oewn-1-n"] and wn.senses("agitations") == ["oewn-2-n"]
    assert wn.antonyms("oewn-1-n") == {"oewn-2-n"} and wn.subsumes("oewn-3-n", "oewn-1-n") == 1
    assert wordnet.variants("the passions") == ["the passions", "passions", "the passion", "passion"]


def test_lexical_proposals_need_a_named_approver(tmp_path):
    wn = wordnet.WordNet(INDEX)
    mirror = {"meta": {"corpus": {"subject": "S"}}, "nodes": {
        "justice": {"label": "justice", "kind": "disposition", "support": 3},
        "theta": {"label": "theta", "kind": "state", "support": 2},
        "sub": {"label": "fear of gamma", "kind": "emotion", "support": 1},
        "unknown": {"label": "prohairesis", "kind": "faculty", "support": 5}}}
    made = {p["concept"]: p for p in anchors.propose(tmp_path, mirror, wn, log=quiet)}
    assert set(made) == {"justice", "theta"}  # a phrase's head word needs Claude; an unknown word gets nothing
    assert made["justice"]["predicate"] == "closeMatch" and made["justice"]["object"] == "oewn:00000020-n"
    assert made["theta"]["predicate"] == "exactMatch" and made["theta"]["up"]["oewn:00000005-n"] == 1
    assert [c["head"] for c in anchors.lexical(wn, mirror["nodes"]["sub"])] == [True]
    assert anchors.approved(anchors.load(tmp_path)) == {}  # nothing is used until approved
    with pytest.raises(ValueError):
        anchors.approve(tmp_path, ["all"], "  ")
    anchors.approve(tmp_path, [made["theta"]["id"]], "Ann")
    assert list(anchors.approved(anchors.load(tmp_path))) == ["theta"]
    tsv = anchors.tsv_path(tmp_path).read_text(encoding="utf-8")
    assert "skos:exactMatch\toewn:00000030-n" in tsv and "Ann" in tsv and "justice" not in tsv.split("subject_id")[1]
    mine = anchors.set_anchor(tmp_path, mirror, wn, "theta", "none", approver="Ann")
    book = anchors.load(tmp_path)
    assert anchors.approved(book)["theta"]["predicate"] == "noMatch"
    assert [p["status"] for p in book["proposals"] if p["concept"] == "theta"] == ["superseded", "approved"]
    assert "sssom:NoTermFound" in anchors.tsv_path(tmp_path).read_text(encoding="utf-8") and mine["status"] == "approved"


def test_relation_verdicts():
    wn = wordnet.WordNet(INDEX)
    rec = lambda n, pred="exactMatch", pol=1: anchors.make(wn, "c", {"label": "c"}, {  # noqa: E731
        "predicate": pred, "synset": syn(n) if n else "", "polarity": pol, "justification": "you"})
    assert pivot.relation(rec(10), rec(10))[0] == "same"
    assert pivot.relation(rec(13), rec(14))[0] == "opposite"
    assert pivot.relation(rec(13, pol=-1), rec(14))[0] == "same"
    assert pivot.relation(rec(12), rec(11))[0] == "kind_of"
    assert pivot.relation(rec(20), rec(21))[0] == "disjoint"
    assert pivot.relation(rec(10), rec(11))[0] == "near"
    assert pivot.relation(rec(0, "noMatch"), rec(10))[0] == "none_fits"
    assert pivot.relation(None, rec(10))[0] == "unknown"


def test_transmutate_reads_anchors_when_both_have_them(built):
    a, b = tm.load_source(built, "synthetic"), tm.load_source(built, "second")
    plain = tm.align(a, b)
    assert not plain["pivot"]["used"] and ("justice", "justice") in {(p["a"], p["b"]) for p in plain["pairs"]}
    before = a["hash"]
    anchor_all(built)
    a, b = tm.load_source(built, "synthetic"), tm.load_source(built, "second")
    assert a["hash"] != before  # approving anchors makes an older proposal stale
    al = tm.align(a, b)
    pairs = {(p["a"], p["b"]): p for p in al["pairs"]}
    assert pairs[("beta", "theta")]["how"] == "anchor"  # same sense, different names
    assert pairs[("zeta", "unease")]["polarity"] == -1  # antonyms, without --opposite
    assert ("justice", "justice") not in pairs
    blocked = {(x["a"], x["b"]): x for x in al["pivot"]["blocked"]}
    assert blocked[("justice", "justice")]["verdict"] == "disjoint"
    links = {(x["s"], x["o"]) for x in al["pivot"]["scaffold"]}
    assert ("second:eta", "gamma") in links  # eta's sense is a kind of gamma's
    assert not any(o in ("second:justice", "justice") for _, o in links)
    off = tm.align(a, b, use_anchors=False)
    assert not off["pivot"]["used"] and ("justice", "justice") in {(p["a"], p["b"]) for p in off["pairs"]}
    yours = tm.align(a, b, add=[tm.parse_pair("justice=justice")])  # your pair is never blocked
    assert ("justice", "justice") in {(p["a"], p["b"]) for p in yours["pairs"]}


def test_scaffold_is_drawn_but_never_held(built):
    anchor_all(built)
    p = tm.propose(built, "synthetic", "second", settings={"samples": 5}, log=quiet)
    assert p["pivot"]["used"] and p["stats"]["combined"]["scaffold"] >= 1
    assert "Blocked merges" in (built / "transmutations" / "synthetic+second" / "proposal.md").read_text(encoding="utf-8")
    m = tm.approve(built, "synthetic", "second", log=quiet)
    assert any(x["s"] == "second:eta" and x["o"] == "gamma" for x in m["scaffold"])
    assert not any(f["s"] == "second:eta" and f["r"] == "is_a" and f["o"] == "gamma" for f in m["facts"])
    view = Mirror.load(built / "mirror" / "synthetic+second")
    assert any(e.get("scaffold") for e in view.graph("kg")["edges"])
    assert view.node("second:eta")["scaffold"][0]["o_label"] == "gamma"
    assert ("second:eta", "is_a", "gamma") not in view.kb.closure


def test_evaluation_compares_the_two_alignments(built):
    anchor_all(built)
    r = anchor_eval.compare(built, "synthetic", "second", samples=2, log=quiet)
    cur, anc = r["stats"]["current"], r["stats"]["anchored"]
    assert anc["blocked"] >= 1 and cur["blocked"] == 0 and anc["scaffold_links"] >= 1
    assert any("beta = theta" in x for x in r["new_pairs"])
    assert (built / "evaluations" / "synthetic+second" / "report.md").exists()
    assert not (built / "mirror" / "synthetic+second-eval").exists()


def test_split_half_knows_the_answer(built):
    r = anchor_eval.split_half(built, "synthetic", rename_share=1.0, homonyms=0, samples=2, log=quiet)
    s = r["stats"]
    assert s["current"]["true_pairs"] > 0 and r["settings"]["renamed"] >= 1
    assert s["anchored"]["renamed_recovered"] >= s["current"]["renamed_recovered"]
    assert s["anchored"]["right_merges"] >= s["current"]["right_merges"]
    half2 = json.loads((built / "evaluations" / "split-half-synthetic" / "synthetic-half2" / "mirror.json").read_text())
    assert any(n.get("renamed_from") for n in half2["nodes"].values())
    book = anchors.load(built / "evaluations" / "split-half-synthetic" / "synthetic-half2")
    assert {p["approved_by"] for p in book["proposals"] if p["status"] == "approved"} == {anchor_eval.AUTO}


def test_half_and_rename_keep_the_mirror_whole(built):
    src = tm.load_source(built, "synthetic")
    keep = {p["id"] for p in src["passages"][::2]}
    h = anchor_eval.half(src, keep, "h")
    m = h["mirror"]
    assert all(e["passage"] in keep or e["passage"] == "codebook" for f in m["facts"] for e in f["evidence"])
    assert {j["passage"] for j in m["judgments"]} <= keep
    r = anchor_eval.rename(m, {"gamma": {"id": "subgamma__r", "label": "subgamma"}})
    assert "gamma" not in r["nodes"] and r["nodes"]["subgamma__r"]["label"] == "subgamma"
    assert not any("gamma" in (j["x"], j["y"], j["z"]) for j in r["judgments"])
    assert any("subgamma__r" in (j["x"], j["y"]) for j in r["judgments"])


def test_anchor_routes_and_cli(built, capsys):
    client = TestClient(create_app(built))
    r = client.get("/api/anchors", params={"mirror": "second"})
    assert r.status_code == 200 and "summary" in r.json()
    assert client.post("/api/anchors/approve", json={"mirror": "second", "ids": ["all"], "approver": ""}).status_code == 400
    assert client.post("/api/anchors/reject", json={"mirror": "second"}).status_code == 404  # never rejects all by default
    main(["--data", str(built), "anchors", "senses", "justice"])
    assert "oewn:00000021-n" in capsys.readouterr().out
