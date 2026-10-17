"""Transmutate: two synthetic Mirrors built offline, combined, scored and trimmed."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from androidmirror.cli import main
from androidmirror.mirror import transmutate as tm
from androidmirror.mirror.build import build
from androidmirror.mirror.corpus import CorpusSpec
from androidmirror.server import create_app
from tests.mirror.fake import FakeCoder, write_corpus

SECOND = ["alpha", "gamma", "delta", "epsilon", "eta", "theta", "unease", "justice"]


def second_tags(i: int) -> list[str]:
    """A second subject: agrees on gamma -> delta, adds eta -> alpha, and disagrees in three ways."""
    tags = []
    if i < 6:
        tags.append("[j causes gamma delta - 1 direct]")  # also said by the first: corroborated
    if 6 <= i < 14:
        tags.append("[j causes eta alpha - 1 direct]")  # new, and strong
    if 14 <= i < 17:
        tags.append("[j causes delta epsilon - 1 direct]")  # the first says it lowers epsilon
    if 17 <= i < 19:
        tags.append("[j causes epsilon alpha - 1]")  # closes a cycle with the first's chain
    if i == 19:
        tags.append("[j causes eta theta - 1 direct]")  # said once: weak
    if 20 <= i < 24:
        tags.append("[j causes unease alpha - -1 direct]")  # unease is the opposite of the first's zeta
    if i == 24:
        tags += ["[s justice is_a bad]", "[s eta is_a good]"]
    if 25 <= i < 28:
        tags.append("[s eta requires gamma]")
    return tags


def write_second(folder: Path, n: int = 40) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        tags = [f"[v {v} 1 state]" for v in SECOND] + second_tags(i)
        (folder / f"{i:05d}.txt").write_text(f"# Doc {i}\ngroup: {i // 20}\nentry {i} " + " ".join(tags), encoding="utf-8")


def make(tmp_path: Path) -> Path:
    write_corpus(tmp_path / "corpus" / "synthetic", 80)
    write_second(tmp_path / "corpus" / "second")
    for cid, subject in (("synthetic", "Tester"), ("second", "Committee")):
        spec = CorpusSpec(id=cid, title=cid.title(), subject=subject, splitter="documents")
        build(spec, tmp_path, coder=FakeCoder(), log=lambda *_: None)
    return tmp_path


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return make(tmp_path_factory.mktemp("tm"))


def quiet(*_):
    pass


def proposal(data: Path, **kw):
    kw.setdefault("add", [tm.parse_pair("zeta=unease", -1)])
    kw.setdefault("settings", {"samples": 30})
    return tm.propose(data, "synthetic", "second", log=quiet, **kw)


def test_alignment_pairs_same_names_and_yours(built):
    a, b = tm.load_source(built, "synthetic"), tm.load_source(built, "second")
    found = {(p["a"], p["b"], p["polarity"]) for p in tm.candidates(a, b)}
    assert {("alpha", "alpha", 1), ("gamma", "gamma", 1), ("delta", "delta", 1), ("epsilon", "epsilon", 1),
            ("justice", "justice", 1)} <= found
    assert not any(p[1] in ("eta", "theta", "unease") for p in found)
    al = tm.align(a, b, add=[tm.parse_pair("zeta=unease", -1)], drop=["justice"])
    pairs = {(p["a"], p["b"], p["polarity"]) for p in al["pairs"]}
    assert ("zeta", "unease", -1) in pairs and not any(p[0] == "justice" for p in pairs)
    assert [p["b"] for p in al["dropped"]] == ["justice"]
    with pytest.raises(KeyError):
        tm.align(a, b, add=[tm.parse_pair("nothing=unease")])


def test_combining_is_consistent_and_reports_where_the_mirrors_meet(built):
    a, b = tm.load_source(built, "synthetic"), tm.load_source(built, "second")
    al = tm.align(a, b, add=[tm.parse_pair("zeta=unease", -1)])
    m = tm.combine(a, b, al["pairs"])["mirror"]
    assert "gamma" in m["nodes"] and m["nodes"]["gamma"]["sources"] == {"synthetic": "gamma", "second": "gamma"}
    assert "second:eta" in m["nodes"] and "synthetic:beta" in m["nodes"]
    assert m["nodes"]["zeta~"]["pole_of"] == "zeta"  # unease became zeta's opposite pole
    # unease lowers alpha, so zeta raises it: the sign flips through the opposite pairing
    flipped = [j for j in m["judgments"] if j["passage"].startswith("second:") and (j["x"], j["y"]) == ("zeta", "alpha")]
    assert flipped and {j["sign"] for j in flipped} == {1}

    model = tm.fit(m["judgments"])
    rows = {c["id"]: c for c in model["constraints"]}
    assert rows["reaches|gamma|delta"]["weight"] == 33 + 6
    assert rows["reaches|second:eta|alpha"]["status"] == "arrow"
    cross = tm.crossings(m, model)
    assert any(x["id"] == "reaches|gamma|delta" for x in cross["agreed"])
    kinds = {x["id"]: x["kind"] for x in cross["causal"]}
    assert kinds["reaches|delta|epsilon"] == "sign"
    assert kinds["reaches|epsilon|alpha"] == "cycle"  # the second's arrow against the first's chain
    assert kinds["reaches|zeta|alpha"] == "denied"  # the first says zeta makes no difference to alpha
    assert "reaches|epsilon|gamma" not in kinds  # the first's own slip was already a cycle on its own
    assert any(v["type"] == "disjoint_evaluations" and v["node"] == "justice" for v in cross["kg"])


def test_propose_scores_and_trims_but_writes_no_mirror(built):
    p = proposal(built)
    assert not (built / "mirror" / "synthetic+second").exists()
    assert (built / "transmutations" / "synthetic+second" / "proposal.md").read_text(encoding="utf-8").startswith("# Transmutation")
    trims = {t["id"]: t for t in p["trims"]}
    assert "c:reaches|second:eta|second:theta" in trims  # said once
    assert "n:second:theta" in trims
    assert "f:second:eta|is_a|good" in trims and "f:second:eta|requires|gamma" not in trims
    assert "c:reaches|synthetic:beta|zeta" in trims and trims["c:reaches|synthetic:beta|zeta"]["what"] == "judgment"  # Epicurus's view
    for strong in ("reaches|gamma|delta", "reaches|alpha|gamma", "reaches|second:eta|alpha"):
        assert f"c:{strong}" not in trims and p["fitness"][strong]["model"] >= 0.9
    # the first's denial and the second's claim are equally strong, so samples split; the denial still clears the bar
    assert "c:no_path|zeta|alpha" not in trims and 0.6 <= p["fitness"]["no_path|zeta|alpha"]["model"] < 0.9
    # a strong claim the model rejects stays, so the trimmed Mirror still shows the tension
    assert "c:reaches|zeta|alpha" not in trims
    assert p["stats"]["trimmed"]["arrows"] <= p["stats"]["combined"]["arrows"] and not p["stats"]["arrows_new"]
    assert [t["n"] for t in p["trims"]] == list(range(1, len(p["trims"]) + 1))


def test_approve_writes_the_combined_mirror_and_archives_trims(built):
    p = proposal(built)
    n_theta_fact = next(t["n"] for t in p["trims"] if t["id"] == "f:second:eta|is_a|good")
    m = tm.approve(built, "synthetic", "second", keep=[str(n_theta_fact)], log=quiet)
    out = built / "mirror" / "synthetic+second"
    assert m["meta"]["transmutation"]["spared"] == [str(n_theta_fact)]
    assert "second:theta" not in m["nodes"]
    assert any((f["s"], f["r"], f["o"]) == ("second:eta", "is_a", "good") for f in m["facts"])
    archive = json.loads((out / "trimmed.json").read_text(encoding="utf-8"))
    assert "second:theta" in archive["nodes"] and archive["judgments"]
    arrows = {(e["src"], e["dst"], e["sign"]) for e in m["causal"]["edges"]}
    assert {("alpha", "gamma", 1), ("gamma", "delta", 1), ("delta", "epsilon", -1), ("second:eta", "alpha", 1)} <= arrows

    client = TestClient(create_app(built))
    assert "synthetic+second" in client.get("/api/status").json()["built"]
    meta = client.get("/api/meta", params={"corpus": "synthetic+second"}).json()
    assert meta["corpus"]["subject"] == "Tester and Committee"
    node = client.get("/api/node/gamma", params={"corpus": "synthetic+second"}).json()
    assert {m["passage"].split(":")[0] for m in node["mentions"]} == {"synthetic", "second"}
    pid = node["mentions"][-1]["passage"]
    assert client.get(f"/api/passage/{pid}", params={"corpus": "synthetic+second"}).json()["text"]

    # approving again archives the earlier result instead of overwriting it
    tm.approve(built, "synthetic", "second", log=quiet)
    assert list((out / "history").iterdir())


def test_approve_refuses_a_stale_proposal_and_unknown_items(built, tmp_path):
    data = make(tmp_path)
    proposal(data)
    with pytest.raises(KeyError):
        tm.approve(data, "synthetic", "second", keep=["9999"], log=quiet)
    path = data / "mirror" / "second" / "mirror.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed since"):
        tm.approve(data, "synthetic", "second", log=quiet)
    assert not (data / "mirror" / "synthetic+second").exists()


def test_cli(built, capsys):
    main(["--data", str(built), "transmutate", "propose", "synthetic", "second", "--opposite", "zeta=unease",
          "--samples", "10"])
    out = capsys.readouterr().out
    assert "Nothing has been changed" in out and "is the opposite of" in out
    with pytest.raises(SystemExit):
        main(["--data", str(built), "transmutate", "propose", "synthetic", "nowhere"])
    main(["--data", str(built), "transmutate", "approve", "synthetic", "second"])
    assert "wrote" in capsys.readouterr().out


def test_api(tmp_path):
    data = make(tmp_path)
    client = TestClient(create_app(data))
    assert client.get("/api/transmutate", params={"a": "synthetic", "b": "second"}).status_code == 404
    r = client.post("/api/transmutate/propose", json={"a": "synthetic", "b": "second", "opposites": ["zeta=unease"]})
    assert r.status_code == 200 and r.json()["trims"]
    assert "synthetic+second" not in client.get("/api/status").json()["built"]
    assert client.get("/api/transmutate", params={"a": "synthetic", "b": "second"}).json()["alignment"]
    assert client.post("/api/transmutate/propose", json={"a": "synthetic", "b": "nowhere"}).status_code == 400
    r = client.post("/api/transmutate/approve", json={"a": "synthetic", "b": "second", "keep": ["1"]})
    assert r.json()["built"] == "synthetic+second" and r.json()["transmutation"]["spared"] == ["1"]
    assert "synthetic+second" in client.get("/api/status").json()["built"]
