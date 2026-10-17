import copy
import json
import os

from fastapi.testclient import TestClient

from androidmirror.mirror import derange
from androidmirror.mirror.build import mirror_dir
from androidmirror.mirror.causal import fit, public
from androidmirror.mirror.kg import aggregate
from androidmirror.mirror.store import Mirror
from androidmirror.server import create_app

from .test_build import make
from .test_units import J


def mirror(judgments, values=(), stmts=(), support=None):
    nodes = {n: {"label": n, "kind": "state", "support": (support or {}).get(n, 1)}
             for n in {j[k] for j in judgments for k in ("x", "y", "z") if j[k]} | {s[k] for s in stmts for k in ("s", "o")}}
    data = {"meta": {"corpus": {"id": "t", "subject": "Tester"}}, "nodes": nodes, "facts": aggregate(list(stmts)),
            "judgments": judgments, "values": list(values), "causal": public(fit(judgments)), "passage_ids": []}
    return Mirror(data, [])


def V(concept, more):
    return {"concept": concept, "more": more, "less": 0, "direction": 1, "evidence": []}


def S(s, r, o, p):
    return {"s": s, "r": r, "o": o, "neg": False, "passage": p, "locator": p, "quote": "q", "modality": "asserts",
            "attribution": "self"}


def chain():
    # practice -> judgment -> calm, one passage each; habit -> calm in ten; calm is what the subject urges
    return [J("causes", "practice", "judgment", scope="direct", p="a"), J("causes", "judgment", "calm", scope="direct", p="b")] + [
        J("causes", "habit", "calm", scope="direct", p=f"h{i}") for i in range(10)]


def test_the_thinnest_support_under_an_end_comes_first():
    m = mirror(chain(), values=[V("calm", 5)])
    out = derange.run(m)
    end = out["causal"]["subgraphs"][0]
    assert (end["id"], end["importance"], end["size"]) == ("calm", 5, 3)
    top = out["causal"]["findings"][0]
    assert [(b["src"], b["dst"]) for b in top["basis"]] == [("judgment", "calm")]
    assert top["strength"] == 1 and top["passages"] == ["b"]
    hit = top["ends"][0]
    assert hit["end"] == "calm" and sorted(hit["lost"]) == ["judgment", "practice"] and hit["of"] == 3
    assert "judgment raises calm" in top["probe"] and "1 passage" in top["probe"]
    # ten passages under habit -> calm: weaker than one, so it ranks below
    ids = [f["id"] for f in out["causal"]["findings"]]
    assert ids.index("cut:judgment>calm") < ids.index("cut:habit>calm")
    assert end["weakest"] == {"finding": "cut:judgment>calm", "strength": 1}


def test_an_arrow_the_user_settled_is_never_cut():
    m = mirror(chain(), values=[V("calm", 5)])
    for c in m.causal["constraints"]:
        c["pinned"] = c["id"] == "reaches|judgment|calm"  # as the Question engine marks an approved answer
    ids = [f["id"] for f in derange.run(m)["causal"]["findings"]]
    assert "cut:judgment>calm" not in ids and "cut:practice>judgment" in ids
    # a ruling judgment (Question engine) bears only on the constraint it names
    assert derange.judgment_constraints({"kind": "ruling", "target": "reaches|a|b"}) == {"reaches|a|b"}


def test_what_the_subjects_other_words_carry_does_not_fall():
    # "practice brings calm, in the end" (said once) follows from the chain; without judgment -> calm it is drawn itself
    js = chain() + [J("causes", "practice", "calm", p="c")]
    m = mirror(js, values=[V("calm", 5)])
    f = next(f for f in derange.run(m)["causal"]["findings"] if f["id"] == "cut:judgment>calm")
    assert f["ends"][0]["lost"] == ["judgment"]
    assert f["now_drawn"] == ["practice → calm"]


def test_a_contested_arrow_says_so_and_takes_all_its_passages_with_it():
    # anger lowers calm, through opinion (1 passage each); the subject also says once "opinion raises calm"
    js = [J("causes", "anger", "opinion", scope="direct", p="a"), J("causes", "opinion", "calm", -1, scope="direct", p="b"),
          J("causes", "opinion", "calm", -1, scope="direct", p="b2"), J("causes", "opinion", "calm", 1, scope="direct", p="c"),
          J("causes", "anger", "calm", -1, p="d")]
    m = mirror(js, values=[V("calm", 3)])
    out = derange.run(m)
    f = next(f for f in out["causal"]["findings"] if f["id"] == "cut:opinion>calm")
    assert f["basis"][0]["sign_conflict"]
    assert f["strength"] == 3  # every held passage behind the arrow goes with it, whichever sign it gave


def test_ends_fall_back_to_the_most_explained_effects_and_empty_models_are_fine():
    m = mirror(chain())
    assert derange.run(m)["causal"]["subgraphs"][0]["why"] == "explained"
    empty = mirror([])
    out = derange.run(empty)
    assert out["causal"]["findings"] == [] and out["conceptual"]["findings"] == []


def test_a_premise_said_once_that_much_rests_on():
    stmts = [S("justice", "is_a", "virtue", "p1")] + [S("virtue", "is_a", "good", f"q{i}") for i in range(5)] + [
        S("virtue", "requires", "reason", f"r{i}") for i in range(5)]
    m = mirror([], stmts=stmts, support={"justice": 40, "virtue": 30, "reason": 2})
    m.nodes["good"] = {"label": "good", "kind": "class"}
    out = derange.run(m)["conceptual"]
    assert [s["id"] for s in out["subgraphs"]][:2] == ["justice", "virtue"]
    top = out["findings"][0]
    assert top["fact"] == "justice|is_a|virtue" and top["strength"] == 1
    hit = next(h for h in top["concepts"] if h["concept"] == "justice")
    assert hit["n_lost"] == hit["of"] == 3  # is a virtue, is a good, requires reason
    assert "justice is a virtue" in top["probe"]


def test_derange_never_changes_the_mirror(tmp_path):
    make(tmp_path)
    root = mirror_dir(tmp_path, "synthetic")
    raw = (root / "mirror.json").read_bytes()
    m = Mirror.load(root)
    before = copy.deepcopy((m.causal, m.facts, m.judgments, m.values))
    out = derange.report(root, m)
    assert (root / "mirror.json").read_bytes() == raw
    assert (m.causal, m.facts, m.judgments, m.values) == before
    assert out["causal"]["findings"] and out["conceptual"]["findings"]
    assert derange.saved(root)["stats"] == json.loads(json.dumps(out["stats"]))


def test_api_and_cli(tmp_path, capsys):
    from androidmirror.cli import main

    make(tmp_path)
    root = mirror_dir(tmp_path, "synthetic")
    client = TestClient(create_app(tmp_path))
    d = client.get("/api/derange").json()
    assert d["causal"]["subgraphs"][0]["id"] == "epsilon" and d["labels"]["epsilon"] == "epsilon"
    assert (root / "derange.json").exists()
    spec = tmp_path / "synthetic.json"
    spec.write_text('{"id": "synthetic", "title": "S", "subject": "T", "splitter": "documents"}')
    main(["--data", str(tmp_path), "derange", "--corpus", str(spec), "--top", "2"])
    printed = capsys.readouterr().out
    assert "Is it so that delta lowers epsilon?" in printed and "Conceptual" in printed
    # a re-fit makes the saved report stale
    os.utime(root / "mirror.json", (1, 1))
    assert derange.saved(root) is None
