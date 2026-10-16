import math

from androidmirror.mirror import codebook as cb
from androidmirror.mirror.causal.citest import chi2_sf, g2_test
from androidmirror.mirror.causal.ida import possible_effects
from androidmirror.mirror.causal.pc import apply_pins, tiers_to_pins
from androidmirror.mirror.causal.pdag import PDAG
from androidmirror.mirror.causal.selection import select
from androidmirror.mirror.corpus import split_documents, split_paragraphs
from androidmirror.mirror.kg import KnowledgeBase, aggregate
from androidmirror.mirror.sources import fomc


def test_chi2_tail():
    assert abs(chi2_sf(3.841459, 1) - 0.05) < 1e-4
    assert abs(chi2_sf(9.487729, 4) - 0.05) < 1e-4
    assert abs(chi2_sf(30.0, 10) - 0.000857) < 1e-5


def test_g2_flags_too_little_data():
    data = {"x": [1, 0, -1, 1], "y": [0, 1, 1, -1]}
    t = g2_test(data, "x", "y")
    assert not t.enough_data and t.p == 1.0


def chain(edges, nodes=None):
    g = PDAG(sorted(nodes or {n for e in edges for n in e}))
    for a, b in edges:
        g.add_undirected(a, b)
    return g


def test_class_size_counts():
    assert chain([("a", "b"), ("b", "c")]).class_size()["count"] == 3
    assert chain([("a", "b"), ("b", "c"), ("a", "c")]).class_size()["count"] == 6
    assert chain([("a", "b"), ("b", "c"), ("c", "d")]).class_size()["count"] == 4
    assert chain([("h", "a"), ("h", "b"), ("h", "c")]).class_size()["count"] == 4
    assert chain([("a", "b"), ("c", "d")]).class_size()["count"] == 4


def test_meek_rules():
    g = chain([("b", "c")], nodes="abc")
    g.directed.add(("a", "b"))
    assert g.meek() == [("b", "c")] and g.why[("b", "c")]["rule"] == "R1"
    g = chain([("a", "c")], nodes="abc")
    g.directed |= {("a", "b"), ("b", "c")}
    g.meek()
    assert ("a", "c") in g.directed and g.why[("a", "c")]["rule"] == "R2"
    g = chain([("a", "b"), ("a", "c"), ("a", "d")])
    g.directed |= {("c", "b"), ("d", "b")}
    g.meek()
    assert ("a", "b") in g.directed and g.why[("a", "b")]["rule"] == "R3"


def test_extension_and_pins():
    g = chain([("a", "b"), ("b", "c")])
    dag = g.extension()
    assert len(dag) == 2
    model, report = apply_pins(g, [{"src": "a", "dst": "b", "support": 2}, {"src": "c", "dst": "b", "support": 1}], [])
    assert report[0]["status"] == "applied" and report[0]["implied"] == [["b", "c"]]
    assert report[1]["status"] == "contradicted by a stronger pin"
    assert model.class_size()["count"] == 1
    # a pin that would make a new collider is refused
    g = chain([("a", "b"), ("b", "c")])
    g2, rep = apply_pins(g, [{"src": "a", "dst": "b", "support": 1}, {"src": "c", "dst": "b", "support": 0}], [])
    assert rep[1]["status"].startswith("contradicted")
    tiers = tiers_to_pins([["a"], ["b", "c"]], chain([("a", "b"), ("b", "c")]))
    assert [(p["src"], p["dst"]) for p in tiers] == [("a", "b")]


def test_ida_ranges_over_parent_sets():
    # x - z, z -> y: either x causes z or not; the effect of x on y is a set
    data = {"x": [], "z": [], "y": []}
    import random
    rng = random.Random(3)
    for _ in range(3000):
        x = rng.choice([0, 1])
        z = x if rng.random() < 0.8 else 1 - x
        y = z if rng.random() < 0.8 else 1 - z
        data["x"].append(x), data["z"].append(z), data["y"].append(y)
    g = chain([("x", "z")], nodes="xyz")
    g.directed.add(("z", "y"))
    out = possible_effects(data, g, "x", "y")
    assert out["n_parent_sets"] == 2 and out["sign"] in ("+", "mixed")
    vals = sorted(m["effect"] for m in out["members"])
    assert vals[0] < 0.05 and vals[1] > 0.25  # adjusting for z blocks it; not adjusting shows it


def test_knowledge_base_inference_and_checks():
    nodes = {"self": {"kind": "agent"}, "justice": {"kind": "disposition"}, "virtue": {"kind": "class"},
             "good": {"kind": "class"}, "bad": {"kind": "class"}, "pleasure": {"kind": "state"},
             "anger": {"kind": "emotion"}, "vice": {"kind": "class"}}
    s = lambda a, r, b, neg=False, att="self", mod="asserts": {  # noqa: E731
        "s": a, "r": r, "o": b, "neg": neg, "passage": f"{a}{r}{b}", "locator": "L", "quote": "q",
        "modality": mod, "attribution": att}
    facts = aggregate([s("justice", "is_a", "virtue"), s("virtue", "is_a", "good"), s("self", "values", "good"),
                       s("anger", "is_a", "vice"), s("vice", "is_a", "bad"), s("anger", "is_a", "good", mod="asks"),
                       s("pleasure", "is_a", "good", att="Epicurus"), s("vice", "opposes", "virtue"),
                       s("justice", "learned_from", "self"), s("justice", "is_a", "good", neg=True)])
    kb = KnowledgeBase(nodes, facts)
    h = kb.holds("self", "values", "justice")
    assert h["status"] == "inferred" and h["proof"]["rule"].startswith("values passes")
    assert kb.holds("justice", "opposes", "anger")["status"] == "inferred"  # inherited both ways along is_a
    assert kb.holds("pleasure", "is_a", "good")["status"] == "not found"  # Epicurus's view, not the subject's
    assert kb.holds("anger", "is_a", "good")["status"] == "not found"  # a question is not an assertion
    types = {v["type"] for v in kb.violations}
    assert "domain_range" in types  # learned_from needs agents
    assert "denied_but_follows" in types
    assert kb.stated_support(("self", "values", "justice")) == {"selfvaluesgood", "justiceis_avirtue", "virtueis_agood"}


def test_selection_reasons():
    n = 100
    matrix = {"anger": [1] * 20 + [0] * 80, "wrath": [1] * 20 + [0] * 80, "rare": [1] * 3 + [0] * 97,
              "always": [1] * 90 + [0] * 10, "rome": [1] * 30 + [0] * 70}
    concepts = {k: {"label": k, "kind": "emotion", "variable": True} for k in matrix}
    concepts["rome"] = {"label": "Rome", "kind": "place", "variable": False}
    rows = {r["id"]: r for r in select(matrix, concepts, {"anger": 2})}
    assert rows["anger"]["status"] == "selected"
    assert rows["wrath"]["status"] == "excluded" and rows["wrath"]["duplicate_of"] == "anger"
    assert "too rare" in rows["rare"]["reason"]
    assert "barely varies" in rows["always"]["reason"]
    assert "not a quantity" in rows["rome"]["reason"]


def test_codebook_folds_opposites():
    codings = {"p1": {"concepts": [{"label": "Anger", "kind": "emotion", "gloss": "", "level": 1, "quote": ""},
                                   {"label": "freedom from anger", "kind": "state", "gloss": "", "level": 1, "quote": ""}],
                      "statements": [], "causal_claims": []}}

    def consolidate(lines, subject):
        return [{"id": "anger", "label": "anger", "kind": "emotion", "gloss": "g", "variable": True,
                 "members": [{"label": "anger", "polarity": 1}, {"label": "freedom from anger", "polarity": -1}]}]

    book = cb.build(codings, "M", consolidate)
    assert cb.resolve(book, "Freedom from anger") == ("anger", -1)
    assert cb.resolve(book, "the angers") == ("anger", 1)
    assert cb.norm("reasons") == "reason" and cb.norm("treason") == "treason"


def test_splitters(tmp_path):
    (tmp_path / "20080101-statement.txt").write_text("# Statement, 2008-01-01\ngroup: 2008\nRates fell.\n\nGrowth slowed.")
    (tmp_path / "20090101-statement.txt").write_text("Plain text.")
    ps = split_documents(tmp_path)
    assert [(p.id, p.locator, p.group) for p in ps] == [("20080101-statement", "Statement, 2008-01-01", "2008"),
                                                        ("20090101-statement", "20090101-statement", "2009")]
    ps = split_documents(tmp_path, unit="paragraph", min_chars=1)
    assert len(ps) == 3 and ps[1].locator == "Statement, 2008-01-01, ¶ 2"
    assert len(split_paragraphs("a\n\nb\n\nc", min_chars=1)) == 3


HTML_INDEX = '''<a href="/newsevents/pressreleases/monetary20221214a.htm">Statement</a>
<a href="/monetarypolicy/fomcminutes20221214.htm">Minutes</a>
<a href="/boarddocs/press/general/1999/19990630/default.htm">Statement</a>'''
HTML_STATEMENT = '''<html><head><script>var x = 1;</script></head><body><nav>Home Menu Item Link Item Link Item Link</nav>
<div id="article"><p>For release at 2:00 p.m. EST December 14, 2022</p>
<p>Recent indicators point to modest growth in spending and production. Job gains have been robust in recent months.</p>
<p>The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.</p>
<p>Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice Chair.</p>
</div><footer>Last Update: December 14, 2022 and a lot of footer words here</footer></body></html>'''


def test_fomc_parsing(tmp_path):
    found = fomc.links(HTML_INDEX, ("statement", "minutes"))
    assert found["20221214"] == {"statement": fomc.BASE + "/newsevents/pressreleases/monetary20221214a.htm",
                                 "minutes": fomc.BASE + "/monetarypolicy/fomcminutes20221214.htm"}
    assert "19990630" in found
    paras = fomc.page_text(HTML_STATEMENT)
    assert len(paras) == 2 and paras[0].startswith("Recent indicators")

    pages = {fomc.BASE + "/monetarypolicy/fomccalendars.htm": HTML_INDEX}

    def get(url):
        if url in pages:
            return pages[url]
        if "monetary20221214a" in url:
            return HTML_STATEMENT
        raise OSError("404")

    n = fomc.download(tmp_path, start=2022, end=2022, delay=0, get=get, log=lambda *_: None)
    assert n == 1
    doc = split_documents(tmp_path)[0]
    assert doc.locator == "Statement, 2022-12-14" and doc.group == "2022" and "maximum employment" in doc.text
    assert math.isclose(len(doc.text.split("\n\n")), 2)


def test_meditations_splitter_ids():
    from androidmirror.mirror.corpus import CorpusSpec, split
    text = ("*** START OF THE PROJECT ***\nTHE FIRST BOOK\n\nI. From my grandfather Verus I learned good morals.\n\n"
            "II. From the reputation of my father, modesty.\n\nTHE SECOND BOOK\n\nXIV. Though thou shouldst live three thousand years.\n"
            "*** END OF THE PROJECT ***")
    ps = split(CorpusSpec(id="m", title="M", subject="M", splitter="meditations"), text)
    assert [(p.id, p.locator, p.group) for p in ps] == [("b01s001", "Book I, I", "Book I"), ("b01s002", "Book I, II", "Book I"),
                                                        ("b02s014", "Book II, XIV", "Book II")]


def test_cli_routes_p1_and_causal(tmp_path, capsys):
    from androidmirror.cli import main
    from androidmirror.mirror.build import build
    from androidmirror.mirror.corpus import CorpusSpec

    from .fake import FakeCoder, write_corpus
    write_corpus(tmp_path / "corpus" / "synthetic", 300)
    spec = tmp_path / "synthetic.json"
    spec.write_text('{"id": "synthetic", "title": "S", "subject": "T", "splitter": "documents"}')
    build(CorpusSpec.load(str(spec)), tmp_path, coder=FakeCoder(), settings={"bootstrap": 0}, log=lambda *_: None)
    main(["--data", str(tmp_path), "causal", "--corpus", str(spec), "--alpha", "0.01", "--bootstrap", "0"])
    assert "variables" in capsys.readouterr().out
    main(["p1", "--data", str(tmp_path), "pending"])  # the paused app still answers


def test_consolidation_survives_truncated_answers():
    from androidmirror.mirror import codebook as cb

    codings = {f"p{i}": {"concepts": [{"label": f"idea{i}", "kind": "belief", "gloss": "", "level": 1, "quote": ""}]}
               for i in range(40)}
    calls = []

    def flaky(lines, subject):
        calls.append(len(lines))
        if len(lines) > 10:
            raise RuntimeError("answer cut off")
        if any(l.startswith("idea3 ") for l in lines):
            raise RuntimeError("still bad")
        return [{"id": "x_" + l.split(" |")[0], "label": l.split(" |")[0], "kind": "belief", "gloss": "",
                 "variable": True, "members": [{"label": l.split(" |")[0], "polarity": 1}]} for l in lines]

    book = cb.build(codings, "someone", flaky)
    assert max(calls) == 40 and min(calls) <= cb.MIN_BATCH
    assert book["label_map"]["idea25"][0] == "x_idea25"   # consolidated after splitting
    assert book["label_map"]["idea3"][0] == "idea3"     # its batch kept failing: normal-form fallback
    assert len(book["label_map"]) == 40


def test_meditations_split_stops_before_translators_back_matter():
    from androidmirror.corpus import split_sections

    text = ("THE FIRST BOOK\n\nI. Of my grandfather Verus.\n\nTHE TWELFTH BOOK\n\nI. All those things.\n\n"
            "II. Go thy ways then well pleased.\n\n\nAPPENDIX\n\nI. A letter of Fronto.\n\nNOTES\n\nII. A note.\n")
    chunks = split_sections(text)
    assert [(c.book, c.section) for c in chunks] == [(1, "I"), (12, "I"), (12, "II")]
    assert "Fronto" not in chunks[-1].text
