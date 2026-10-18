import json
import math

import pytest

from androidmirror.mirror import codebook as cb
from androidmirror.mirror import judgments as jd
from androidmirror.mirror.causal import explain, fit, holdout
from androidmirror.mirror.corpus import split_documents, split_paragraphs
from androidmirror.mirror.kg import KnowledgeBase, aggregate
from androidmirror.mirror.sources import dgn, fomc


def J(kind, x, y, sign=1, z="", p="p1", scope="unsaid", held=True):
    return {"kind": kind, "x": x, "y": y, "z": z, "sign": sign, "passage": p, "locator": p, "quote": "q",
            "register": "describes", "attribution": "self" if held else "someone", "scope": scope, "held": held}


def status(model):
    return {c["id"]: c["status"] for c in model["constraints"]}


def arrows(model):
    return {(e["src"], e["dst"], e["sign"]) for e in model["edges"]}


def test_a_claim_counts_once_per_passage():
    cs = jd.constraints([J("causes", "a", "b", p="p1"), J("causes", "a", "b", p="p1"), J("causes", "a", "b", p="p2"),
                         J("causes", "a", "b", p="p3", held=False)])
    c = cs[0]
    assert c["weight"] == 2 and c["mentions"] == 3 and len(c["evidence"]) == 4


def test_overall_claims_wait_for_a_path_and_add_no_shortcut():
    # "a, in the end, brings c" is said most often, but a -> b -> c explains it
    js = [J("causes", "a", "c", p=f"x{i}") for i in range(5)] + [J("causes", "a", "b", scope="direct", p="y"),
                                                                 J("causes", "b", "c", scope="direct", p="z")]
    m = fit(js)
    assert arrows(m) == {("a", "b", 1), ("b", "c", 1)}
    assert status(m)["reaches|a|c"] == "follows"
    assert next(c for c in m["constraints"] if c["id"] == "reaches|a|c")["path"] == ["a", "b", "c"]


def test_an_overall_claim_with_no_path_becomes_an_arrow():
    m = fit([J("causes", "a", "c", p="x")])
    assert arrows(m) == {("a", "c", 1)}


def test_cycles_are_rejected_and_reported():
    js = [J("causes", "a", "b", scope="direct", p=f"p{i}") for i in range(3)] + [J("causes", "b", "a", scope="direct", p="q")]
    m = fit(js)
    assert arrows(m) == {("a", "b", 1)}
    c = next(c for c in m["constraints"] if c["id"] == "reaches|b|a")
    assert c["status"] == "rejected" and c["reason"] == "cycle" and c["conflicts"] == ["reaches|a|b"]
    assert m["feedback"] == [{"pair": ["a", "b"], "status": ["arrow", "rejected"]}]


def test_a_stronger_denial_blocks_a_chain_and_a_stronger_claim_overrules_a_denial():
    # "pain does not touch the soul" (3 passages) against pain -> distress -> soul (1 each)
    js = [J("independent", "pain", "soul", sign=0, p=f"d{i}") for i in range(3)] + [
        J("causes", "pain", "distress", scope="direct", p="c1"), J("causes", "distress", "soul", -1, scope="direct", p="c2")]
    m = fit(js)
    st = status(m)
    assert st["no_path|pain|soul"] == "holds"
    assert sorted([st["reaches|pain|distress"], st["reaches|distress|soul"]]) == ["arrow", "rejected"]
    # said more often, the claim wins and the denial is reported as outweighed
    js2 = [J("causes", "pain", "soul", -1, p=f"c{i}") for i in range(4)] + js[:3]
    m2 = fit(js2)
    assert status(m2)["reaches|pain|soul"] == "arrow" and status(m2)["no_path|pain|soul"] == "rejected"
    assert explain(m, "pain", "soul")["verdict"].startswith("makes no difference")


def test_not_x_but_y():
    # "it is not things but our opinions of them that disturb us"
    js = [J("screens_off", "things", "opinion", z="disturbance", p="s"), J("causes", "things", "opinion", p="t"),
          J("causes", "things", "disturbance", scope="direct", p="u")]
    m = fit(js)
    st = status(m)
    assert st["only_through|things|opinion|disturbance"] == "holds"
    assert st["reaches|opinion|disturbance"] == "arrow" and st["reaches|things|opinion"] == "arrow"
    assert st["reaches|things|disturbance"] in ("follows", "rejected")
    assert ("things", "disturbance", 1) not in arrows(m)
    assert explain(m, "things", "disturbance")["verdict"] == "raises"


def test_thought_experiments_and_signs():
    js = [J("intervention", "opinion", "complaint", 1, p="a"), J("intervention", "x", "y", 0, p="b"),
          J("causes", "a", "b", -1, scope="direct", p="c"), J("causes", "b", "d", -1, scope="direct", p="d"),
          J("causes", "a", "d", 1, scope="direct", p="e")]
    m = fit(js)
    st = status(m)
    assert st["reaches|opinion|complaint"] == "arrow" and st["no_path|x|y"] == "holds"
    assert ("a", "d", 1) in arrows(m)  # beside a path of the other sign (a lowers b lowers d)
    assert explain(m, "a", "d")["verdict"].startswith("raises")
    assert explain(m, "d", "a")["verdict"].startswith("not said")


def test_reading_resolves_ids_labels_polarity_and_voices():
    book = {"concepts": {"anger": {}, "distress": {}, "self": {}, "good": {}},
            "label_map": {"freedom from anger": ["anger", -1], "distress": ["distress", 1]}}
    reading = {"judgments": [
        {"kind": "causes", "x": "freedom from anger", "y": "distress", "z": "", "sign": 1, "scope": "direct",
         "register": "describes", "attribution": "Marcus", "quote": "q"},
        {"kind": "causes", "x": "anger", "y": "distress", "z": "", "sign": 1, "scope": "direct",
         "register": "describes", "attribution": "Epicurus", "quote": "q"},
        {"kind": "causes", "x": "anger", "y": "unicorns", "z": "", "sign": 1, "scope": "direct",
         "register": "entertains", "attribution": "self", "quote": "q"},
        {"kind": "independent", "x": "anger", "y": "good", "z": "", "sign": 0, "scope": "unsaid",
         "register": "describes", "attribution": "self", "quote": "q"}],
        "values": [{"concept": "freedom from anger", "direction": 1, "quote": "q"}]}
    r = jd.read({"id": "p", "locator": "L"}, reading, book, "Marcus")
    first, second = r["judgments"]
    assert (first["x"], first["sign"], first["held"]) == ("anger", -1, True)  # polarity folded, the subject's own view
    assert second["held"] is False  # Epicurus's view is kept, not held
    assert len(r["unresolved"]) == 2  # unicorns are not in the codebook; "good" is an evaluation, not a cause
    assert r["values"] == [{"concept": "anger", "direction": -1, "passage": "p", "locator": "L", "quote": "q"}]
    v = jd.values(r["values"] + [{**r["values"][0], "passage": "p2"}])
    assert v[0]["less"] == 2 and v[0]["direction"] == -1


def test_holdout_counts_restated_follows_and_contradicted():
    js = [J("causes", "a", "b", scope="direct", p="p0"), J("causes", "b", "c", scope="direct", p="p1"),
          J("causes", "a", "c", p="p2"), J("causes", "a", "b", scope="direct", p="p3"), J("causes", "c", "a", p="p4")]
    h = holdout(js, {"folds": 5})
    assert h["folds"] == 5
    assert h["positive"]["follows"] >= 1  # a -> c follows from a -> b -> c without being said
    assert h["positive"]["restated"] >= 2
    assert h["positive"]["contradicted"] >= 1  # c -> a runs against the rest


def test_offline_benchmark_recovers_a_planted_subject():
    from androidmirror.mirror import benchmark

    r = benchmark.run_offline(n=300, seeds=2, noises=(0.0, 0.1))
    clean = r["results"]["reading errors 0%"]
    assert clean["fit"]["true_arrows"] == 1.0 and clean["fit"]["recall_of_effects"] >= 0.95
    assert clean["every claim an arrow"]["true_arrows"] < 0.6
    noisy = r["results"]["reading errors 10%"]
    assert noisy["fit"]["true_arrows"] > noisy["every claim an arrow"]["true_arrows"] + 0.3
    assert noisy["fit"]["in_cycles"] == 0 and noisy["every claim an arrow"]["in_cycles"] > 0


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


def test_codebook_folds_opposites():
    codings = {"p1": {"concepts": [{"label": "Anger", "kind": "emotion", "gloss": "", "level": 1, "quote": ""},
                                   {"label": "freedom from anger", "kind": "state", "gloss": "", "level": 1, "quote": ""}],
                      "statements": []}}

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
    write_corpus(tmp_path / "corpus" / "synthetic")
    spec = tmp_path / "synthetic.json"
    spec.write_text('{"id": "synthetic", "title": "S", "subject": "T", "splitter": "documents"}')
    build(CorpusSpec.load(str(spec)), tmp_path, coder=FakeCoder(), log=lambda *_: None)
    main(["--data", str(tmp_path), "causal", "--corpus", str(spec), "--min-weight", "2"])
    assert "arrows" in capsys.readouterr().out
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


def test_benchmark_with_a_written_corpus(tmp_path):
    from androidmirror.mirror import benchmark

    from .fake import FakeCoder

    def writer(said):  # stands in for Claude writing the passage: the judgments as tags the fake coder reads
        tags = [f"[v {v} 1 state]" for v in benchmark.VARIABLES]
        for j in said:
            tags.append(f"[j {j['kind']} {j['x']} {j['y']} {j['z'] or '-'} {j['sign']} {j['scope']} describes "
                        f"{'self' if j['held'] else 'someone'}]")
        return " ".join(tags)

    r = benchmark.run_with_claude(tmp_path, n=120, writer=writer, coder=FakeCoder(), log=lambda *_: None)
    assert r["passages"] == 120 and r["reading_precision"] == 1.0 and r["reading_recall"] == 1.0
    assert r["fit"] == r["fit_on_perfect_reading"]


def link(model, a, b):
    return next((x for x in model["common_causes"] if {x["a"], x["b"]} == {a, b}), None)


def test_claims_that_clash_become_a_shared_cause_with_no_directed_path():
    # a loop: a -> b -> c is said more often than c -> a
    js = [J("causes", "a", "b", scope="direct", p=f"p{i}") for i in range(2)] + [
        J("causes", "b", "c", scope="direct", p=f"q{i}") for i in range(2)] + [J("causes", "c", "a", scope="direct", p="r")]
    m = fit(js)
    x = link(m, "a", "c")
    assert x["claims"] == ["reaches|c|a"] and x["reasons"] == ["cycle"] and not x["via"] and not x["settled"]
    assert x["conflicts"] == ["reaches|a|b", "reaches|b|c"]
    assert next(c for c in m["constraints"] if c["id"] == "reaches|c|a")["common_cause"] == "a<->c"
    assert arrows(m) == {("a", "b", 1), ("b", "c", 1)} and m["stats"]["common_causes"] == 1
    # against a stronger "makes no difference", too; and the denial still holds
    js2 = [J("independent", "d", "e", sign=0, p=f"d{i}") for i in range(2)] + [J("causes", "d", "e", p="c")]
    m2 = fit(js2)
    assert link(m2, "d", "e")["reasons"] == ["denied"] and status(m2)["no_path|d|e"] == "holds"
    assert m2["nodes"] == ["d", "e"] and not m2["edges"]  # drawn though it has no arrow
    assert explain(m2, "d", "e")["verdict"].startswith("makes no difference")  # the subject's denial comes first
    # an outweighed denial is not a shared cause
    m3 = fit([J("causes", "d", "e", p=f"c{i}") for i in range(3)] + js2[:2])
    assert status(m3)["no_path|d|e"] == "rejected" and not m3["common_causes"]


def test_a_ruling_settles_a_shared_cause_and_can_name_it():
    base = [J("causes", "z", "a", scope="direct", p="s1"), J("causes", "z", "b", scope="direct", p="s2"),
            J("causes", "a", "b", scope="direct", p="p")]
    ruling = {**J("ruling", "a", "b", p="answer:1"), "target": "reaches|a|b", "ruling": "common_cause", "held": False,
              "direct": False, "indirect": False, "via": ""}
    m = fit(base + [ruling])
    x = link(m, "a", "b")
    assert ("a", "b", 1) not in arrows(m) and x["settled"] and not x["via"] and x["reasons"] == ["settled"]
    assert explain(m, "a", "b")["verdict"].startswith("makes no difference by itself")
    m2 = fit(base + [{**ruling, "via": "z"}])
    assert link(m2, "a", "b")["via"] == "z" and m2["stats"]["common_causes"] == 0
    # a concept that reaches one end only through the other does not explain why they go together
    m3 = fit(base[1:] + [J("causes", "b", "a", scope="direct", p="s3"), {**ruling, "via": "z"}])
    assert link(m3, "a", "b")["named"] == "z" and not link(m3, "a", "b")["via"]


def test_a_claim_ruled_out_by_a_denial_about_other_ideas_records_the_chain_it_would_complete():
    # "thoughts make no difference to happiness" (3), thoughts -> acceptance and holiness -> happiness (2 each):
    # acceptance -> holiness (1) would run thoughts -> acceptance -> holiness -> happiness
    js = [J("independent", "thoughts", "happiness", sign=0, p=f"d{i}") for i in range(3)] + [
        J("causes", "thoughts", "acceptance", scope="direct", p=f"a{i}") for i in range(2)] + [
        J("causes", "holiness", "happiness", scope="direct", p=f"h{i}") for i in range(2)] + [
        J("causes", "acceptance", "holiness", p="x")]
    m = fit(js)
    c = next(c for c in m["constraints"] if c["id"] == "reaches|acceptance|holiness")
    assert c["reason"] == "denied" and c["conflicts"] == ["no_path|thoughts|happiness"]
    assert c["path"] == ["thoughts", "acceptance", "holiness", "happiness"]
    assert c["through"] == ["reaches|thoughts|acceptance", "reaches|holiness|happiness"]
    assert link(m, "acceptance", "holiness")


DGN_SEARCH = {"resultList": {"result": [
    {"pmcid": "PMC1", "doi": "10.1/a", "pubYear": "2024", "journalTitle": "J Neurol",
     "title": "Pharmacotherapy of motor symptoms in early and mid-stage Parkinson's disease: guideline "
              "“Parkinson's disease” of the German Society of Neurology"},
    {"pmcid": "PMC2", "title": "Guideline “Parkinson's disease” of the German Society of Neurology (Deutsche "
                               "Gesellschaft für Neurologie): concepts of care"},
    {"pmcid": "PMC3", "title": "Correction: Guideline “Parkinson's disease” of the German Society of Neurology"},
]}, "nextCursorMark": "*"}
DGN_XML = '''<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><permissions>
<license xlink:href="http://creativecommons.org/licenses/by/4.0/"><p>Open access.</p></license></permissions></front>
<body><sec><title>Introduction</title><p>Levodopa is the most effective drug.</p></sec>
<sec><title>Early therapy</title><p>Overview.</p>
<sec><title>Levodopa</title><p>Early monotherapy brings fluctuations sooner.<boxed-text><caption><title>Recommendation</title></caption>
<p>Amantadine should be used to reduce dyskinesias.</p></boxed-text> And so on.</p></sec></sec>
<sec><title>Declarations</title><p>None.</p></sec></body><back><ref-list><ref>Ref</ref></ref-list></back></article>'''


def test_dgn_parsing(tmp_path):
    assert dgn.short_name(DGN_SEARCH["resultList"]["result"][0]["title"]) == \
        "Parkinson's disease: pharmacotherapy of motor symptoms in early and mid-stage Parkinson's disease"
    assert dgn.short_name(DGN_SEARCH["resultList"]["result"][1]["title"]) == "Parkinson's disease: concepts of care"

    def get(url):
        if "/search?" in url:
            return json.dumps(DGN_SEARCH)
        if "PMC1/" in url:
            return DGN_XML
        if "PMC2/" in url:
            return DGN_XML.replace("licenses/by/", "licenses/by-nc-nd/")
        raise OSError("404")

    n = dgn.download(tmp_path, delay=0, get=get, log=lambda *_: None)
    docs = split_documents(tmp_path)
    assert n == 3 and len(docs) == 3
    assert [d.locator.split(" - ")[1] for d in docs] == ["Introduction", "Early therapy", "Early therapy / Levodopa"]
    assert docs[2].group.startswith("Parkinson's disease: pharmacotherapy")
    assert docs[2].text.split("\n\n") == ["Early monotherapy brings fluctuations sooner.", "Recommendation",
                                          "Amantadine should be used to reduce dyskinesias.", "And so on."]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["PMC1"]["licence"] == "CC BY" and manifest["PMC2"]["kept"] is False and "PMC3" not in manifest
