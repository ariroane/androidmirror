import itertools
import random

import networkx as nx

from androidmirror.mirror.causal import fit
from androidmirror.mirror.causal.pag import pag, to_mag

from .test_units import J


def model(directed, bidirected=(), nodes=()):
    names = sorted(set(nodes) | {n for e in directed for n in e} | {n for p in bidirected for n in p})
    return {"nodes": names, "edges": [{"src": a, "dst": b} for a, b in directed],
            "common_causes": [{"a": min(p), "b": max(p), "via": ""} for p in bidirected], "constraints": []}


def marks(result):
    return {(e["a"], e["b"]): tuple(e["marks"]) for e in result["edges"]}


def test_a_two_way_link_beside_a_chain_becomes_an_arrow_and_its_ties_are_flagged_not_added():
    # p -> a -> b -> c, and a <-> c (the subject's c -> a would have made a circle)
    m = model([("p", "a"), ("a", "b"), ("b", "c")], [("a", "c")])
    mag = to_mag(m)
    assert mag["converted"] == [{"src": "a", "dst": "c", "chain": ["a", "b", "c"], "also_arrow": False}]
    assert mag["marks"][("c", "a")] == "-" and mag["marks"][("a", "c")] == ">"
    # holding b fixed blocks p's chain to c, but the hidden cause of a and c still ties them: p - c can't be pulled apart
    assert mag["inseparable"] == [{"a": "c", "b": "p", "marks": ["arrow", "tail"], "path": ["c", "a", "p"]}]
    assert ("p", "c") not in mag["marks"] and not mag["maximal"]
    off, on = pag(m), pag(m, inseparable=True)
    assert not off["valid"] and on["valid"] and ("c", "p") not in marks(off)
    assert marks(on)[("c", "p")] == ("arrow", "tail") and next(e for e in on["edges"] if e["a"] == "c" and e["b"] == "p")["inseparable"]


def test_without_background_only_what_every_graph_shares_keeps_its_marks():
    m = model([("a", "c"), ("b", "c"), ("c", "d")])
    free = marks(pag(m, background=False))
    # a -> c <- b is a collider every equivalent graph has; c -> d follows from it; a and b's own ends stay open
    assert free[("a", "c")] == ("circle", "arrow") and free[("b", "c")] == ("circle", "arrow")
    assert free[("c", "d")] == ("tail", "arrow")
    # the subject's directions fill in the tails
    assert marks(pag(m))[("a", "c")] == ("tail", "arrow")
    # a lone arrow says nothing about its direction by itself
    assert marks(pag(model([("a", "b")]), background=False))[("a", "b")] == ("circle", "circle")


def test_a_held_no_difference_claim_is_background_knowledge():
    m = model([], [("a", "b")])
    assert marks(pag(m, background=False))[("a", "b")] == ("circle", "circle")
    m["constraints"] = [{"type": "no_path", "x": "a", "y": "b", "status": "holds"}]
    assert marks(pag(m))[("a", "b")] == ("arrow", "circle")  # a does not lead to b: an arrowhead at a


def test_works_on_a_fitted_model():
    js = [J("causes", "a", "b", scope="direct", p=f"p{i}") for i in range(2)] + [
        J("causes", "b", "c", scope="direct", p=f"q{i}") for i in range(2)] + [J("causes", "c", "a", scope="direct", p="r")]
    r = pag(fit(js))
    assert [(x["src"], x["dst"]) for x in r["converted"]] == [("a", "c")] and r["valid"]
    assert r["stats"]["leads_to"] == 3 and r["stats"]["background_marks"] == 3


def _seps(nodes, directed, bidirected):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(directed)
    for i, p in enumerate(bidirected):
        for n in p:
            g.add_edge(f"L{i}", n)
    out = {}
    for a, b in itertools.combinations(nodes, 2):
        rest = [n for n in nodes if n not in (a, b)]
        out[(a, b)] = {frozenset(z) for r in range(len(rest) + 1) for z in itertools.combinations(rest, r)
                       if nx.is_d_separator(g, {a}, {b}, set(z))}
    return out


def test_against_brute_force_on_small_random_graphs():
    """Links match what no set separates, and every mark the PAG keeps is shared by all equivalent graphs."""
    rng = random.Random(7)
    checked = 0
    while checked < 12:
        nodes = list("abcd")
        order = rng.sample(nodes, 4)
        directed = {(order[i], order[j]) for i in range(4) for j in range(i + 1, 4) if rng.random() < 0.4}
        bidirected = {frozenset(p) for p in itertools.combinations(nodes, 2) if rng.random() < 0.3}
        m = model(directed, [tuple(p) for p in bidirected], nodes)
        truth = _seps(nodes, directed, bidirected)
        mag = to_mag(m, inseparable=True)
        linked = {frozenset(k) for k in mag["marks"]}
        assert all((frozenset(k) in linked) == (not s) for k, s in truth.items())
        edges = sorted({tuple(sorted(k)) for k in linked})
        if len(edges) > 6:
            continue
        checked += 1
        members = []
        for pick in itertools.product(range(3), repeat=len(edges)):
            d = {e if k == 0 else e[::-1] for e, k in zip(edges, pick) if k < 2}
            bi = {frozenset(e) for e, k in zip(edges, pick) if k == 2}
            g = nx.DiGraph(d)
            g.add_nodes_from(nodes)
            if not nx.is_directed_acyclic_graph(g) or any(a in nx.ancestors(g, b) or b in nx.ancestors(g, a)
                                                          for a, b in map(tuple, bi)):
                continue
            if _seps(nodes, d, bi) == truth:
                mk = {(a, b): "arrow" for a, b in d} | {(b, a): "tail" for a, b in d}
                mk |= {(a, b): "arrow" for p in bi for a, b in itertools.permutations(p)}
                members.append(mk)
        for e in pag(m, background=False, inseparable=True)["edges"]:
            for end, mark in zip(((e["b"], e["a"]), (e["a"], e["b"])), e["marks"]):
                seen = {mk[end] for mk in members}
                assert (mark == "circle") == (len(seen) > 1) and (mark == "circle" or seen == {mark})
