"""The archetype matcher: recovery of every card, null calibration, stability, determinism, read-only."""

import random
from pathlib import Path

import networkx as nx
import pytest

from androidmirror import archetypes
from androidmirror.ingest import load_seed
from androidmirror.ontology import GraphSnapshot, Ontology
from androidmirror.steward import Steward

SEED = Path(__file__).resolve().parent.parent / "data" / "seed_book1.json"
SAMPLES = 60  # enough null shuffles for tests; the app uses the data file's setting

# A few people who only know each other: gives the type shuffle something to swap with.
BACKGROUND_NODES = {f"bg{i}": "person" for i in range(4)}
BACKGROUND_KG = [("bg0", "knows", "bg1"), ("bg1", "knows", "bg2"), ("bg2", "knows", "bg3")]


def snap(nodes, kg=(), dag=(), background=True):
    """nodes: {id: type}; kg: (src, rel, dst); dag: (src, dst, sign[, strength[, confidence]])."""
    nodes = {**(BACKGROUND_NODES if background else {}), **nodes}
    kg = [*(BACKGROUND_KG if background else []), *kg]
    k = nx.MultiDiGraph()
    k.add_nodes_from(nodes)
    for s, r, d in kg:
        k.add_edge(s, d, key=r, provenance=[{"quote": f"{s} {r} {d}"}])
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for s, d, sign, *rest in dag:
        strength, conf = (rest + [0.7, 0.8])[:2] if len(rest) < 2 else rest
        g.add_edge(s, d, sign=sign, strength=strength, confidence=conf)
    return GraphSnapshot(1, {n: {"label": n, "type": t} for n, t in nodes.items()}, k, g)


V, P, X, E, B, A, W = "virtue", "practice", "vice", "emotion", "belief", "person", "event"

# A plain rising chain (with one shortcut, so it isn't a tree), for fixtures whose pattern is about signs: without it the
# sign shuffle has too few edges to tell a planted pattern from chance.
CHAIN_NODES = {f"k{i}": "concept" for i in range(5)}
CHAIN_DAG = [(f"k{i}", f"k{i + 1}", 1) for i in range(4)] + [("k0", "k2", 1)]

# One small graph per card, built from that card's patterns and little else.
FIXTURES = {
    0: dict(nodes={"v1": V, "v2": V, "v3": "value", "v4": "value"},
            dag=[("v1", "v2", 1, 0.5, 0.3), ("v3", "v4", 1, 0.5, 0.3)]),
    1: dict(nodes={"p": A, "q": A, "a": P, "b": P, "v1": V, "v2": V, "v3": V, "v4": "value"},
            kg=[("p", "practices", "a"), ("q", "practices", "b")],
            dag=[("a", "v1", 1), ("a", "v2", 1), ("a", "v3", 1), ("b", "v2", 1), ("b", "v3", 1), ("b", "v4", 1)]),
    2: dict(nodes={"b1": B, "b2": B, "c": "concept", "p": A, "v": V},
            kg=[("b1", "part_of", "c"), ("b2", "part_of", "c"), ("p", "believes_in", "b1"),
                ("b1", "about", "v"), ("b2", "about", "v"), ("c", "about", "v")]),
    3: dict(nodes={"t": A, "s": A, "s2": A, "v1": V, "v2": V, "v3": "value", "v4": V},
            kg=[("t", "taught", "s"), ("s", "taught", "s2")],
            dag=[("v1", "v2", 1), ("v2", "v3", 1), ("v4", "v2", 1)]),
    4: dict(nodes={"r": "concept", "c1": "concept", "c2": "concept", "c3": "concept", "x": X, "y": E, "z": X},
            kg=[("c2", "part_of", "c1"), ("c1", "part_of", "r"), ("c3", "part_of", "c1")],
            dag=[("r", "x", 1), ("r", "y", 1), ("y", "z", 1), ("c3", "c1", 1), ("c3", "c2", 1), ("c1", "r", 1)]),
    5: dict(nodes={"m": A, "t1": A, "t2": A, "v1": V, "v2": V, "v3": V},
            kg=[("m", "learned_from", "t1"), ("m", "learned_from", "t2"),
                ("t1", "exemplifies", "v1"), ("t2", "exemplifies", "v2")],
            dag=[("v1", "v3", 1), ("v2", "v3", -1)]),
    6: dict(nodes={"v1": "value", "v2": "value", "v3": V, "v4": V, "t": "concept", "u": "concept"},
            kg=[("v1", "opposes", "v2"), ("v3", "opposes", "v4")],
            dag=[("v1", "t", 1), ("v2", "t", 1), ("v3", "u", 1), ("v4", "u", 1)]),
    7: dict(nodes={"p": A, "q": A, "v": V, "x": X, "a": "concept", "b": "concept", "c": "concept", "d": "concept",
                   "e": "concept", "f": "concept"},
            kg=[("p", "values", "v"), ("p", "rejects", "x"), ("q", "values", "v"), ("q", "rejects", "x")],
            dag=[("a", "c", 1, 0.9), ("b", "c", -1, 0.3), ("c", "d", 1), ("d", "v", 1),
                 ("a", "e", 1, 0.9), ("b", "e", -1, 0.3), ("e", "f", 1), ("f", "v", 1)]),
    8: dict(nodes={"v1": V, "v2": V, "x1": E, "x2": X},
            kg=[("v1", "opposes", "x1"), ("v2", "opposes", "x2")],
            dag=[("v1", "x1", -1), ("v2", "x2", -1)]),
    9: dict(nodes={"p": A, "b1": B, "b2": B, "c": "concept", "a": P, "v": V},
            kg=[("p", "believes_in", "b1"), ("p", "holds", "b2")],
            dag=[("b1", "v", 1), ("b2", "v", 1), ("a", "c", 1)]),
    10: dict(nodes={"w1": W, "w2": W, "w3": "place", "v": V, "x": E},
             kg=[("w1", "happened_in", "w3"), ("w2", "happened_in", "w3")],
             dag=[("w1", "v", 1), ("w2", "x", 1), ("w3", "x", 1)]),
    11: dict(nodes={**CHAIN_NODES, "j": "concept", "k": "concept", "v1": V, "x1": X, "c": "concept", "a": "concept",
                    "b": "concept", "d": "concept", "d2": "concept"},
             kg=[("j", "opposes", "v1"), ("j", "opposes", "x1"), ("k", "opposes", "v1"), ("k", "opposes", "x1")],
             dag=[*CHAIN_DAG, ("a", "c", 1, 0.5), ("b", "c", -1, 0.5), ("a", "d", 1, 0.6), ("b", "d", -1, 0.6),
                  ("j", "d2", 1, 0.4), ("k", "d2", -1, 0.4)]),
    12: dict(nodes={**CHAIN_NODES, "a1": P, "a2": P, "x1": E, "x2": X, "c": "concept", "d": "concept"},
             kg=[("a1", "opposes", "x1"), ("a2", "resists", "x2")],
             dag=[*CHAIN_DAG, ("a1", "x1", -1), ("x1", "c", -1), ("a2", "x2", -1), ("x2", "d", -1)]),
    13: dict(nodes={"a": "concept", "b": "concept", "e": "concept", "f": "concept", "c": "concept", "d": "concept"},
             kg=[("e", "about", "c"), ("f", "about", "d")],
             dag=[("a", "e", -1), ("b", "e", -1), ("a", "f", -1), ("b", "f", -1)]),
    14: dict(nodes={**CHAIN_NODES, "v1": V, "v2": V, "v3": V, "a": "concept", "b": "concept", "c": "concept", "t": "concept",
                    "u": "concept"},
             kg=[("v1", "part_of", "v2"), ("v3", "part_of", "v2")],
             dag=[*CHAIN_DAG, ("a", "t", 1, 0.4), ("b", "t", -1, 0.4), ("c", "t", 1, 0.4),
                  ("a", "u", -1, 0.3), ("b", "u", 1, 0.3), ("c", "u", 1, 0.3)]),
    15: dict(nodes={"p": A, "x1": X, "x2": E, "x3": X, "x4": E},
             kg=[("p", "values", "x1")],
             dag=[("x1", "x2", 1), ("x2", "x3", 1), ("x4", "x3", 1)]),
    16: dict(nodes={"w": W, "h": "concept", "v1": V, "c1": "concept", "c2": "concept", "c3": "concept",
                    "c4": "concept", "c5": "concept"},
             kg=[("h", "part_of", "c1"), ("c2", "part_of", "c1"), ("c3", "part_of", "c1")],
             dag=[("w", "v1", -1), ("v1", "h", 1), ("h", "c1", 1), ("h", "c2", 1), ("h", "c3", 1),
                  ("c1", "c4", 1), ("c2", "c4", 1), ("c3", "c5", 1)]),
    17: dict(nodes={"p": A, "a": P, "v1": V, "v2": V, "goal": "value", "q": A},
             kg=[("p", "values", "goal"), ("q", "strives_for", "goal")],
             dag=[("a", "v1", 1), ("v1", "v2", 1), ("v2", "goal", 1)]),
    18: dict(nodes={"b": B, "e": E, "a": "concept", "m": "concept", "z": "concept", "b2": B, "e2": E},
             kg=[("b", "feeds", "e"), ("b2", "about", "e2")],
             dag=[("a", "m", 1, 0.5, 0.4), ("m", "z", -1, 0.5, 0.4), ("a", "z", 1, 0.5, 0.4)]),
    19: dict(nodes={"a": "concept", "b": "concept", "c": "concept", "d": "concept"},
             dag=[("a", "b", 1, 0.7, 0.9), ("b", "c", 1, 0.7, 0.9), ("c", "d", 1, 0.7, 0.9)],
             background=False),
    20: dict(nodes={"b": B, "b2": B, "x": X, "x2": E, "v": V, "v2": V},
             kg=[("b", "opposes", "x"), ("b2", "rejects", "x2")],
             dag=[("b", "x", -1), ("b", "v", 1), ("b2", "x2", -1), ("b2", "v2", 1)]),
    21: dict(nodes={"r1": "concept", "r2": "concept", "m": "concept", "s1": "concept", "s2": "concept"},
             kg=[("r1", "about", "r2"), ("m", "about", "s1"), ("s2", "about", "r1")],
             dag=[("r1", "m", 1, 0.5, 0.5), ("r2", "m", -1, 0.5, 0.5), ("m", "s1", 1, 0.5, 0.5), ("m", "s2", -1, 0.5, 0.5)],
             background=False),
}


def run(s, **kw):
    return archetypes.match(s, samples=SAMPLES, **kw)


def test_every_card_has_a_fixture_and_the_set_is_well_formed():
    cards = archetypes.load_set()["cards"]
    assert [c["n"] for c in cards] == list(range(22))
    assert set(FIXTURES) == set(range(22))
    for c in cards:
        assert c["motifs"] and all(m["text"] and m["layer"] in ("descriptive", "causal") for m in c["motifs"])


@pytest.mark.parametrize("n", sorted(FIXTURES))
def test_recovers_each_card(n):
    r = run(snap(**FIXTURES[n]))
    assert r["archetype"]["n"] == n, (r["archetype"]["name"], [(x["name"], x["score"]) for x in r["runners_up"]])
    assert not r["undecided"], (r["archetype"]["score"], r["margin"])


def test_reversed_reading():
    # Strength reversed: the same virtue-passion pairs, but each virtue inflames its passion.
    s = snap({**CHAIN_NODES, "v1": V, "v2": V, "x1": E, "x2": X},
             kg=[("v1", "opposes", "x1"), ("v2", "opposes", "x2")],
             dag=[*CHAIN_DAG, ("v1", "x1", 1), ("v2", "x2", 1)])
    r = run(s)
    assert r["archetype"]["n"] == 8 and r["archetype"]["reversed"]


def test_random_graphs_are_mostly_undecided():
    """Null calibration: on graphs with random types and signs, few should claim a card."""
    types = ["person", "virtue", "vice", "emotion", "belief", "practice", "value", "concept", "event", "place"]
    rels = ["learned_from", "exemplifies", "opposes", "values", "part_of", "related_to"]
    rng = random.Random(1)
    decided = 0
    for _ in range(20):
        ids = [f"n{i}" for i in range(14)]
        nodes = {i: rng.choice(types) for i in ids}
        kg = [(rng.choice(ids), rng.choice(rels), rng.choice(ids)) for _ in range(12)]
        kg = [(s, r, d) for s, r, d in kg if s != d]
        order = ids[:]
        rng.shuffle(order)
        dag = {(order[a], order[b]) for a, b in (sorted(rng.sample(range(14), 2)) for _ in range(12))}
        r = run(snap(nodes, kg, [(s, d, rng.choice([1, -1]), rng.random(), rng.random()) for s, d in dag], background=False))
        decided += not r["undecided"]
    assert decided <= 5, decided


def _seed_snapshots(tmp_path):
    st = Steward(Ontology(), tmp_path)
    queued = load_seed(st, SEED)
    for p in queued:
        st.approve(p.id, "test")
    return st, queued


def test_book_one_seed(tmp_path):
    st, _ = _seed_snapshots(tmp_path)
    r = run(st.ontology.graph_snapshot())
    assert r["archetype"]["name"] == "The Hierophant"
    lineage = r["evidence"][0]
    assert lineage["examples"][0]["quotes"], "evidence must carry the source quotes"


def test_stable_when_one_seed_patch_is_left_out(tmp_path):
    st, queued = _seed_snapshots(tmp_path)
    full = run(st.ontology.graph_snapshot())
    flips = 0
    for i in range(len(queued)):
        st2 = Steward(Ontology(), tmp_path / str(i))
        kept = [p for j, p in enumerate(load_seed(st2, SEED)) if j != i]
        for p in kept:
            st2.approve(p.id, "test")
        flips += run(st2.ontology.graph_snapshot())["archetype"]["n"] != full["archetype"]["n"]
    assert flips <= 2, flips


def test_deterministic_and_read_only(tmp_path):
    st, _ = _seed_snapshots(tmp_path)
    before = st.ontology.snapshot()
    a = run(st.ontology.graph_snapshot())
    b = run(st.ontology.graph_snapshot())
    assert a == b
    assert st.ontology.snapshot() == before


def test_focus_and_unclassified_relations():
    s = snap(**FIXTURES[8])
    r = run(s, focus="v1")
    assert r["focus"] == "v1"
    assert "knows" not in r["unclassified_relations"]  # background is outside v1's neighbourhood
    assert "knows" in run(s)["unclassified_relations"]
    with pytest.raises(KeyError):
        run(s, focus="nope")


def test_tiny_graph_is_undecided():
    r = run(snap({"a": V}, background=False))
    assert r["undecided"] and r["archetype"] is None


def test_mirror_relations_are_all_read():
    from androidmirror.mirror.schema import RELATIONS
    family_of = archetypes.load_set()["_family_of"]
    assert [r for r in RELATIONS if r not in family_of] == []
    assert all(family_of[r] == spec["family"] for r, spec in RELATIONS.items())
