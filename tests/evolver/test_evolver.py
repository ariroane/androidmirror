from pathlib import Path

import pytest

from androidmirror import evolver
from androidmirror.evolver import operators as ops_mod
from androidmirror.ingest import load_seed
from androidmirror.ontology import Ontology
from androidmirror.patches import Op, Patch
from androidmirror.steward import GovernanceError, Steward

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "data" / "seed_book1.json"


@pytest.fixture
def steward(tmp_path):
    st = Steward(Ontology(), tmp_path)
    load_seed(st, SEED)
    for p in st.list("proposed"):
        st.approve(p.id, "Android K")
    return st


# Stand-ins for the refactor / break / chain engines, which plug in through the same interface.

def merge_morals(snap):
    """A bad chain: unions two lessons the probes say are distinct."""
    if "good_morals" not in snap.nodes or "control_of_temper" not in snap.nodes:
        return []
    edges = [Op(op="add_relation", src=s, rel=k, dst="good_morals") for s, d, k in snap.kg.in_edges("control_of_temper", keys=True)
             if not snap.kg.has_edge(s, "good_morals", key=k)]
    return [Patch(kind="chain", rationale="merge temper into morals",
                  ops=[*edges, Op(op="remove_node", id="control_of_temper"),
                       Op(op="update_node", id="good_morals", attrs={"aliases": ["Control of temper"]})])]


def link_discipline(snap):
    """A good chain: links self-discipline to control of temper causally."""
    if "self_discipline" not in snap.nodes or snap.dag.has_edge("self_discipline", "control_of_temper"):
        return []
    return [Patch(kind="chain", rationale="self-discipline strengthens control of temper",
                  ops=[Op(op="add_causal_edge", src="self_discipline", dst="control_of_temper", sign=1,
                          strength=0.7, confidence=0.6)])]


def vandal(snap):
    """An engine that scribbles on the snapshot it was given."""
    snap.nodes.clear()
    snap.kg.clear()
    snap.dag.clear()
    return []


def broken(snap):
    raise RuntimeError("boom")


DISCIPLINE = evolver.Probe(id="discipline_temper", kind="causes", a="Self-discipline", b="Control of temper",
                           claim="Discipline steadies his temper")


def test_evolve_never_touches_live_ontology(steward):
    onto = steward.ontology
    before, version = onto.snapshot(), onto.version
    result = evolver.evolve(onto, [DISCIPLINE], {"vandal": vandal, "chain": link_discipline, "x": broken})
    assert onto.snapshot() == before and onto.version == version
    assert result["errors"] == ["x: boom"]
    assert result["designs"]  # the vandal's scribbles did not leak into other engines' snapshots


def test_designs_that_lose_a_probe_are_dropped(steward):
    probes = evolver.default_probes(steward.data_dir)  # no probes.json here, so the Meditations probes
    assert {p.id for p in probes} >= {"morals_vs_temper"}
    result = evolver.evolve(steward.ontology, probes, {"chain": merge_morals})
    assert result["explored"] == 1 and result["designs"] == []
    loose = evolver.evolve(steward.ontology, probes, {"chain": merge_morals}, protect_probes=False)
    assert all("morals_vs_temper" in d.fitness.failed for d in loose["designs"])


def test_improving_design_is_proposed_not_applied(steward):
    onto = steward.ontology
    result = evolver.evolve(onto, [DISCIPLINE], {"chain": link_discipline, "bad": merge_morals})
    best = result["designs"][0]
    assert result["base"]["failed"] == ["discipline_temper"] and "discipline_temper" in best.fitness.passed
    assert [fam for fam, _ in best.steps] == ["chain"]

    version = onto.version
    (patch,) = evolver.queue_designs(steward, {**result, "designs": [best]}, [DISCIPLINE])
    assert patch.kind == "evolve" and patch.status == "proposed" and patch.extra["require_fresh_base"]
    assert patch.extra["probes_gained"] == ["discipline_temper"] and "Discipline steadies" in patch.rationale
    assert onto.version == version and not onto.dag.has_edge("self_discipline", "control_of_temper")

    steward.approve(patch.id, "Android K")
    assert onto.dag.has_edge("self_discipline", "control_of_temper")


def test_stale_design_is_refused(steward):
    result = evolver.evolve(steward.ontology, [DISCIPLINE], {"chain": link_discipline})
    (patch,) = evolver.queue_designs(steward, result, [DISCIPLINE])
    steward.approve(steward.propose(Patch(ops=[Op(op="update_node", id="self_discipline", attrs={"x": 1})])).id, "k")
    with pytest.raises(GovernanceError, match="stale"):
        steward.approve(patch.id, "Android K")


def test_aliases_resolve_probes():
    o = Ontology()
    o.apply_ops([Op(op="add_node", id="virtue", label="Virtue", attrs={"aliases": ["Good morals"]})])
    assert evolver.fitness.check(o, evolver.Probe(kind="exists", a="Good morals"))


def test_discover_skips_missing_engines(monkeypatch):
    monkeypatch.setattr(ops_mod, "ENGINE_MODULES", {"chain": "androidmirror.no_such_engine", "x": "androidmirror.analysis"})
    assert ops_mod.discover() == {}  # missing module, and a module without propose()


def test_extra_steps_must_pay(steward):
    """A design that only adds a useless step on top of a better one is not offered."""
    def noise(snap):
        n = len(snap.nodes)
        return [Patch(kind="refactor", rationale="noise", ops=[Op(op="add_node", id=f"noise{n}", label=f"Noise {n}")])]
    result = evolver.evolve(steward.ontology, [DISCIPLINE], {"chain": link_discipline, "noise": noise}, generations=2)
    assert result["designs"] and all(fam == "chain" for d in result["designs"] for fam, _ in d.steps)


def test_merging_different_concepts_costs_distinctness(steward):
    onto = steward.ontology
    loose = evolver.evolve(onto, [], {"chain": merge_morals}, protect_probes=False, keep=10)
    assert loose["designs"] == []  # smaller, but it conflates two lessons, so not better
    child = evolver.sandbox(onto.graph_snapshot())
    child.apply_ops(merge_morals(onto.graph_snapshot())[0].ops)
    assert evolver.score(child, [], onto).components["distinctness"] < 1.0


def test_with_real_engines_when_installed(steward):
    """Runs once the refactor, break and chain engines are on the branch."""
    found = evolver.discover()
    if len(found) < 3:
        pytest.skip(f"engines installed: {sorted(found)}")
    onto = steward.ontology
    steward.approve(steward.propose(Patch(ops=[
        Op(op="add_node", id="dup", label="Steadiness of purposes", type=onto.nodes["steadiness_of_purpose"]["type"]),
        Op(op="add_causal_edge", src="dup", dst="dignity", sign=1, strength=0.6, confidence=0.6),
    ])).id, "k")
    probes = evolver.default_probes(steward.data_dir)
    before = onto.snapshot()
    result = evolver.evolve(onto, probes)
    assert onto.snapshot() == before and not result["errors"]
    assert result["designs"] and all(d.fitness.total > result["base"]["total"] for d in result["designs"])
