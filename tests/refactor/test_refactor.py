import ast
from pathlib import Path

import pytest

from androidmirror import refactor
from androidmirror.ontology import Ontology
from androidmirror.patches import Op, Patch
from androidmirror.steward import GovernanceError, Steward

QUOTE = {"patch": "p0", "locator": "Book V, 11", "quote": "the ruling faculty"}


def build(tmp_path) -> Steward:
    """A small Meditations-shaped ontology with one of each problem."""
    st = Steward(Ontology(), tmp_path)
    ops = [
        Op(op="add_node", id="ruling_faculty", label="Ruling faculty", type="concept",
           attrs={"provenance": [QUOTE]}),
        Op(op="add_node", id="the_ruling_faculties", label="The ruling faculties", type="concept"),
        Op(op="add_node", id="judgment", label="Judgment", type="concept"),
        Op(op="add_node", id="impression", label="Impression", type="concept"),
        Op(op="add_node", id="disturbance", label="Disturbance", type="emotion"),
        Op(op="add_node", id="tranquility", label="Tranquility", type="value"),
        Op(op="add_node", id="fame", label="Fame", type="concept"),
        Op(op="add_node", id="stray", label="Stray thought", type="concept"),
        Op(op="add_relation", src="ruling_faculty", rel="governs", dst="judgment"),
        Op(op="add_relation", src="the_ruling_faculties", rel="governs", dst="impression"),
        Op(op="add_relation", src="judgment", rel="values", dst="tranquility"),
        Op(op="add_relation", src="impression", rel="value", dst="fame"),
        Op(op="add_causal_edge", src="impression", dst="judgment", sign=1),
        Op(op="add_causal_edge", src="judgment", dst="disturbance", sign=1),
        Op(op="add_causal_edge", src="impression", dst="disturbance", sign=1),  # redundant, no quote
        Op(op="add_causal_edge", src="disturbance", dst="tranquility", sign=-1),
        Op(op="add_causal_edge", src="judgment", dst="tranquility", sign=1),  # conflicts with the path
    ]
    st.ontology.apply_ops(ops)
    return st


def by_type(result):
    return {p.extra["refactor_type"]: p for p in result.patches}


def test_suggest_finds_each_refactor_and_changes_nothing(tmp_path):
    st = build(tmp_path)
    before = st.ontology.snapshot()
    result = refactor.suggest(st.ontology.graph_snapshot())
    found = by_type(result)
    assert set(found) == {"merge_duplicates", "normalize_relations", "reduce_transitive", "archive_orphans"}
    assert found["merge_duplicates"].extra["evidence"] == {"keep": "ruling_faculty",
                                                          "absorb": ["the_ruling_faculties"]}
    assert found["reduce_transitive"].ops == [Op(op="remove_causal_edge", src="impression", dst="disturbance")]
    assert found["archive_orphans"].ops[0].id == "stray"
    assert [f.nodes for f in result.findings] == [["judgment", "tranquility"]]
    for p in result.patches:
        assert p.kind == "refactor" and p.status == "proposed" and p.approval is None
        assert p.base_version == st.ontology.version and p.extra["require_fresh_base"]
        assert p.extra["metrics"]
    assert st.ontology.snapshot() == before


def test_propose_returns_unqueued_patches(tmp_path):
    st = build(tmp_path)
    patches = refactor.propose(st.ontology.graph_snapshot())
    assert {p.extra["refactor_type"] for p in patches} == set(by_type(refactor.suggest(st.ontology.graph_snapshot())))
    assert st.list() == [] and not (tmp_path / "patches.json").exists()


def test_queue_only_proposes_and_does_not_duplicate(tmp_path):
    st = build(tmp_path)
    version = st.ontology.version
    first = refactor.queue_suggestions(st)
    assert first.patches and st.ontology.version == version
    assert all(p.status == "proposed" for p in st.list())
    again = refactor.queue_suggestions(st)
    assert again.patches == [] and len(st.list("proposed")) == len(first.patches)
    reloaded = Steward(Ontology.load(tmp_path / "ontology.json"), tmp_path)
    assert {p.id for p in reloaded.list("proposed")} == {p.id for p in first.patches}


def test_approved_merge_moves_edges_and_quotes_and_rolls_back(tmp_path):
    st = build(tmp_path)
    merge = by_type(refactor.queue_suggestions(st))["merge_duplicates"]
    before = st.ontology.snapshot()
    st.approve(merge.id, "Android K")
    o = st.ontology
    assert "the_ruling_faculties" not in o.nodes
    assert o.kg.has_edge("ruling_faculty", "impression", key="governs")
    assert o.nodes["ruling_faculty"]["attrs"]["aliases"] == ["The ruling faculties"]
    assert o.nodes["ruling_faculty"]["provenance"][0]["quote"] == "the ruling faculty"
    st.rollback(merge.id, "Android K")
    strip = lambda rows: sorted(str({k: v for k, v in r.items() if k not in ("modified", "provenance")})  # noqa: E731
                                for r in rows)
    after = st.ontology.snapshot()
    for key in ("nodes", "relations", "causal_edges"):
        assert strip(before[key]) == strip(after[key])


def test_normalize_relations_only_touches_its_family(tmp_path):
    st = build(tmp_path)
    patch = by_type(refactor.suggest(st.ontology.graph_snapshot()))["normalize_relations"]
    assert {o.rel for o in patch.ops} == {"value", "values"}
    st.approve(st.propose(patch).id, "Android K")
    rels = {k for *_, k in st.ontology.kg.edges(keys=True)}
    assert rels == {"governs", "value"} or rels == {"governs", "values"}


def test_stale_refactor_is_refused(tmp_path):
    st = build(tmp_path)
    orphan = by_type(refactor.queue_suggestions(st))["archive_orphans"]
    st.approve(st.propose(Patch(ops=[Op(op="update_node", id="stray", label="A stray thought")])).id, "Android K")
    with pytest.raises(GovernanceError, match="stale"):
        st.approve(orphan.id, "Android K")
    assert not st.ontology.nodes["stray"]["attrs"].get("archived")


def test_merge_that_would_create_a_cycle_is_not_proposed(tmp_path):
    st = Steward(Ontology(), tmp_path)
    st.ontology.apply_ops([
        Op(op="add_node", id="virtue", label="Virtue"),
        Op(op="add_node", id="virtues", label="Virtues"),
        Op(op="add_node", id="habit", label="Habit"),
        Op(op="add_causal_edge", src="virtue", dst="habit"),
        Op(op="add_causal_edge", src="habit", dst="virtues"),
    ])
    assert "merge_duplicates" not in by_type(refactor.suggest(st.ontology.graph_snapshot()))


def test_archived_nodes_are_flagged_not_deleted(tmp_path):
    st = build(tmp_path)
    orphan = by_type(refactor.queue_suggestions(st))["archive_orphans"]
    st.approve(orphan.id, "Android K")
    assert st.ontology.nodes["stray"]["attrs"]["archived"] is True
    assert "archive_orphans" not in by_type(refactor.suggest(st.ontology.graph_snapshot()))


def test_refactorer_has_no_path_to_approval_or_scheduling():
    """Never automatic: the module cannot approve, reject or roll back, and starts nothing in the background."""
    source = Path(refactor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not called & {"approve", "reject", "rollback"}
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names} | {(n.module or "").split(".")[0] for n in ast.walk(tree)
                                    if isinstance(n, ast.ImportFrom)}
    assert not imported & {"threading", "sched", "asyncio", "subprocess", "multiprocessing", "apscheduler"}
    # apply_ops runs only on the private trial copy inside _dry_run
    applies = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "apply_ops"]
    assert [ast.unparse(n.func.value) for n in applies] == ["trial"]


def test_cli_dry_run_queues_nothing(tmp_path, capsys):
    st = build(tmp_path)
    st.save()
    refactor.main(["--data", str(tmp_path), "--dry"])
    assert "refactors suggested" in capsys.readouterr().out
    assert Steward(Ontology.load(tmp_path / "ontology.json"), tmp_path).list() == []
