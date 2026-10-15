"""The Refactorer: proposes ways to condense or restructure the two graphs.

It reads a GraphSnapshot, finds places where the knowledge graph or the causal
DAG is redundant or inconsistent, and returns refactor patches. It never
changes the ontology: every patch it makes goes to the Steward's queue and is
applied only when a person approves it there. Nothing here runs on its own;
it runs when someone asks for suggestions (`python -m androidmirror.refactor`).

v1 detectors are structural (no Claude call):
- merge_duplicates: nodes of the same type with near-identical labels
- normalize_relations: relation names that differ only in form ("value" / "values")
- reduce_transitive: a causal edge already explained by a longer path, with no quote of its own
- archive_orphans: nodes with no edges and no quote, flagged as archived (never deleted)

Contradictions (a direct causal edge whose sign disagrees with a longer path)
are reported as findings for the user to judge, never resolved here.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from .ontology import GraphSnapshot, Ontology, OntologyError
from .patches import Op, Patch, Source

LABEL_SIMILARITY = 0.88
MAX_PATH_LEN = 4  # longest indirect path considered when reducing a causal edge


@dataclass
class Finding:
    """Something the user should judge; not a patch."""

    type: str
    summary: str
    nodes: list[str] = field(default_factory=list)


@dataclass
class Suggestions:
    base_version: int
    patches: list[Patch]
    findings: list[Finding]
    metrics: dict[str, float]


# ---------- entry points ----------


def suggest(snap: GraphSnapshot) -> Suggestions:
    """Return refactor patches (not applied) and findings for one snapshot."""
    candidates: list[tuple[str, str, list[Op], str, str, dict[str, Any]]] = []
    candidates += _merge_duplicates(snap)
    candidates += _normalize_relations(snap)
    candidates += _reduce_transitive(snap)
    candidates += _archive_orphans(snap)

    before = metrics(snap)
    patches = []
    for rtype, key, ops, rationale, risk, evidence in candidates:
        after = _dry_run(snap, ops)
        if after is None:
            continue  # would not apply cleanly (for example a merge that creates a cycle)
        patches.append(Patch(
            kind="refactor",
            ops=ops,
            rationale=rationale,
            source=Source(work="refactor", locator=rtype),
            proposed_by="refactorer",
            base_version=snap.version,
            extra={
                "refactor_type": rtype,
                "key": key,
                "risk": risk,
                "evidence": evidence,
                "metrics": {k: [before[k], after[k]] for k in before if before[k] != after[k]},
                "require_fresh_base": True,
            },
        ))
    return Suggestions(snap.version, patches, contradictions(snap), before)


def queue_suggestions(steward) -> Suggestions:
    """Put new suggestions in the Steward's queue as proposed patches.

    Proposing is all this does: the patches wait for a person to approve,
    edit or reject them. A suggestion already waiting in the queue is not
    queued twice.
    """
    result = suggest(steward.ontology.graph_snapshot())
    pending = {p.extra.get("key") for p in steward.list("proposed") if p.kind == "refactor"}
    queued = []
    for patch in result.patches:
        if patch.extra["key"] in pending:
            continue
        queued.append(steward.propose(patch))
    if queued:
        steward.save()
    result.patches = queued
    return result


# ---------- detectors ----------


def _merge_duplicates(snap: GraphSnapshot):
    """Nodes of one type whose labels match closely: fold the rest into the best-connected one."""
    by_type: dict[str, list[str]] = {}
    for nid, node in snap.nodes.items():
        if not node.get("attrs", {}).get("archived"):
            by_type.setdefault(node.get("type", "concept"), []).append(nid)

    groups = nx.Graph()
    for ids in by_type.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if _same_label(snap.nodes[a], snap.nodes[b]):
                    groups.add_edge(a, b)

    out = []
    for group in nx.connected_components(groups):
        ids = sorted(group, key=lambda n: (-_degree(snap, n), n))
        keep, absorb = ids[0], ids[1:]
        ops = _merge_ops(snap, keep, absorb)
        labels = ", ".join(repr(snap.nodes[n]["label"]) for n in absorb)
        out.append((
            "merge_duplicates",
            "merge:" + ",".join(sorted(group)),
            ops,
            f"Merge {labels} into {snap.nodes[keep]['label']!r}: same type and nearly the same label. "
            "Their relations, causal edges and quotes move to the kept node.",
            "merge",
            {"keep": keep, "absorb": absorb},
        ))
    return out


def _merge_ops(snap: GraphSnapshot, keep: str, absorb: list[str]) -> list[Op]:
    ops: list[Op] = []
    kept = snap.nodes[keep]
    aliases = list(kept.get("attrs", {}).get("aliases", []))
    absorbed = list(kept.get("attrs", {}).get("absorbed", []))
    have_rel = {(s, k, d) for s, d, k in snap.kg.edges(keys=True)}
    have_dag = set(snap.dag.edges())
    gone = set(absorb)

    def to_keep(n: str) -> str:
        return keep if n in gone else n

    for old in absorb:
        node = snap.nodes[old]
        aliases.append(node["label"])
        absorbed.append({"id": old, "label": node["label"], "provenance": node.get("provenance", [])})
        for s, d, k, a in list(snap.kg.in_edges(old, keys=True, data=True)) + list(
            snap.kg.out_edges(old, keys=True, data=True)
        ):
            s2, d2 = to_keep(s), to_keep(d)
            if s2 != d2 and (s2, k, d2) not in have_rel:
                have_rel.add((s2, k, d2))
                ops.append(Op(op="add_relation", src=s2, rel=k, dst=d2, attrs=_clean(a)))
        for s, d, a in list(snap.dag.in_edges(old, data=True)) + list(snap.dag.out_edges(old, data=True)):
            s2, d2 = to_keep(s), to_keep(d)
            if s2 != d2 and (s2, d2) not in have_dag:
                have_dag.add((s2, d2))
                ops.append(Op(op="add_causal_edge", src=s2, dst=d2, sign=a.get("sign", 1),
                              strength=a.get("strength"), confidence=a.get("confidence"),
                              mechanism=a.get("mechanism", ""),
                              attrs={"provenance": list(a.get("provenance", []))}))
    ops.append(Op(op="update_node", id=keep,
                  attrs={"aliases": sorted(set(aliases)), "absorbed": absorbed}))
    ops += [Op(op="remove_node", id=old) for old in absorb]
    return ops


def _normalize_relations(snap: GraphSnapshot):
    """Relation names that differ only in case, separators or a plural 's'."""
    counts: dict[str, int] = {}
    for _, _, k in snap.kg.edges(keys=True):
        counts[k] = counts.get(k, 0) + 1
    families: dict[str, list[str]] = {}
    for name in counts:
        families.setdefault(_rel_stem(name), []).append(name)

    out = []
    for names in families.values():
        if len(names) < 2:
            continue
        canon = max(names, key=lambda n: (counts[n], -len(n), n))
        ops: list[Op] = []
        added = set()
        moved = 0
        for s, d, k, a in snap.kg.edges(keys=True, data=True):
            if k == canon or k not in names:
                continue
            ops.append(Op(op="remove_relation", src=s, rel=k, dst=d))
            if not snap.kg.has_edge(s, d, key=canon) and (s, d) not in added:
                added.add((s, d))
                ops.append(Op(op="add_relation", src=s, rel=canon, dst=d, attrs=_clean(a)))
            moved += 1
        others = sorted(n for n in names if n != canon)
        out.append((
            "normalize_relations",
            "relations:" + ",".join(sorted(names)),
            ops,
            f"Rename relation {', '.join(repr(n) for n in others)} to {canon!r} on {moved} edges; "
            "they differ only in form.",
            "rename",
            {"canonical": canon, "renamed": others},
        ))
    return out


def _reduce_transitive(snap: GraphSnapshot):
    """A direct causal edge that a longer path with the same sign already explains."""
    out = []
    for a, c, attrs in snap.dag.edges(data=True):
        if _has_quote(attrs):
            continue  # the text states this link directly: keep it
        path = _indirect_path(snap.dag, a, c)
        if not path or _path_sign(snap.dag, path) != attrs.get("sign", 1):
            continue
        via = " -> ".join(snap.nodes[n]["label"] for n in path)
        out.append((
            "reduce_transitive",
            f"reduce:{a}->{c}",
            [Op(op="remove_causal_edge", src=a, dst=c)],
            f"Remove the direct edge {snap.nodes[a]['label']} -> {snap.nodes[c]['label']}: "
            f"it has no quote of its own and the path {via} already carries it with the same sign.",
            "remove",
            {"path": path},
        ))
    return out


def _archive_orphans(snap: GraphSnapshot):
    """Nodes with no edges in either graph and no quote: flag them archived, never delete."""
    out = []
    for nid, node in snap.nodes.items():
        if node.get("attrs", {}).get("archived"):
            continue
        if snap.kg.degree(nid) or snap.dag.degree(nid) or _has_quote(node):
            continue
        out.append((
            "archive_orphans",
            f"archive:{nid}",
            [Op(op="update_node", id=nid, attrs={"archived": True})],
            f"Archive {node['label']!r}: it has no relations, no causal edges and no quote. "
            "It stays in the ontology, marked archived, and can be restored.",
            "archive",
            {},
        ))
    return out


def contradictions(snap: GraphSnapshot) -> list[Finding]:
    """Direct causal edges whose sign disagrees with an indirect path between the same nodes."""
    found = []
    for a, c, attrs in snap.dag.edges(data=True):
        path = _indirect_path(snap.dag, a, c)
        if path and _path_sign(snap.dag, path) != attrs.get("sign", 1):
            via = " -> ".join(snap.nodes[n]["label"] for n in path)
            word = "raises" if attrs.get("sign", 1) > 0 else "lowers"
            found.append(Finding(
                "sign_conflict",
                f"{snap.nodes[a]['label']} directly {word} {snap.nodes[c]['label']}, "
                f"but the path {via} has the opposite sign. Is this a real tension, or two different ideas?",
                [a, c],
            ))
    return found


# ---------- scoring ----------


def metrics(snap: GraphSnapshot) -> dict[str, float]:
    """Efficiency and consistency readings shown before and after each proposal."""
    nodes = len(snap.nodes)
    kg_edges = snap.kg.number_of_edges()
    dag_edges = snap.dag.number_of_edges()
    records = list(snap.nodes.values()) + [a for *_, a in snap.kg.edges(data=True)] + [
        a for *_, a in snap.dag.edges(data=True)
    ]
    quoted = sum(1 for r in records if _has_quote(r))
    return {
        "nodes": nodes,
        "relations": kg_edges,
        "causal_edges": dag_edges,
        "relation_types": len({k for *_, k in snap.kg.edges(keys=True)}),
        "duplicate_groups": len(_merge_duplicates_groups(snap)),
        "redundant_causal_edges": len(_reduce_transitive(snap)),
        "sign_conflicts": len(contradictions(snap)),
        "orphans": sum(1 for n, node in snap.nodes.items() if not snap.kg.degree(n) and not snap.dag.degree(n)
                       and not node.get("attrs", {}).get("archived")),
        "provenance_coverage": round(quoted / len(records), 3) if records else 0.0,
    }


def _merge_duplicates_groups(snap: GraphSnapshot) -> list[set[str]]:
    return [set([c[5]["keep"], *c[5]["absorb"]]) for c in _merge_duplicates(snap)]


def _dry_run(snap: GraphSnapshot, ops: list[Op]) -> dict[str, float] | None:
    """Apply the ops to a private copy and score it; None if they would not apply."""
    trial = Ontology()
    trial.nodes = copy.deepcopy(snap.nodes)
    trial.kg, trial.dag = snap.kg.copy(), snap.dag.copy()
    trial.version = snap.version
    try:
        trial.apply_ops(ops)
    except OntologyError:
        return None
    return metrics(trial.graph_snapshot())


# ---------- helpers ----------


def _norm_label(label: str) -> str:
    words = re.sub(r"[^a-z0-9 ]+", " ", label.lower()).split()
    words = [w for w in words if w not in {"the", "a", "an", "of", "one's", "ones"}]
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words)


def _same_label(a: dict[str, Any], b: dict[str, Any]) -> bool:
    la, lb = _norm_label(a.get("label", "")), _norm_label(b.get("label", ""))
    if not la or not lb:
        return False
    if la == lb:
        return True
    aliases_a = {_norm_label(x) for x in a.get("attrs", {}).get("aliases", [])}
    aliases_b = {_norm_label(x) for x in b.get("attrs", {}).get("aliases", [])}
    if lb in aliases_a or la in aliases_b:
        return True
    return min(len(la), len(lb)) >= 6 and difflib.SequenceMatcher(None, la, lb).ratio() >= LABEL_SIMILARITY


def _rel_stem(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", name.lower())
    return s[:-1] if len(s) > 3 and s.endswith("s") else s


def _degree(snap: GraphSnapshot, n: str) -> int:
    return snap.kg.degree(n) + snap.dag.degree(n) + len(snap.nodes[n].get("provenance", []))


def _has_quote(record: dict[str, Any]) -> bool:
    return any(p.get("quote") for p in record.get("provenance", []) if isinstance(p, dict))


def _indirect_path(dag: nx.DiGraph, a: str, c: str) -> list[str] | None:
    """Shortest path from a to c of length 2..MAX_PATH_LEN that avoids the direct edge."""
    g = nx.restricted_view(dag, [], [(a, c)])
    try:
        path = nx.shortest_path(g, a, c)
    except nx.NetworkXNoPath:
        return None
    return path if 3 <= len(path) <= MAX_PATH_LEN + 1 else None


def _path_sign(dag: nx.DiGraph, path: list[str]) -> int:
    sign = 1
    for s, d in zip(path, path[1:]):
        sign *= 1 if dag.edges[s, d].get("sign", 1) >= 0 else -1
    return sign


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in attrs.items() if k != "modified"}


# ---------- command line ----------


def main(argv: list[str] | None = None) -> None:
    from .steward import Steward

    ap = argparse.ArgumentParser(prog="python -m androidmirror.refactor",
                                 description="Queue refactor suggestions for your approval. Nothing is applied.")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--dry", action="store_true", help="print suggestions without queuing them")
    args = ap.parse_args(argv)

    steward = Steward(Ontology.load(args.data / "ontology.json"), args.data)
    if args.dry:
        result = suggest(steward.ontology.graph_snapshot())
    else:
        result = queue_suggestions(steward)
    verb = "suggested" if args.dry else "queued for approval"
    print(f"version {result.base_version}: {len(result.patches)} refactors {verb}")
    for p in result.patches:
        print(f"  {p.id}  [{p.extra['risk']}] {p.rationale}")
        if p.extra["metrics"]:
            print("      " + json.dumps(p.extra["metrics"]))
    for f in result.findings:
        print(f"  finding: {f.summary}")
    if result.patches and not args.dry:
        print("Review them with `androidmirror pending` or in the web app; nothing changes until you approve.")


if __name__ == "__main__":
    main()
