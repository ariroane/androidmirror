"""Break operations: split a node, or sever a link, in both graphs at once.

`analysis.breaker` finds where the subjective world is fragile. This module
turns those findings into proposals:

- split: one node becomes two or more parts. Each of the node's relations
  (knowledge graph) and causal edges (DAG) is moved to one part, or copied to
  every part, so the two graphs stay consistent. The original node is kept,
  archived and emptied, with `split_into` pointing at its parts.
- sever: every link between two nodes is cut together, the causal edge in
  either direction and the descriptive relations between them.

Nothing here changes the ontology. Each function returns a `Patch` of kind
"break" that the Steward queues; it applies only when a person approves it,
and its inverse undoes it. Splitting never creates a cycle: each part's causal
edges are a subset of the original node's, and parts are never linked.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .analysis import propagate
from .ontology import Ontology, slugify
from .patches import Op, Patch, Source

PROPOSER = "Breaker"
ALL = "*"  # in a split assignment: copy the link to every part


def edge_key(kind: str, src: str, dst: str, rel: str | None = None) -> str:
    """Stable key for one link: "kg:src|rel|dst" or "dag:src|dst"."""
    return f"kg:{src}|{rel}|{dst}" if kind == "kg" else f"dag:{src}|{dst}"


def links(onto: Ontology, node: str) -> list[dict[str, Any]]:
    """Every relation and causal edge touching `node`, with its key and the node on the other end."""
    out, seen = [], set()
    for s, d, rel, a in list(onto.kg.in_edges(node, keys=True, data=True)) + list(
        onto.kg.out_edges(node, keys=True, data=True)
    ):
        if (key := edge_key("kg", s, d, rel)) in seen:
            continue  # a self-loop shows up as both an in- and an out-edge
        seen.add(key)
        out.append({"key": key, "kind": "kg", "src": s, "dst": d, "rel": rel,
                    "other": d if s == node else s, "attrs": dict(a)})
    for s, d, a in list(onto.dag.in_edges(node, data=True)) + list(onto.dag.out_edges(node, data=True)):
        out.append({"key": edge_key("dag", s, d), "kind": "dag", "src": s, "dst": d,
                    "other": d if s == node else s, "attrs": dict(a)})
    return out


def _live(onto: Ontology) -> list[str]:
    return [n for n, v in onto.nodes.items() if not v.get("attrs", {}).get("archived")]


def _combined(onto: Ontology) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(_live(onto))
    g.add_edges_from((s, d) for s, d, _ in onto.kg.edges(keys=True) if s in g and d in g)
    g.add_edges_from((s, d) for s, d in onto.dag.edges() if s in g and d in g)
    return g


def neighbor_groups(onto: Ontology, node: str) -> tuple[list[list[str]], list[str]]:
    """The node's neighbours, grouped by which ones stay connected without it.

    Returns (groups, loose_ends). A group is a cluster of two or more nodes the
    node links into; two or more groups mean the node is the only bridge
    between them, a hint that it may be standing in for two different things.
    Loose ends are neighbours linked to nothing else (details, not clusters).
    """
    g = _combined(onto)
    if node not in g:
        return [], []
    nbrs = set(g.neighbors(node))
    g.remove_node(node)
    groups, loose = [], []
    for c in sorted(nx.connected_components(g), key=lambda c: (-len(c), sorted(c))):
        if c & nbrs:
            (groups.append(sorted(c & nbrs)) if len(c) > 1 else loose.extend(c))
    return groups, sorted(loose)


# ---------- finding candidates (read-only) ----------


def split_candidates(onto: Ontology, top: int = 10, include_people: bool = False) -> list[dict[str, Any]]:
    """Nodes that look like several concepts merged into one.

    Signals: the node alone bridges separate clusters (`groups`), and it drives
    other nodes in opposite directions (`ambivalence`, 0..1). People are
    skipped unless `include_people`: the author is a hub by construction, and
    splitting a person into sub-selves is a deliberate choice, not a cleanup.
    """
    rows = []
    for n in _live(onto):
        if onto.nodes[n].get("type") == "person" and not include_people:
            continue
        groups, _ = neighbor_groups(onto, n)
        signs = [a.get("sign", 1) for _, _, a in onto.dag.out_edges(n, data=True)]
        pos, neg = sum(1 for s in signs if s > 0), sum(1 for s in signs if s < 0)
        ambivalence = min(pos, neg) / max(pos, neg) if pos and neg else 0.0
        if len(groups) < 2 and not ambivalence:
            continue
        balance = (len(groups[1]) / len(groups[0])) if len(groups) >= 2 else 0.0
        rows.append({
            "id": n,
            "label": onto.nodes[n]["label"],
            "groups": [[{"id": m, "label": onto.nodes[m]["label"]} for m in grp] for grp in groups],
            "ambivalence": round(ambivalence, 3),
            "score": round(min(len(groups), 4) / 4 * 0.5 + balance * 0.25 + ambivalence * 0.25, 4),
        })
    return sorted(rows, key=lambda r: (r["score"], r["id"]), reverse=True)[:top]


def sever_candidates(onto: Ontology, top: int = 10) -> list[dict[str, Any]]:
    """Links that are weak, yet hold parts of the world together or sit in a tension.

    `weakness` is 1 - strength * confidence of the causal edge (1.0 for a pair
    linked only descriptively). `bridge` means cutting it splits two clusters apart.
    """
    g = _combined(onto)
    bridges = set()
    for e in (nx.bridges(g) if g.number_of_edges() else []):
        cut = g.copy()
        cut.remove_edge(*e)
        if min(len(nx.node_connected_component(cut, e[0])), len(nx.node_connected_component(cut, e[1]))) > 1:
            bridges.add(frozenset(e))  # cutting it splits two clusters, not just a loose end
    pairs: dict[frozenset, dict[str, Any]] = {}
    for s, d, rel in onto.kg.edges(keys=True):
        if s in g and d in g and s != d:
            pairs.setdefault(frozenset((s, d)), {"src": s, "dst": d, "rels": [], "causal": None})["rels"].append(rel)
    for s, d, a in onto.dag.edges(data=True):
        if s in g and d in g:
            p = pairs.setdefault(frozenset((s, d)), {"src": s, "dst": d, "rels": [], "causal": None})
            p["src"], p["dst"], p["causal"] = s, d, a
    rows = []
    for key, p in pairs.items():
        a = p["causal"]
        weakness = 1.0 - a.get("strength", 0.5) * a.get("confidence", 0.5) if a else 1.0
        bridge = key in bridges
        tension = bool(a) and _in_tension(onto, p["src"], p["dst"])
        if a and weakness < 0.75 and not bridge and not tension:
            continue  # strong, redundant, uncontested: nothing to suggest
        rows.append({
            "src": p["src"], "dst": p["dst"],
            "src_label": onto.nodes[p["src"]]["label"], "dst_label": onto.nodes[p["dst"]]["label"],
            "rels": sorted(p["rels"]), "causal": a is not None,
            "weakness": round(weakness, 3), "bridge": bridge, "tension": tension,
            "score": round(weakness * 0.6 + (0.25 if bridge else 0) + (0.15 if tension else 0), 4),
        })
    return sorted(rows, key=lambda r: (r["score"], r["src"], r["dst"]), reverse=True)[:top]


def _in_tension(onto: Ontology, s: str, d: str) -> bool:
    """True if `d` has another cause pushing it the opposite way to `s`."""
    sign = onto.dag.edges[s, d].get("sign", 1)
    return any(onto.dag.edges[p, d].get("sign", 1) != sign for p in onto.dag.predecessors(d) if p != s)


# ---------- building proposals (read-only) ----------


def propose_split(
    onto: Ontology,
    node: str,
    parts: list[dict[str, Any]] | None = None,
    assign: dict[str, str] | None = None,
    rationale: str = "",
    proposed_by: str = PROPOSER,
) -> Patch:
    """A patch that splits `node` into parts across both graphs.

    parts: [{"label", "type"?, "id"?}, ...], at least two. When omitted, one
        part is made per neighbour group (see `neighbor_groups`).
    assign: link key (see `links`) -> part id, or "*" to copy the link to every
        part. Links left out follow their neighbour's group; a link whose
        neighbour is in no group goes to every part.
    """
    if node not in onto.nodes:
        raise KeyError(node)
    orig = onto.nodes[node]
    if orig.get("attrs", {}).get("archived"):
        raise ValueError(f"{node} is archived")
    groups, loose = neighbor_groups(onto, node)
    if parts is None:
        if len(groups) < 2:
            raise ValueError(f"{node} bridges fewer than two groups; name the parts to split it")
        parts = [{"label": f"{orig['label']} ({onto.nodes[grp[0]]['label']})"} for grp in groups]
    if len(parts) < 2:
        raise ValueError("a split needs at least two parts")

    taken = set(onto.nodes)
    ids = []
    for p in parts:
        pid = p.get("id") or slugify(p["label"])
        base, i = pid, 2
        while pid in taken:
            pid, i = f"{base}_{i}", i + 1
        taken.add(pid)
        ids.append(pid)

    by_neighbor = {m: ids[min(i, len(ids) - 1)] for i, grp in enumerate(groups) for m in grp}
    by_neighbor.update({m: ids[0] for m in loose})  # loose ends stay with the largest part
    assign = dict(assign or {})
    for k, v in assign.items():
        if v != ALL and v not in ids:
            raise ValueError(f"link {k} is assigned to unknown part {v!r}")

    ops: list[Op] = []
    for pid, p in zip(ids, parts):
        attrs = {"split_from": node, "provenance": list(orig.get("provenance", []))}
        ops.append(Op(op="add_node", id=pid, label=p["label"], type=p.get("type") or orig.get("type"), attrs=attrs))

    moved: dict[str, list[str]] = {pid: [] for pid in ids}
    for ln in links(onto, node):
        target = assign.get(ln["key"]) or by_neighbor.get(ln["other"], ALL)
        dests = ids if target == ALL else [target]
        swap = lambda x, pid: pid if x == node else x  # noqa: E731
        if ln["kind"] == "kg":
            ops.append(Op(op="remove_relation", src=ln["src"], rel=ln["rel"], dst=ln["dst"]))
            for pid in dests:
                ops.append(Op(op="add_relation", src=swap(ln["src"], pid), rel=ln["rel"],
                              dst=swap(ln["dst"], pid), attrs=dict(ln["attrs"])))
        else:
            ops.append(Op(op="remove_causal_edge", src=ln["src"], dst=ln["dst"]))
            for pid in dests:
                ops.append(_causal(swap(ln["src"], pid), swap(ln["dst"], pid), ln["attrs"]))
        for pid in dests:
            moved[pid].append(ln["key"])

    ops.append(Op(op="update_node", id=node, attrs={"archived": True, "split_into": ids}))
    labels = ", ".join(p["label"] for p in parts)
    summary = "; ".join(f"{p['label']}: {len(moved[pid])} links" for pid, p in zip(ids, parts))
    return Patch(
        kind="break",
        ops=ops,
        rationale=rationale or f"Split {orig['label']} into {labels} ({summary}). The original is archived, not deleted.",
        source=_source(orig),
        proposed_by=proposed_by,
        base_version=onto.version,
        extra={"require_fresh_base": True,
               "breaker": {"operation": "split", "node": node, "parts": ids, "links": moved}},
    )


def propose_sever(
    onto: Ontology,
    src: str,
    dst: str,
    rels: list[str] | None = None,
    causal: bool = True,
    rationale: str = "",
    proposed_by: str = PROPOSER,
) -> Patch:
    """A patch that cuts the links between two nodes in both graphs.

    Removes the causal edge in either direction (unless `causal` is False) and
    the descriptive relations between them in either direction (only those in
    `rels`, if given). The patch records how much causal influence and
    connectivity the cut would lose, so the reviewer can see the cost.
    """
    for n in (src, dst):
        if n not in onto.nodes:
            raise KeyError(n)
    ops: list[Op] = []
    for s, d in ((src, dst), (dst, src)):
        if onto.kg.has_node(s):
            for _, t, rel in onto.kg.out_edges(s, keys=True):
                if t == d and (rels is None or rel in rels):
                    ops.append(Op(op="remove_relation", src=s, rel=rel, dst=d))
        if causal and onto.dag.has_edge(s, d):
            ops.append(Op(op="remove_causal_edge", src=s, dst=d))
    if not ops:
        raise ValueError(f"nothing links {src} and {dst}")

    trial = Ontology()
    trial.nodes, trial.kg, trial.dag = dict(onto.nodes), onto.kg.copy(), onto.dag.copy()
    for op in ops:
        if op.op == "remove_relation":
            trial.kg.remove_edge(op.src, op.dst, key=op.rel)
        else:
            trial.dag.remove_edge(op.src, op.dst)
    reach = lambda g: sum(abs(v) for n in (src, dst) for k, v in propagate(g, {n: 1.0}).items() if k != n)  # noqa: E731
    lost = reach(onto) - reach(trial)
    disconnects = nx.number_connected_components(_combined(trial)) > nx.number_connected_components(_combined(onto))

    a, b = onto.nodes[src]["label"], onto.nodes[dst]["label"]
    cost = [f"{lost:.3f} causal influence lost"] if lost > 1e-9 else []
    cost += ["the world splits in two"] if disconnects else []
    cut_attrs = [onto.kg.edges[o.src, o.dst, o.rel] if o.op == "remove_relation" else onto.dag.edges[o.src, o.dst]
                 for o in ops]
    n = len(ops)
    return Patch(
        kind="break",
        ops=ops,
        rationale=rationale or f"Sever {a} and {b}: {n} link{'s' if n > 1 else ''} cut"
        + (f" ({'; '.join(cost)})." if cost else "."),
        source=_source({"provenance": [p for at in cut_attrs for p in at.get("provenance", [])]}),
        proposed_by=proposed_by,
        base_version=onto.version,
        warnings=["this cut disconnects the graph"] if disconnects else [],
        extra={"require_fresh_base": True,
               "breaker": {"operation": "sever", "src": src, "dst": dst, "influence_lost": round(lost, 4),
                           "disconnects": disconnects}},
    )


def _causal(s: str, d: str, a: dict[str, Any]) -> Op:
    return Op(op="add_causal_edge", src=s, dst=d, sign=a.get("sign", 1), strength=a.get("strength"),
              confidence=a.get("confidence"), mechanism=a.get("mechanism", ""),
              attrs={"provenance": list(a.get("provenance", []))})


def _source(node: dict[str, Any]) -> Source:
    """Cite the quote that grounded the node being broken, if it has one."""
    for prov in node.get("provenance", []):
        if isinstance(prov, dict) and prov.get("quote"):
            return Source(work=prov.get("work", ""), locator=prov.get("locator", ""), quote=prov["quote"])
    return Source(locator="Breaker")
