"""The Chainer: union operations across the knowledge graph and the causal DAG.

Two operations, each a read-only planner that returns a `Patch` (kind "chain")
for the Steward's queue. Nothing here changes the ontology; a person approves
the patch, and the Steward applies it and keeps its inverse for rollback.

- merge: fold one or more nodes into a survivor. Relations and causal edges move
  to the survivor in both graphs at once, duplicate edges collapse (keeping all
  provenance), and edges between the merged nodes are dropped.
- chain: union another ontology (another mirror's `ontology.json`) into this one.
  Nodes that mean the same thing are aligned onto existing nodes; the rest come
  in under a namespace (`"epictetus:fear"`), so several mirrors can form one
  community context.

Both refuse, rather than resolve, what the user should decide: a union that
would create a causal cycle, or two causal edges between the same pair with
opposite signs (unless `on_conflict` says which to keep).
"""

from __future__ import annotations

import copy
import difflib
import json
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, Field

from .ontology import GraphSnapshot, Ontology, OntologyError, slugify
from .patches import Op, Patch, Source

OnConflict = Literal["refuse", "keep_existing", "keep_incoming"]

PROPOSER = "chainer"


class ChainError(ValueError):
    pass


# ---------- merge ----------


def plan_merge(
    snap: GraphSnapshot,
    survivor: str,
    absorbed: list[str],
    label: str | None = None,
    on_conflict: OnConflict = "refuse",
    rationale: str = "",
) -> Patch:
    """Propose folding `absorbed` into `survivor` across both graphs."""
    absorbed = list(dict.fromkeys(absorbed))
    for nid in [survivor, *absorbed]:
        if nid not in snap.nodes:
            raise ChainError(f"unknown node: {nid!r}")
    if not absorbed or survivor in absorbed:
        raise ChainError("name a survivor and at least one other node to merge into it")
    group = set(absorbed)

    def m(n: str) -> str:
        return survivor if n in group else n

    _check_acyclic(((m(s), m(d)) for s, d in snap.dag.edges), "merging these nodes")

    ops: list[Op] = [Op(op="remove_node", id=nid) for nid in absorbed]
    dropped: list[str] = []

    rels: dict[tuple[str, str, str], dict] = {}
    for nid in absorbed:
        for s, d, k, a in list(snap.kg.in_edges(nid, keys=True, data=True)) + list(
            snap.kg.out_edges(nid, keys=True, data=True)
        ):
            rels.setdefault((s, d, k), a)
    incoming_rels = []
    for (s, d, k), a in rels.items():
        if m(s) == m(d):
            dropped.append(f"relation {s} -{k}-> {d} (both sides merged)")
        else:
            incoming_rels.append((m(s), k, m(d), a))

    causal: dict[tuple[str, str], dict] = {}
    for nid in absorbed:
        for s, d, a in list(snap.dag.in_edges(nid, data=True)) + list(snap.dag.out_edges(nid, data=True)):
            causal.setdefault((s, d), a)
    incoming_causal = []
    for (s, d), a in causal.items():
        if m(s) == m(d):
            dropped.append(f"causal edge {s} -> {d} (both sides merged)")
        else:
            incoming_causal.append((m(s), m(d), a, f"{s} -> {d}"))

    conflicts = _union_edges(snap, group, incoming_rels, incoming_causal, on_conflict, ops)

    node = snap.nodes[survivor]
    merged_from = list(node["attrs"].get("merged_from", []))
    aliases = list(node["attrs"].get("aliases", []))
    for nid in absorbed:
        n = snap.nodes[nid]
        merged_from.append({"id": nid, "label": n["label"], "type": n["type"], "attrs": copy.deepcopy(n["attrs"]),
                            "provenance": copy.deepcopy(n.get("provenance", []))})
        if n["label"] not in aliases and n["label"] != (label or node["label"]):
            aliases.append(n["label"])
    ops.append(Op(op="update_node", id=survivor, label=label, attrs={"merged_from": merged_from, "aliases": aliases}))

    names = ", ".join(snap.nodes[n]["label"] for n in absorbed)
    patch = Patch(
        kind="chain",
        ops=ops,
        rationale=rationale or f"Merge {names} into {label or node['label']} in both graphs.",
        source=Source(work="chainer", locator="merge"),
        proposed_by=PROPOSER,
        base_version=snap.version,
        extra={"operation": "merge", "survivor": survivor, "absorbed": absorbed, "dropped": dropped,
               "conflicts": conflicts, "require_fresh_base": True},
    )
    return _finish(snap, patch)


# ---------- chain ontologies ----------


def suggest_alignment(snap: GraphSnapshot, other: GraphSnapshot, cutoff: float = 0.85) -> dict[str, Any]:
    """Which nodes of `other` look like nodes already here.

    `exact` (same label) is what `plan_chain` aligns by default; `possible`
    are near matches it leaves alone unless the user aligns them explicitly.
    """
    by_label = {slugify(n["label"]): nid for nid, n in snap.nodes.items()}
    exact: dict[str, str] = {}
    possible: list[dict[str, Any]] = []
    for oid, n in other.nodes.items():
        key = slugify(n["label"])
        hit = by_label.get(key)
        if hit:
            exact[oid] = hit
            continue
        for cand in difflib.get_close_matches(key, list(by_label), n=3, cutoff=cutoff):
            score = difflib.SequenceMatcher(None, key, cand).ratio()
            possible.append({"other": oid, "base": by_label[cand], "score": round(score, 3)})
    return {"exact": exact, "possible": possible}


def plan_chain(
    snap: GraphSnapshot,
    other: GraphSnapshot,
    namespace: str,
    align: dict[str, str] | None = None,
    separate: list[str] | None = None,
    on_conflict: OnConflict = "refuse",
    rationale: str = "",
) -> Patch:
    """Propose the union of `other` into this ontology.

    `align` maps other ids to ids here (adds to or overrides the exact label
    matches); ids in `separate` are never aligned and come in namespaced, with
    the namespace in their label if it would otherwise clash.
    """
    ns = slugify(namespace)
    if not namespace.strip() or ns == "node":
        raise ChainError("give the chained ontology a name")
    if any(n.get("attrs", {}).get("ontology") == ns for n in snap.nodes.values()):
        raise ChainError(f"an ontology named {ns!r} is already chained in; roll that patch back or pick another name")
    separate_set = set(separate or [])
    mapping = {k: v for k, v in suggest_alignment(snap, other)["exact"].items() if k not in separate_set}
    for oid, bid in (align or {}).items():
        if oid not in other.nodes:
            raise ChainError(f"{oid!r} is not in the {ns} ontology")
        if bid not in snap.nodes:
            raise ChainError(f"{bid!r} is not in this ontology")
        mapping[oid] = bid

    labels_here = {slugify(n["label"]) for n in snap.nodes.values()} | set(snap.nodes)
    ops: list[Op] = []
    for oid, n in other.nodes.items():
        if oid in mapping:
            continue
        nid = f"{ns}:{oid}"
        lbl = n["label"]
        if slugify(lbl) in labels_here:
            lbl = f"{lbl} ({namespace.strip()})"  # keeps the Steward's dedupe from folding it in
        attrs = {**copy.deepcopy(n.get("attrs", {})), "ontology": ns, "provenance": copy.deepcopy(n.get("provenance", []))}
        ops.append(Op(op="add_node", id=nid, label=lbl, type=n.get("type"), attrs=attrs))
        mapping[oid] = nid

    chained: dict[str, list] = {}
    for oid, bid in mapping.items():
        if bid in snap.nodes:
            chained.setdefault(bid, list(snap.nodes[bid]["attrs"].get("chained_with", [])))
            chained[bid].append({"ontology": ns, "id": oid, "label": other.nodes[oid]["label"]})

    _check_acyclic(
        list(snap.dag.edges) + [(mapping[s], mapping[d]) for s, d in other.dag.edges if mapping[s] != mapping[d]],
        f"chaining {ns}",
    )
    dropped: list[str] = []
    incoming_rels = []
    for s, d, k, a in other.kg.edges(keys=True, data=True):
        if mapping[s] == mapping[d]:
            dropped.append(f"relation {s} -{k}-> {d} (both ends align to {mapping[s]})")
        else:
            incoming_rels.append((mapping[s], k, mapping[d], a))
    incoming_causal = []
    for s, d, a in other.dag.edges(data=True):
        if mapping[s] == mapping[d]:
            dropped.append(f"causal edge {s} -> {d} (both ends align to {mapping[s]})")
        else:
            incoming_causal.append((mapping[s], mapping[d], a, f"{ns}: {s} -> {d}"))

    conflicts = _union_edges(snap, set(), incoming_rels, incoming_causal, on_conflict, ops)
    for bid, entries in chained.items():
        ops.append(Op(op="update_node", id=bid, attrs={"chained_with": entries}))

    aligned = {k: v for k, v in mapping.items() if v in snap.nodes}
    patch = Patch(
        kind="chain",
        ops=ops,
        rationale=rationale or f"Chain the {namespace.strip()} ontology into this one: "
        f"{len(aligned)} shared nodes, {len(mapping) - len(aligned)} new.",
        source=Source(work=f"ontology:{ns}", locator="chain"),
        proposed_by=PROPOSER,
        base_version=snap.version,
        extra={"operation": "chain", "namespace": ns, "alignment": aligned, "dropped": dropped,
               "conflicts": conflicts, "require_fresh_base": True},
    )
    return _finish(snap, patch)


# ---------- shared ----------


def _union_edges(
    snap: GraphSnapshot,
    removed: set[str],
    incoming_rels: list[tuple[str, str, str, dict]],
    incoming_causal: list[tuple[str, str, dict, str]],
    on_conflict: OnConflict,
    ops: list[Op],
) -> list[dict[str, Any]]:
    """Append ops that add incoming edges, collapsing duplicates. Returns conflicts."""
    rels: dict[tuple[str, str, str], dict] = {}
    for s, k, d, a in incoming_rels:
        key = (s, d, k)
        existing = snap.kg.edges[s, d, k] if s not in removed and d not in removed and snap.kg.has_edge(s, d, key=k) else None
        if key not in rels:
            rels[key] = {"existing": existing, "attrs": copy.deepcopy(dict(existing or a))}
        rels[key]["attrs"]["provenance"] = _union_prov(rels[key]["attrs"].get("provenance", []), a.get("provenance", []))
    for (s, d, k), r in rels.items():
        if r["existing"] is not None:
            if r["attrs"].get("provenance", []) == r["existing"].get("provenance", []):
                continue
            ops.append(Op(op="remove_relation", src=s, rel=k, dst=d))
        ops.append(Op(op="add_relation", src=s, rel=k, dst=d, attrs=r["attrs"]))

    pairs: dict[tuple[str, str], list[tuple[dict, str]]] = {}
    for s, d, a, origin in incoming_causal:
        pairs.setdefault((s, d), []).append((a, origin))
    conflicts: list[dict[str, Any]] = []
    for (s, d), cands in pairs.items():
        existing = snap.dag.edges[s, d] if s not in removed and d not in removed and snap.dag.has_edge(s, d) else None
        ranked = ([(existing, "existing")] if existing is not None else []) + cands
        signs = {_sign(a) for a, _ in ranked}
        if len(signs) > 1:
            entry = {"src": s, "dst": d, "edges": [{"from": o, "sign": _sign(a)} for a, o in ranked]}
            if on_conflict == "refuse":
                conflicts.append(entry)
                continue
            pick = ranked[0] if on_conflict == "keep_existing" else (cands[0] if cands else ranked[0])
            entry["kept_sign"] = _sign(pick[0])
            conflicts.append(entry)
            ranked = [(a, o) for a, o in ranked if _sign(a) == entry["kept_sign"]]
        merged = _combine([a for a, _ in ranked])
        if existing is not None and merged == {k: existing.get(k) for k in merged}:
            continue
        ops.append(Op(op="add_causal_edge", src=s, dst=d, sign=merged["sign"], strength=merged["strength"],
                      confidence=merged["confidence"], mechanism=merged["mechanism"],
                      attrs={"provenance": merged["provenance"]}))
    if on_conflict == "refuse" and conflicts:
        detail = "; ".join(f"{c['src']} -> {c['dst']} ({', '.join(e['from'] for e in c['edges'])})" for c in conflicts)
        raise ChainError(
            f"opposite causal signs between the same nodes: {detail}. "
            "Choose on_conflict='keep_existing' or 'keep_incoming', or fix the edges first."
        )
    return conflicts


def _sign(a: dict) -> int:
    return 1 if (a.get("sign", 1) or 1) >= 0 else -1


def _combine(edges: list[dict]) -> dict[str, Any]:
    mechanisms = list(dict.fromkeys(e.get("mechanism", "") for e in edges if e.get("mechanism")))
    prov: list = []
    for e in edges:
        prov = _union_prov(prov, e.get("provenance", []))
    return {
        "sign": _sign(edges[0]),
        "strength": max(float(e.get("strength", 0.5)) for e in edges),
        "confidence": max(float(e.get("confidence", 0.5)) for e in edges),
        "mechanism": "; ".join(mechanisms),
        "provenance": prov,
    }


def _union_prov(a: list, b: list) -> list:
    out = list(copy.deepcopy(a))
    for p in b:
        if p not in out:
            out.append(copy.deepcopy(p))
    return out


def _check_acyclic(edges, what: str) -> None:
    g = nx.DiGraph()
    g.add_edges_from((s, d) for s, d in edges if s != d)
    try:
        cycle = nx.find_cycle(g)
    except nx.NetworkXNoCycle:
        return
    path = " -> ".join([cycle[0][0], *(d for _, d in cycle)])
    raise ChainError(f"{what} would create a causal cycle: {path}")


def _trial(snap: GraphSnapshot) -> Ontology:
    onto = Ontology()
    onto.nodes, onto.kg, onto.dag = copy.deepcopy(snap.nodes), snap.kg.copy(), snap.dag.copy()
    onto.version = snap.version
    return onto


def _counts(onto: Ontology) -> dict[str, int]:
    return {"nodes": len(onto.nodes), "relations": onto.kg.number_of_edges(), "causal_edges": onto.dag.number_of_edges()}


def _finish(snap: GraphSnapshot, patch: Patch) -> Patch:
    """Dry-run the patch on a copy to prove it applies, and record before/after counts."""
    trial = _trial(snap)
    before = _counts(trial)
    try:
        trial.apply_ops(patch.ops)
    except OntologyError as e:
        raise ChainError(f"the union would not apply: {e}") from e
    patch.extra["metrics"] = {"before": before, "after": _counts(trial)}
    patch.warnings.extend(
        f"opposite causal signs on {c['src']} -> {c['dst']}; kept sign {c['kept_sign']:+d}"
        for c in patch.extra.get("conflicts", [])
    )
    return patch


def snapshot_from_dict(data: dict[str, Any]) -> GraphSnapshot:
    """Read another mirror's ontology (the JSON `Ontology.snapshot()` writes)."""
    onto = Ontology(data.get("name", "mirror"))
    for n in data.get("nodes", []):
        n = dict(n)
        nid = n.pop("id")
        n.setdefault("label", nid)
        n.setdefault("type", "concept")
        n.setdefault("attrs", {})
        onto.nodes[nid] = n
        onto.kg.add_node(nid)
        onto.dag.add_node(nid)
    for r in data.get("relations", []):
        r = dict(r)
        s, rel, d = r.pop("src"), r.pop("rel"), r.pop("dst")
        if s not in onto.nodes or d not in onto.nodes:
            raise ChainError(f"relation {s} -{rel}-> {d} names a node that is not in the file")
        onto.kg.add_edge(s, d, key=rel, **r)
    for e in data.get("causal_edges", []):
        e = dict(e)
        s, d = e.pop("src"), e.pop("dst")
        if s not in onto.nodes or d not in onto.nodes:
            raise ChainError(f"causal edge {s} -> {d} names a node that is not in the file")
        onto.dag.add_edge(s, d, **e)
    onto.version = data.get("version", 0)
    return onto.graph_snapshot()


def load_snapshot(path: Path) -> tuple[str, GraphSnapshot]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("name") or path.stem, snapshot_from_dict(data)


# ---------- HTTP and command line ----------


class MergeReq(BaseModel):
    survivor: str
    absorbed: list[str]
    label: str | None = None
    on_conflict: OnConflict = "refuse"
    rationale: str = ""


class ChainReq(BaseModel):
    ontology: dict[str, Any]
    namespace: str | None = None
    align: dict[str, str] = Field(default_factory=dict)
    separate: list[str] = Field(default_factory=list)
    on_conflict: OnConflict = "refuse"


def register(app, steward) -> None:
    """Add the Chainer's routes. Each one only queues a patch for review."""
    from fastapi import HTTPException

    def queue(patch: Patch) -> dict:
        steward.propose(patch)
        steward.save()
        return patch.model_dump()

    def guard(fn):
        try:
            return fn()
        except (ChainError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/chain/merge")
    def merge(r: MergeReq) -> dict:
        snap = steward.ontology.graph_snapshot()
        return queue(guard(lambda: plan_merge(snap, r.survivor, r.absorbed, r.label, r.on_conflict, r.rationale)))

    @app.post("/api/chain/align")
    def align(r: ChainReq) -> dict:
        return guard(lambda: suggest_alignment(steward.ontology.graph_snapshot(), snapshot_from_dict(r.ontology)))

    @app.post("/api/chain/import")
    def chain(r: ChainReq) -> dict:
        snap = steward.ontology.graph_snapshot()
        name = r.namespace or r.ontology.get("name") or ""
        return queue(guard(lambda: plan_chain(snap, snapshot_from_dict(r.ontology), name, r.align, r.separate,
                                              r.on_conflict)))


def add_cli(sub) -> None:
    c = sub.add_parser("chain", help="propose merges or chain in another ontology (queued for approval)")
    csub = c.add_subparsers(dest="chain_cmd", required=True)
    m = csub.add_parser("merge", help="merge nodes into a survivor in both graphs")
    m.add_argument("survivor")
    m.add_argument("absorbed", nargs="+")
    m.add_argument("--label")
    m.add_argument("--on-conflict", default="refuse", choices=["refuse", "keep_existing", "keep_incoming"])
    a = csub.add_parser("align", help="show which nodes of another ontology.json match this one")
    a.add_argument("file", type=Path)
    i = csub.add_parser("import", help="chain another ontology.json into this one")
    i.add_argument("file", type=Path)
    i.add_argument("--as", dest="namespace", help="name for the chained ontology (default: its own name)")
    i.add_argument("--align", nargs="*", default=[], metavar="OTHER=HERE", help="extra alignments")
    i.add_argument("--separate", nargs="*", default=[], metavar="OTHER", help="never align these ids")
    i.add_argument("--on-conflict", default="refuse", choices=["refuse", "keep_existing", "keep_incoming"])


def run_cli(args, steward) -> None:
    snap = steward.ontology.graph_snapshot()
    if args.chain_cmd == "align":
        print(json.dumps(suggest_alignment(snap, load_snapshot(args.file)[1]), indent=2))
        return
    if args.chain_cmd == "merge":
        patch = plan_merge(snap, args.survivor, args.absorbed, args.label, args.on_conflict)
    else:
        name, other = load_snapshot(args.file)
        align = dict(pair.split("=", 1) for pair in args.align)
        patch = plan_chain(snap, other, args.namespace or name, align, args.separate, args.on_conflict)
    steward.propose(patch)
    steward.save()
    m = patch.extra["metrics"]
    print(f"queued {patch.id}: {patch.rationale}")
    print(f"  nodes {m['before']['nodes']} -> {m['after']['nodes']}, relations {m['before']['relations']} -> "
          f"{m['after']['relations']}, causal edges {m['before']['causal_edges']} -> {m['after']['causal_edges']}")
    for w in patch.warnings:
        print(f"  warning: {w}")
    print(f"approve with: python -m androidmirror approve {patch.id} --as \"Your Name\"")
