"""The two linked ontologies.

- Descriptive ontology: a knowledge graph of nodes and labelled relations
  (who, what, which values and memories, and how they relate).
- Causal ontology: a DAG over the same node ids, where each edge says one node
  drives another, with a sign (+1 / -1), a strength and a confidence.

A node id is shared by both graphs, so something learned descriptively can be
reasoned about causally. All changes go through `apply_ops`, which is atomic
and returns the inverse ops so any change can be rolled back.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from .patches import Op


class OntologyError(ValueError):
    pass


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "node"


@dataclass(frozen=True)
class GraphSnapshot:
    """Copies of both graphs at one version, for engines that read the ontology."""

    version: int
    nodes: dict[str, dict[str, Any]]
    kg: nx.MultiDiGraph
    dag: nx.DiGraph


class Ontology:
    def __init__(self, name: str = "mirror") -> None:
        self.name = name
        self.version = 0
        self.nodes: dict[str, dict[str, Any]] = {}
        self.kg = nx.MultiDiGraph()  # descriptive: key = relation name
        self.dag = nx.DiGraph()  # causal
        self.log: list[str] = []  # ids of applied patches, in order

    # ---------- reading ----------

    def find_node(self, label: str) -> str | None:
        """Return the id of a node whose label matches (case-insensitive), if any."""
        want = slugify(label)
        if want in self.nodes:
            return want
        for nid, node in self.nodes.items():
            if slugify(node.get("label", "")) == want:
                return nid
        return None

    def snapshot(self) -> dict[str, Any]:
        """A read-only, JSON-ready copy of the whole ontology with its version."""
        return {
            "name": self.name,
            "version": self.version,
            "nodes": [{"id": nid, **copy.deepcopy(n)} for nid, n in self.nodes.items()],
            "relations": [
                {"src": s, "rel": k, "dst": d, **copy.deepcopy(a)}
                for s, d, k, a in self.kg.edges(keys=True, data=True)
            ],
            "causal_edges": [
                {"src": s, "dst": d, **copy.deepcopy(a)} for s, d, a in self.dag.edges(data=True)
            ],
            "log": list(self.log),
        }

    def graph_snapshot(self) -> GraphSnapshot:
        """Read-only copies of the node table and both graphs, with the version."""
        return GraphSnapshot(self.version, copy.deepcopy(self.nodes), self.kg.copy(), self.dag.copy())

    def summary(self, limit: int = 200) -> str:
        """Compact text listing of node ids and labels, for prompts."""
        lines = [f"{nid}: {n.get('label')} ({n.get('type', 'concept')})" for nid, n in self.nodes.items()]
        return "\n".join(lines[:limit]) or "(empty)"

    # ---------- writing ----------

    def apply_ops(self, ops: list[Op], patch_id: str | None = None, provenance: dict | None = None) -> list[Op]:
        """Apply ops atomically. Returns the inverse ops (already in undo order).

        `provenance` (e.g. patch id, locator, quote) is appended to every node and
        edge the ops create or change. Each touched node records the version that
        last changed it in its `modified` field.
        """
        saved = (copy.deepcopy(self.nodes), self.kg.copy(), self.dag.copy())
        inverse: list[Op] = []
        self._prov = provenance
        try:
            for op in ops:
                inverse.extend(self._apply_one(op))
                for nid in (op.id, op.src, op.dst):
                    if nid in self.nodes:
                        self.nodes[nid]["modified"] = self.version + 1
        except Exception:
            self.nodes, self.kg, self.dag = saved
            raise
        self.version += 1
        if patch_id:
            self.log.append(patch_id)
        return list(reversed(inverse))

    def check_ops(self, ops: list[Op]) -> None:
        """Raise OntologyError if the ops would not apply, without changing anything."""
        trial = Ontology(self.name)
        trial.nodes = copy.deepcopy(self.nodes)
        trial.kg, trial.dag = self.kg.copy(), self.dag.copy()
        trial.apply_ops(ops)

    def _need(self, nid: str | None) -> str:
        if not nid or nid not in self.nodes:
            raise OntologyError(f"unknown node: {nid!r}")
        return nid

    def _apply_one(self, op: Op) -> list[Op]:
        if op.op == "add_node":
            nid = op.id or slugify(op.label or "")
            if nid in self.nodes:
                raise OntologyError(f"node already exists: {nid!r}")
            attrs = {k: v for k, v in op.attrs.items() if k != "provenance"}
            self.nodes[nid] = {"label": op.label or nid, "type": op.type or "concept", "attrs": attrs,
                               "provenance": list(op.attrs.get("provenance", []))}
            self._stamp(self.nodes[nid])
            self.kg.add_node(nid)
            self.dag.add_node(nid)
            return [Op(op="remove_node", id=nid)]

        if op.op == "update_node":
            nid = self._need(op.id)
            node = self.nodes[nid]
            undo = Op(op="update_node", id=nid, label=node["label"], type=node["type"])
            self._stamp(node)
            if op.label:
                node["label"] = op.label
            if op.type:
                node["type"] = op.type
            for k, v in op.attrs.items():
                undo.attrs[k] = node["attrs"].get(k)
                if v is None:
                    node["attrs"].pop(k, None)
                else:
                    node["attrs"][k] = v
            return [undo]

        if op.op == "remove_node":
            nid = self._need(op.id)
            node = self.nodes.pop(nid)
            undo = [Op(op="add_node", id=nid, label=node["label"], type=node["type"],
                       attrs={**node["attrs"], "provenance": node.get("provenance", [])})]
            for s, d, k, a in list(self.kg.in_edges(nid, keys=True, data=True)) + list(
                self.kg.out_edges(nid, keys=True, data=True)
            ):
                undo.append(Op(op="add_relation", src=s, rel=k, dst=d, attrs=a))
            for s, d, a in list(self.dag.in_edges(nid, data=True)) + list(self.dag.out_edges(nid, data=True)):
                undo.append(_causal_op("add_causal_edge", s, d, a))
            self.kg.remove_node(nid)
            self.dag.remove_node(nid)
            return list(reversed(undo))  # reversed again by apply_ops: node first, then edges

        if op.op == "add_relation":
            s, d = self._need(op.src), self._need(op.dst)
            rel = op.rel or "related_to"
            if self.kg.has_edge(s, d, key=rel):
                raise OntologyError(f"relation exists: {s} -{rel}-> {d}")
            self.kg.add_edge(s, d, key=rel, **op.attrs)
            self._stamp(self.kg.edges[s, d, rel])
            return [Op(op="remove_relation", src=s, rel=rel, dst=d)]

        if op.op == "remove_relation":
            s, d, rel = op.src, op.dst, op.rel or "related_to"
            if not self.kg.has_edge(s, d, key=rel):
                raise OntologyError(f"no such relation: {s} -{rel}-> {d}")
            attrs = dict(self.kg.edges[s, d, rel])
            self.kg.remove_edge(s, d, key=rel)
            return [Op(op="add_relation", src=s, rel=rel, dst=d, attrs=attrs)]

        if op.op == "add_causal_edge":
            s, d = self._need(op.src), self._need(op.dst)
            if s == d:
                raise OntologyError("a node cannot cause itself")
            old = dict(self.dag.edges[s, d]) if self.dag.has_edge(s, d) else None
            self.dag.add_edge(
                s,
                d,
                sign=1 if (op.sign or 1) >= 0 else -1,
                strength=_clamp(op.strength, 0.5),
                confidence=_clamp(op.confidence, 0.5),
                mechanism=op.mechanism or "",
                provenance=list(op.attrs.get("provenance", old.get("provenance", []) if old else [])),
            )
            self._stamp(self.dag.edges[s, d])
            if not nx.is_directed_acyclic_graph(self.dag):
                raise OntologyError(f"causal edge {s} -> {d} would create a cycle")
            if old is None:
                return [Op(op="remove_causal_edge", src=s, dst=d)]
            return [_causal_op("add_causal_edge", s, d, old)]

        if op.op == "remove_causal_edge":
            s, d = op.src, op.dst
            if not self.dag.has_edge(s, d):
                raise OntologyError(f"no such causal edge: {s} -> {d}")
            attrs = dict(self.dag.edges[s, d])
            self.dag.remove_edge(s, d)
            return [_causal_op("add_causal_edge", s, d, attrs)]

        raise OntologyError(f"unsupported op: {op.op}")

    def _stamp(self, record: dict[str, Any]) -> None:
        prov = getattr(self, "_prov", None)
        if prov:
            record.setdefault("provenance", [])
            if prov not in record["provenance"]:
                record["provenance"].append(prov)

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Ontology":
        onto = cls()
        if not path.exists():
            return onto
        data = json.loads(path.read_text(encoding="utf-8"))
        onto.name = data.get("name", "mirror")
        for n in data["nodes"]:
            nid = n.pop("id")
            onto.nodes[nid] = n
            onto.kg.add_node(nid)
            onto.dag.add_node(nid)
        for r in data["relations"]:
            s, rel, d = r.pop("src"), r.pop("rel"), r.pop("dst")
            onto.kg.add_edge(s, d, key=rel, **r)
        for e in data["causal_edges"]:
            onto.dag.add_edge(e.pop("src"), e.pop("dst"), **e)
        onto.version = data.get("version", 0)
        onto.log = data.get("log", [])
        return onto


def _clamp(x: float | None, default: float) -> float:
    return default if x is None else max(0.0, min(1.0, float(x)))


def _causal_op(kind: str, s: str, d: str, a: dict[str, Any]) -> Op:
    return Op(
        op=kind,
        src=s,
        dst=d,
        sign=a.get("sign", 1),
        strength=a.get("strength"),
        confidence=a.get("confidence"),
        mechanism=a.get("mechanism", ""),
        attrs={"provenance": list(a.get("provenance", []))},
    )
