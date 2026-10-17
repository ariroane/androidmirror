"""A built Mirror, loaded for reading: the views the web app and the Oracle use."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

from ..ontology import GraphSnapshot
from .causal import explain, graph_of
from .kg import KnowledgeBase, fact_key
from .schema import EVALUATIONS, SELF, archetype_type


class Mirror:
    def __init__(self, data: dict[str, Any], passages: list[dict[str, Any]]) -> None:
        self.meta = data["meta"]
        self.nodes: dict[str, dict[str, Any]] = data["nodes"]
        self.facts: list[dict[str, Any]] = data["facts"]
        self.judgments: list[dict[str, Any]] = data.get("judgments", [])
        self.unresolved: list[dict[str, Any]] = data.get("unresolved", [])
        self.values: list[dict[str, Any]] = data.get("values", [])
        self.value_of = {v["concept"]: v for v in self.values}
        self.causal: dict[str, Any] = data.get("causal", {})
        self.passage_ids: list[str] = data["passage_ids"]
        self.passages = {p["id"]: p for p in passages}
        self.kb = KnowledgeBase(self.nodes, self.facts)
        self.graph_ = graph_of(self.causal)
        self.constraint = {c["id"]: c for c in self.causal.get("constraints", [])}
        self.variables: list[str] = self.causal.get("nodes", [])

    @classmethod
    def load(cls, root: Path) -> "Mirror | None":
        path = root / "mirror.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        passages = json.loads((root / "passages.json").read_text(encoding="utf-8"))
        return cls(data, passages)

    def label(self, nid: str) -> str:
        return self.nodes.get(nid, {}).get("label", nid)

    def evaluation(self, nid: str) -> str | None:
        evs = [e for e in EVALUATIONS if (nid, "is_a", e) in self.kb.closure]
        return evs[0] if len(evs) == 1 else ("conflicted" if evs else None)

    # ---------- the canvas ----------

    def graph(self, layer: str = "kg", limit: int = 120) -> dict[str, Any]:
        if layer == "causal":
            weight: Counter = Counter()
            for e in self.causal.get("edges", []):
                weight[e["src"]] += e["weight"]
                weight[e["dst"]] += e["weight"]
            nodes = [self._node_view(v, support=weight[v]) for v in self.variables]
            edges = [{"id": f"{e['src']}|{e['dst']}", "source": e["src"], "target": e["dst"], "sign": e["sign"],
                      "weight": e["weight"], "direct": e["direct"], "open": bool(e["open"]), "kinds": e["kinds"],
                      "status": "open" if e["open"] else "direct" if e["direct"] else "stated"}
                     for e in self.causal.get("edges", [])]
            return {"layer": layer, "nodes": nodes, "edges": edges}
        held = [f for f in self.facts if f["held"] and not f["neg"] and f["o"] not in EVALUATIONS]
        degree: dict[str, int] = {}
        for f in held:
            for n in (f["s"], f["o"]):
                degree[n] = degree.get(n, 0) + f["support"]
        ranked = sorted(degree, key=lambda n: (n != SELF, -(degree[n] + self.nodes.get(n, {}).get("support", 0))))
        keep = set(ranked[:limit])
        nodes = [self._node_view(n, support=degree[n]) for n in ranked[:limit]]
        edges = [{"id": fact_key(f["s"], f["r"], f["o"]), "source": f["s"], "target": f["o"], "rel": f["r"],
                  "support": f["support"], "directed": True}
                 for f in held if f["s"] in keep and f["o"] in keep]
        return {"layer": layer, "nodes": nodes, "edges": edges, "hidden": max(0, len(degree) - limit)}

    def _node_view(self, nid: str, support: int = 0) -> dict[str, Any]:
        n = self.nodes.get(nid, {})
        v = self.value_of.get(nid)
        return {"id": nid, "label": n.get("label", nid), "kind": n.get("kind", "concept"),
                "evaluation": self.evaluation(nid), "support": support, "variable": nid in self.variables,
                "value": v["direction"] if v else 0}

    # ---------- a node and its sources ----------

    def node(self, nid: str) -> dict[str, Any]:
        if nid not in self.nodes:
            raise KeyError(nid)
        n = self.nodes[nid]
        stated = [f for f in self.facts if nid in (f["s"], f["o"])]
        inferred = [x for x in self.kb.query(s=nid) + self.kb.query(o=nid) if x["status"] == "inferred"]
        return {
            "id": nid, "label": n["label"], "kind": n["kind"], "gloss": n.get("gloss", ""),
            "aliases": n.get("aliases", []), "pole_of": n.get("pole_of"), "evaluation": self.evaluation(nid),
            "classes": sorted(o for (s, r, o) in self.kb.closure if s == nid and r == "is_a" and o not in EVALUATIONS),
            "mentions": n.get("mentions", []), "support": n.get("support", 0),
            "facts": [self._fact_view(f) for f in sorted(stated, key=lambda f: -f["support"])],
            "inferred": [{**x, "s_label": self.label(x["s"]), "o_label": self.label(x["o"])} for x in inferred[:40]],
            "causal": {
                "in_model": nid in self.variables,
                "edges": [self._edge_view(e) for e in self.causal.get("edges", []) if nid in (e["src"], e["dst"])],
                "constraints": [self._constraint_view(c) for c in self.causal.get("constraints", [])
                                if nid in (c["x"], c["y"], c["z"])],
            },
            "value": self.value_of.get(nid),
            "violations": [v for v in self.kb.violations if nid in json.dumps(v)],
        }

    def _fact_view(self, f: dict[str, Any]) -> dict[str, Any]:
        return {"s": f["s"], "r": f["r"], "o": f["o"], "neg": f["neg"], "held": f["held"], "support": f["support"],
                "s_label": self.label(f["s"]), "o_label": self.label(f["o"]), "modalities": f.get("modalities", []),
                "qualifiers": f.get("qualifiers", []), "evidence": f["evidence"][:8]}

    def _constraint_view(self, c: dict[str, Any]) -> dict[str, Any]:
        return {**c, "x_label": self.label(c["x"]), "y_label": self.label(c["y"]),
                "z_label": self.label(c["z"]) if c["z"] else "", "text": self.say(c)}

    def say(self, c: dict[str, Any]) -> str:
        """A constraint in words."""
        x, y, z = self.label(c["x"]), self.label(c["y"]), self.label(c["z"]) if c["z"] else ""
        if c["type"] == "reaches":
            return f"{x} {'raises' if c['sign'] > 0 else 'lowers'} {y}" + (" (directly)" if c.get("direct") else "")
        if c["type"] == "no_path":
            return f"{x} makes no difference to {y}"
        return f"{x} acts on {z} only through {y}"

    def _edge_view(self, e: dict[str, Any]) -> dict[str, Any]:
        return {**e, "src_label": self.label(e["src"]), "dst_label": self.label(e["dst"]),
                "open": {"through": e["open"]["through"], "labels": [self.label(n) for n in e["open"]["through"]]} if e["open"] else None}

    def passage(self, pid: str) -> dict[str, Any]:
        if pid not in self.passages:
            raise KeyError(pid)
        p = self.passages[pid]
        lab = self.label
        return {**p, "facts": [self._fact_view(f) for f in self.facts if any(e["passage"] == pid for e in f["evidence"])],
                "judgments": [{**j, "x_label": lab(j["x"]), "y_label": lab(j["y"]), "z_label": lab(j["z"]) if j["z"] else ""}
                              for j in self.judgments if j["passage"] == pid],
                "unresolved": [j for j in self.unresolved if j["passage"] == pid],
                "values": [{**v, "label": lab(v["concept"]), "direction": d["direction"]}
                           for v in self.values for d in v["evidence"] if d["passage"] == pid]}

    # ---------- knowledge ----------

    def knowledge(self) -> dict[str, Any]:
        def view(v: dict[str, Any]) -> dict[str, Any]:
            out = dict(v)
            proofs = []
            for k in v.get("facts", []):
                s, r, o, *neg = k.split("|")
                if neg:
                    f = next((f for f in self.facts if fact_key(f["s"], f["r"], f["o"], True) == k), None)
                    proofs.append({"fact": k, "s": s, "r": r, "o": o, "rule": "denied",
                                   "evidence": f["evidence"][:3] if f else []})
                else:
                    proofs.append(self.kb.explain((s, r, o)))
            out["proofs"] = proofs
            return out

        return {"stats": self.kb.stats(), "violations": [view(v) for v in self.kb.violations[:100]],
                "relations": sorted({f["r"] for f in self.facts})}

    def holds(self, s: str, r: str, o: str) -> dict[str, Any]:
        return self.kb.holds(s, r, o)

    # ---------- causal ----------

    def causal_view(self) -> dict[str, Any]:
        c = self.causal
        labels = {n: self.label(n) for n in c.get("nodes", []) + c.get("isolated", [])}
        for x in c.get("constraints", []):
            for k in ("x", "y", "z"):
                if x[k]:
                    labels[x[k]] = self.label(x[k])
        return {"settings": c.get("settings"), "stats": c.get("stats"), "holdout": c.get("holdout"),
                "edges": [self._edge_view(e) for e in c.get("edges", [])],
                "constraints": [self._constraint_view(x) for x in c.get("constraints", [])],
                "tensions": c.get("tensions", []), "feedback": c.get("feedback", []), "isolated": c.get("isolated", []),
                "values": [{**v, "label": self.label(v["concept"])} for v in self.values],
                "unresolved": self.unresolved[:200], "n_unresolved": len(self.unresolved), "labels": labels}

    def edge(self, a: str, b: str) -> dict[str, Any]:
        e = next((x for x in self.causal.get("edges", []) if (x["src"], x["dst"]) == (a, b)), None)
        if e is None:
            raise KeyError(f"{a}|{b}")
        own = self.constraint.get(e["constraint"])
        supports = [self._constraint_view(self.constraint[i]) for i in e["supports"] if i in self.constraint]
        related = [self._constraint_view(c) for c in self.causal.get("constraints", [])
                   if e["constraint"] in (c.get("conflicts") or [])]
        return {**self._edge_view(e), "own": self._constraint_view(own) if own else None, "supports": supports,
                "challenged_by": related}

    def explain(self, x: str, y: str) -> dict[str, Any]:
        out = explain({**self.causal, "_graph": self.graph_}, x, y)
        out["x_label"], out["y_label"] = self.label(x), self.label(y)
        for p in out["paths"]:
            p["labels"] = [self.label(n) for n in p["nodes"]]
        out["claims"] = [self._constraint_view(c) for c in out["claims"]]
        if out["denial"]:
            out["denial"] = self._constraint_view(out["denial"])
        return out

    # ---------- for the archetype face ----------

    def snapshot(self) -> GraphSnapshot:
        nodes = {}
        for nid, n in self.nodes.items():
            if nid in EVALUATIONS:
                continue
            ev = self.evaluation(nid)
            nodes[nid] = {"label": n["label"], "type": archetype_type(n["kind"], ev if ev != "conflicted" else None),
                          "provenance": [{"quote": m["quote"], "locator": m["locator"]} for m in n.get("mentions", [])[:2]]}
        kg = nx.MultiDiGraph()
        kg.add_nodes_from(nodes)
        for f in self.facts:
            if f["held"] and not f["neg"] and f["s"] in nodes and f["o"] in nodes:
                kg.add_edge(f["s"], f["o"], key=f["r"], provenance=[{"quote": e["quote"]} for e in f["evidence"][:2]])
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        for e in self.causal.get("edges", []):
            if e["src"] in nodes and e["dst"] in nodes:
                ev = (self.constraint.get(e["constraint"]) or {}).get("evidence", [])
                dag.add_edge(e["src"], e["dst"], sign=e["sign"], strength=min(1.0, e["weight"] / 5),
                             confidence=0.5 if e["open"] else 1.0, provenance=[{"quote": x["quote"]} for x in ev[:2]])
        return GraphSnapshot(0, nodes, kg, dag)
