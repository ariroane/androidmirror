"""A built Mirror, loaded for reading: the views the web app and the Oracle use."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

from ..ontology import GraphSnapshot
from . import situations as sit
from .causal import effects
from .causal.pdag import PDAG
from .kg import KnowledgeBase, fact_key
from .schema import EVALUATIONS, SELF, archetype_type


class Mirror:
    def __init__(self, data: dict[str, Any], passages: list[dict[str, Any]]) -> None:
        self.meta = data["meta"]
        self.nodes: dict[str, dict[str, Any]] = data["nodes"]
        self.facts: list[dict[str, Any]] = data["facts"]
        self.claims: list[dict[str, Any]] = data["claims"]
        self.situations: list[dict[str, Any]] = data.get("situations", [])
        self.by_situation = {x["id"]: x for x in self.situations}
        self.stated_model: list[dict[str, Any]] = data.get("stated_model", [])
        self.typical: dict[str, Any] = data.get("typical", {})
        self.transformation: dict[str, Any] = data.get("transformation", {})
        self.causal: dict[str, Any] = data.get("causal", {})
        self.passage_ids: list[str] = data["passage_ids"]
        self.passages = {p["id"]: p for p in passages}
        self.kb = KnowledgeBase(self.nodes, self.facts)
        self.cpdag = PDAG.from_dict(self.causal["cpdag"]) if "cpdag" in self.causal else PDAG([])
        self.model = PDAG.from_dict(self.causal["model"]) if "model" in self.causal else PDAG([])
        self.variables: list[str] = self.causal.get("variables", [])
        self.view: str = self.causal.get("view", sit.DEFAULT_VIEW)
        vig = self.causal.get("vignettes") or {}
        self.vmodel = PDAG.from_dict(vig["model"]) if "model" in vig else None

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
            sel = {r["id"]: r for r in self.causal.get("selection", []) if r["status"] == "selected"}
            nodes = [self._node_view(v, support=sel[v]["support"]) for v in self.variables]
            edges = [{"id": f"{e['a']}|{e['b']}", "source": e["a"], "target": e["b"], "directed": e["directed"],
                      "status": e["status"], "effect": e["effect"], "origin": "corpus",
                      "stability": e["stability"].get("adjacent")} for e in self.causal.get("edges", [])]
            vig = self.causal.get("vignettes")
            if vig:  # what the vignettes change, drawn in their own colour (the app can hide it)
                by_pair = {frozenset((e["source"], e["target"])): e for e in edges}
                for d in vig["diff"]:
                    if d["change"] == "same":
                        continue
                    key = frozenset((d["a"], d["b"]))
                    wv = d["with_vignettes"]
                    src, dst = wv if isinstance(wv, list) else [d["a"], d["b"]]
                    if d["change"] == "added by vignettes":
                        edges.append({"id": f"{d['a']}|{d['b']}", "source": src, "target": dst, "directed": isinstance(wv, list),
                                      "status": "vignette", "effect": d["effect_with_vignettes"], "origin": "vignette",
                                      "change": d["change"], "stability": None})
                    elif key in by_pair:
                        e = by_pair[key]
                        e["change"] = d["change"]
                        e["vignette_direction"] = wv if isinstance(wv, list) else None
            return {"layer": layer, "nodes": nodes, "edges": edges, "view": self.view,
                    "vignettes": bool(vig)}
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
        return {"id": nid, "label": n.get("label", nid), "kind": n.get("kind", "concept"),
                "evaluation": self.evaluation(nid), "support": support, "variable": nid in self.variables}

    # ---------- a node and its sources ----------

    def node(self, nid: str) -> dict[str, Any]:
        if nid not in self.nodes:
            raise KeyError(nid)
        n = self.nodes[nid]
        stated = [f for f in self.facts if nid in (f["s"], f["o"])]
        inferred = [x for x in self.kb.query(s=nid) + self.kb.query(o=nid) if x["status"] == "inferred"]
        sel = next((r for r in self.causal.get("selection", []) if r["id"] == nid), None)
        return {
            "id": nid, "label": n["label"], "kind": n["kind"], "gloss": n.get("gloss", ""),
            "aliases": n.get("aliases", []), "pole_of": n.get("pole_of"), "evaluation": self.evaluation(nid),
            "classes": sorted(o for (s, r, o) in self.kb.closure if s == nid and r == "is_a" and o not in EVALUATIONS),
            "mentions": n.get("mentions", []), "support": n.get("support", 0),
            "facts": [self._fact_view(f) for f in sorted(stated, key=lambda f: -f["support"])],
            "inferred": [{**x, "s_label": self.label(x["s"]), "o_label": self.label(x["o"])} for x in inferred[:40]],
            "causal": {
                "selection": sel,
                "edges": [e for e in self.causal.get("edges", []) if nid in (e["a"], e["b"])],
                "claims": [self._claim_view(c) for c in self.claims if nid in (c["cause"], c["effect"])],
            },
            "violations": [v for v in self.kb.violations if nid in json.dumps(v)],
            "situations": self.situations_of(nid), "vignettes": self.situations_of(nid, "vignette", 20),
            "typical": self.typical.get(nid),
        }

    def _fact_view(self, f: dict[str, Any]) -> dict[str, Any]:
        return {"s": f["s"], "r": f["r"], "o": f["o"], "neg": f["neg"], "held": f["held"], "support": f["support"],
                "s_label": self.label(f["s"]), "o_label": self.label(f["o"]), "modalities": f.get("modalities", []),
                "qualifiers": f.get("qualifiers", []), "evidence": f["evidence"][:8]}

    def _claim_view(self, c: dict[str, Any]) -> dict[str, Any]:
        return {**{k: c[k] for k in ("cause", "effect", "sign", "support", "held", "bases", "sign_conflict")},
                "cause_label": self.label(c["cause"]), "effect_label": self.label(c["effect"]),
                "evidence": c["evidence"][:8]}

    def passage(self, pid: str) -> dict[str, Any]:
        if pid not in self.passages:
            raise KeyError(pid)
        p = self.passages[pid]
        return {**p, "situations": [self.situation_view(x, brief=True) for x in self.situations if x.get("passage") == pid],
                "facts": [self._fact_view(f) for f in self.facts if any(e["passage"] == pid for e in f["evidence"])],
                "claims": [self._claim_view(c) for c in self.claims if any(e["passage"] == pid for e in c["evidence"])]}

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
        keys = ("settings", "selection", "variables", "n_rows", "class_size", "pins", "feedback_claims", "conflicts",
                "n_tests", "depth_reached", "empty", "view", "views", "independences", "untestable", "coverage", "ambiguous")
        out = {k: c.get(k) for k in keys}
        vig = c.get("vignettes")
        out["vignettes"] = {k: vig.get(k) for k in ("targets", "n_vignettes", "class_size", "diff", "responses")} if vig else None
        tiers = Counter(cell["tier"] for x in self.situations if x["origin"] == "corpus"
                        for v, cell in x["cells"].items() if v in self.variables)
        out["tiers"] = dict(tiers)
        out["situations"] = sum(1 for x in self.situations if x["origin"] == "corpus")
        out["transformation"] = self.transformation
        out["typical"] = self.typical
        out["edges"] = [{**e, "a_label": self.label(e["a"]), "b_label": self.label(e["b"])} for e in c.get("edges", [])]
        out["removed"] = c.get("removed", [])[:200]
        out["labels"] = {v: self.label(v) for v in self.variables}
        return out

    def edge(self, a: str, b: str) -> dict[str, Any]:
        e = next((x for x in self.causal.get("edges", []) if {x["a"], x["b"]} == {a, b}), None)
        d = next((d for d in (self.causal.get("vignettes") or {}).get("diff", []) if {d["a"], d["b"]} == {a, b}), None)
        if e is None and d is None:
            raise KeyError(f"{a}|{b}")
        if e is None:  # an edge only the vignettes find: no corpus evidence to show
            e = {"a": a, "b": b, "directed": False, "status": None, "why": None, "member_direction": [a, b], "effect": None,
                 "stability": {}, "weakest_test": None}
        tiers = sit.VIEWS[self.view]
        both = [x for x in self.situations if all((x["cells"].get(v) or {}).get("tier") in tiers for v in (a, b))]
        corpus = [x for x in both if x["origin"] == "corpus"]
        vig = [x for x in both if x["origin"] == "vignette"]
        return {**e, "a_label": self.label(a), "b_label": self.label(b), "chain": self._why_chain(e),
                "together": [self.situation_view(x, brief=True, only=(a, b)) for x in corpus[:40]], "n_together": len(corpus),
                "vignettes": [self.situation_view(x, brief=True, only=(a, b)) for x in vig[:20]], "n_vignettes": len(vig),
                "vignette_change": d,
                "stated": [self._claim_view(c) for c in self.claims if {c["cause"], c["effect"]} == {a, b}]}

    def _why_chain(self, e: dict[str, Any]) -> list[dict[str, Any]]:
        """Walk an arrow's provenance back to the v-structures and pins it rests on."""
        if not e["directed"]:
            return []
        out, todo, seen = [], [(e["a"], e["b"])], set()
        while todo and len(out) < 12:
            edge = todo.pop(0)
            if edge in seen:
                continue
            seen.add(edge)
            why = self.model.why.get(edge)
            if not why:
                continue
            out.append({"edge": list(edge), **{k: v for k, v in why.items() if k != "pin"}})
            todo += [tuple(v) for v in why.get("via", [])]
        return out

    def effect(self, x: str, y: str) -> dict[str, Any]:
        data, _ = sit.table(self.situations, self.variables, self.view)
        out = {"data_only": effects(data, self.cpdag, x, y), "with_pins": effects(data, self.model, x, y),
               "x_label": self.label(x), "y_label": self.label(y)}
        if self.vmodel is not None:
            vdata, _ = sit.table(self.situations, self.variables, self.view, origins=("corpus", "vignette"))
            out["with_vignettes"] = effects(vdata, self.vmodel, x, y)
        return out

    # ---------- situations and the subject transformation ----------

    def situation_view(self, x: dict[str, Any], brief: bool = False, only: tuple[str, ...] | None = None) -> dict[str, Any]:
        cells = [{"id": v, "label": self.label(v), "variable": v in self.variables, **c}
                 for v, c in x["cells"].items() if only is None or v in only]
        order = {t: i for i, t in enumerate(sit.TIERS)}
        cells.sort(key=lambda c: (order.get(c["tier"], 9), not c["variable"], c["label"]))
        if brief:
            cells = [c for c in cells if c["tier"] != "defaulted"] if only is None else cells
        return {k: x.get(k) for k in ("id", "passage", "locator", "origin", "kind", "description", "contrast", "target",
                                      "based_on", "tensions")} | {"cells": cells}

    def profile(self) -> dict[str, Any]:
        """What the subject transformation knows about the subject, and how well it did on held-out situations."""
        from .subject import build_profile

        corpus = [x for x in self.situations if x["origin"] == "corpus"]
        prof = build_profile(self.label(SELF), self.nodes, self.facts, self.kb, self.claims, corpus, self.passages,
                             self.variables)
        return {"profile": prof, **self.transformation}

    def situation(self, sid: str) -> dict[str, Any]:
        if sid not in self.by_situation:
            raise KeyError(sid)
        return self.situation_view(self.by_situation[sid])

    def situations_of(self, nid: str, origin: str = "corpus", limit: int = 60) -> list[dict[str, Any]]:
        hits = [x for x in self.situations if x["origin"] == origin and nid in x["cells"]
                and x["cells"][nid]["tier"] != "defaulted"]
        return [self.situation_view(x, brief=True) for x in hits[:limit]]

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
        eff = {tuple(e["member_direction"]): e for e in self.causal.get("edges", [])}
        for a, b in self.causal.get("dag", []):
            e = eff.get((a, b), {})
            effect = e.get("effect") or 0.0
            dag.add_edge(a, b, sign=1 if effect >= 0 else -1, strength=min(1.0, abs(effect) * 2),
                         confidence=e.get("stability", {}).get("adjacent", 0.5),
                         provenance=[{"quote": s["evidence"][0]["quote"]} for s in e.get("stated", []) if s.get("evidence")])
        return GraphSnapshot(0, nodes, kg, dag)
