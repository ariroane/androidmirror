"""Match the Mirror to an archetype (by default the 22 Major Arcana), for the Face tab.

Each card in `data/archetypes_major_arcana.json` is a few operative patterns
(motifs) over the Mirror's own vocabulary: node types grouped into classes,
relation names grouped into families, and signed causal edges. `match` counts
every motif in a graph snapshot and compares the count with a null model that
shuffles node types across nodes and causal edge attributes (sign, strength,
confidence) across edges while keeping both graphs' shape. A card wins only if
its patterns occur more than that shuffle would produce; the top-scoring card is
shown even when a runner-up is close.

Read-only: this module takes a `GraphSnapshot` and never proposes or applies
anything.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx

from .ontology import GraphSnapshot

DEFAULT_SET = Path(__file__).resolve().parent.parent / "data" / "archetypes_major_arcana.json"

Instance = tuple[str, ...]
PREDICATES = {"sun", "overlap", "reach"}  # global motifs that are true or false, not counted


@lru_cache(maxsize=4)
def load_set(path: str | Path = DEFAULT_SET) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["_family_of"] = {name: fam for fam, names in data["relation_families"].items() for name in names}
    return data


# ---------- the graph, as the matcher sees it ----------


@dataclass
class _View:
    nodes: list[str]
    typ: dict[str, str]
    cls: dict[str, str]
    kg: list[tuple[str, str, str, str]]  # src, dst, relation name, family
    attr: dict[tuple[str, str], tuple[int, float, float]]  # sign, strength, confidence
    succ: dict[str, list[str]] = field(default_factory=dict)
    pred: dict[str, list[str]] = field(default_factory=dict)
    kg_out: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)  # dst, name, family
    kg_in: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)  # shared, type- and sign-free facts

    def match(self, token: str, n: str) -> bool:
        return token == "*" or token == self.cls[n] or token == self.typ[n]

    def any_match(self, tokens: list[str], n: str) -> bool:
        return any(self.match(t, n) for t in tokens)


def _build(snap: GraphSnapshot, keep: set[str] | None, cset: dict[str, Any]) -> _View:
    nodes = sorted(n for n in snap.nodes if keep is None or n in keep)
    inside = set(nodes)
    typ = {n: snap.nodes[n].get("type", "concept") for n in nodes}
    classes = cset["node_classes"]
    family_of = cset["_family_of"]
    kg = sorted((s, d, k, family_of.get(k, "other")) for s, d, k in snap.kg.edges(keys=True) if s in inside and d in inside)
    attr = {}
    for s, d, a in snap.dag.edges(data=True):
        if s in inside and d in inside:
            attr[(s, d)] = (1 if a.get("sign", 1) >= 0 else -1, float(a.get("strength", 0.5)), float(a.get("confidence", 0.5)))
    v = _View(nodes, typ, {n: classes.get(t, "OTHER") for n, t in typ.items()}, kg, attr)
    v.succ = {n: [] for n in nodes}
    v.pred = {n: [] for n in nodes}
    v.kg_out = {n: [] for n in nodes}
    v.kg_in = {n: [] for n in nodes}
    for s, d in sorted(attr):
        v.succ[s].append(d)
        v.pred[d].append(s)
    for s, d, k, f in kg:
        v.kg_out[s].append((d, k, f))
        v.kg_in[d].append((s, k, f))
    v.structure = _structure(v, snap, keep)
    return v


def _structure(v: _View, snap: GraphSnapshot, keep: set[str] | None) -> dict[str, Any]:
    """Facts that depend only on the graphs' shape, so every null sample shares them."""
    dag = nx.DiGraph()
    dag.add_nodes_from(v.nodes)
    dag.add_edges_from(v.attr)
    combined = nx.Graph()
    combined.add_nodes_from(v.nodes)
    combined.add_edges_from(v.attr)
    combined.add_edges_from((s, d) for s, d, _, _ in v.kg)
    topo = list(nx.topological_sort(dag))
    longest_from = {n: 0 for n in v.nodes}
    for n in reversed(topo):
        longest_from[n] = max((1 + longest_from[c] for c in v.succ[n]), default=0)
    desc = {n: nx.descendants(dag, n) for n in v.nodes}
    pairs = sum(len(d) for d in desc.values())
    load = {}
    for n in v.nodes:  # share of ancestor->descendant pairs lost if n were removed
        if pairs < 6 or not (v.succ[n] or v.pred[n]):
            load[n] = 0.0
            continue
        g = dag.copy()
        g.remove_node(n)
        load[n] = (pairs - sum(len(nx.descendants(g, x)) for x in g.nodes)) / pairs
    return {
        "load": load,
        "topo": topo,
        "longest_from": longest_from,
        "descendants": desc,
        "articulation": set(nx.articulation_points(combined)) if combined.number_of_edges() else set(),
        "components": nx.number_connected_components(combined) if v.nodes else 0,
    }


def _shuffled(v: _View, rng: random.Random, cset: dict[str, Any]) -> _View:
    """Same shape; node types permuted across nodes, causal attributes permuted across edges."""
    types = [v.typ[n] for n in v.nodes]
    rng.shuffle(types)
    typ = dict(zip(v.nodes, types, strict=True))
    keys = list(v.attr)
    vals = [v.attr[k] for k in keys]
    rng.shuffle(vals)
    classes = cset["node_classes"]
    return _View(v.nodes, typ, {n: classes.get(t, "OTHER") for n, t in typ.items()}, v.kg, dict(zip(keys, vals, strict=True)),
                 v.succ, v.pred, v.kg_out, v.kg_in, v.structure)


# ---------- motifs ----------


def _flip(m: dict[str, Any]) -> dict[str, Any]:
    """The reversed reading of a motif: every causal sign negated."""
    out = dict(m)
    if "sign" in out:
        out["sign"] = -out["sign"]
    if "signs" in out:
        out["signs"] = [-s for s in out["signs"]]
    for key in ("children", "parents"):
        if key in out:
            out[key] = [{**c, "sign": -c["sign"]} for c in out[key]]
    return out


def _count_kin(v: _View, n: str, spec: dict[str, Any], kin: list[str], parent: bool) -> int:
    total = 0
    for k in kin:
        edge = (k, n) if parent else (n, k)
        if v.attr[edge][0] == spec["sign"] and v.any_match(spec["classes"], k):
            total += 1
    return total


def _node_ok(v: _View, n: str, m: dict[str, Any]) -> bool:
    if not v.any_match(m.get("classes", ["*"]), n):
        return False
    succ, pred, st = v.succ[n], v.pred[n], v.structure
    if m.get("root") and (pred or not succ):
        return False
    if m.get("sink") and (succ or not pred):
        return False
    if len(v.kg_out[n]) + len(v.kg_in[n]) < m.get("kg_degree_min", 0):
        return False
    if "dag_out_max" in m and len(succ) > m["dag_out_max"]:
        return False
    if "lacks_families" in m:
        fams = set(m["lacks_families"])
        if any(f in fams for _, _, f in v.kg_out[n] + v.kg_in[n]):
            return False
    for spec in m.get("children", []):
        if _count_kin(v, n, spec, succ, parent=False) < spec["min"]:
            return False
    for spec in m.get("parents", []):
        if _count_kin(v, n, spec, pred, parent=True) < spec["min"]:
            return False
    if len(pred) < m.get("min_parents", 0):
        return False
    signs = [v.attr[(p, n)][0] for p in pred]
    if m.get("mixed_parents") and not (1 in signs and -1 in signs):
        return False
    if "balanced_max" in m and abs(sum(v.attr[(p, n)][0] * v.attr[(p, n)][1] for p in pred)) > m["balanced_max"]:
        return False
    if "max_parent_strength" in m and any(v.attr[(p, n)][1] > m["max_parent_strength"] for p in pred):
        return False
    if m.get("only_neg_parents") and not (signs and all(s < 0 for s in signs)):
        return False
    if st["load"][n] < m.get("load_bearing_min", 0):
        return False
    if m.get("articulation") and n not in st["articulation"]:
        return False
    if st["longest_from"][n] < m.get("out_chain_min", 0):
        return False
    if "tree_descendants_min" in m:
        d = st["descendants"][n]
        if len(d) < m["tree_descendants_min"] or any(len(v.pred[x]) != 1 for x in d):
            return False
    if "kg_in" in m:
        want = m["kg_in"]
        if not any(f == want["family"] and v.any_match(want["classes"], s) for s, _, f in v.kg_in[n]):
            return False
    for spec in m.get("kg_out", []):
        if sum(1 for _, _, f in v.kg_out[n] if f == spec["family"]) < spec["min"]:
            return False
    if "all_out_conf_below" in m and not (succ and all(v.attr[(n, c)][2] < m["all_out_conf_below"] for c in succ)):
        return False
    return True


def _rel_ok(m: dict[str, Any], fam: str, name: str) -> bool:
    if m.get("names") and name not in m["names"]:
        return False
    return m.get("family", "*") in ("*", fam)


def _instances(v: _View, m: dict[str, Any]) -> list[Instance]:
    kind = m["kind"]
    if kind == "node":
        return [(n,) for n in v.nodes if _node_ok(v, n, m)]
    if kind == "rel":
        return [(s, d) for s, d, k, f in v.kg if _rel_ok(m, f, k) and v.match(m["src"], s) and v.match(m["dst"], d)]
    if kind == "rel_chain":
        out = []
        for a, b, _, f1 in v.kg:
            if (m["f1"] == "*" or f1 == m["f1"]) and v.match(m["a"], a) and v.match(m["b"], b):
                for c, _, f2 in v.kg_out[b]:
                    if c != a and (m["f2"] == "*" or f2 == m["f2"]) and v.match(m["c"], c):
                        out.append((a, b, c))
        return out
    if kind == "causal":
        cap = m.get("max_conf", 2.0)
        return [(s, d) for (s, d), (sign, _, conf) in v.attr.items()
                if sign == m["sign"] and conf <= cap and v.match(m["src"], s) and v.match(m["dst"], d)]
    if kind == "causal_chain":
        ta, tb, tc = m["nodes"]
        s1, s2 = m["signs"]
        out = []
        for b in v.nodes:
            if not v.match(tb, b):
                continue
            for a in v.pred[b]:
                if v.attr[(a, b)][0] == s1 and v.match(ta, a):
                    out.extend((a, b, c) for c in v.succ[b] if v.attr[(b, c)][0] == s2 and v.match(tc, c))
        return out
    if kind == "causal_path":
        return _same_sign_paths(v, m)
    if kind == "global":
        return _global(v, m)
    raise ValueError(f"unknown motif kind: {kind}")


def _same_sign_paths(v: _View, m: dict[str, Any]) -> list[Instance]:
    best: dict[str, tuple[int, str | None]] = {}
    for n in v.structure["topo"]:
        best[n] = max(((best[p][0] + 1, p) for p in v.pred[n] if v.attr[(p, n)][0] == m["sign"]), default=(0, None))
    out = []
    for n in v.nodes:
        if best[n][0] >= m["min_len"] and v.match(m["end"], n) and (not m.get("end_sink") or not v.succ[n]):
            path, cur = [n], best[n][1]
            while cur is not None:
                path.append(cur)
                cur = best[cur][1]
            out.append(tuple(reversed(path)))
    return out


def _global(v: _View, m: dict[str, Any]) -> list[Instance]:
    name = m["name"]
    causal_nodes = [n for n in v.nodes if v.succ[n] or v.pred[n]]
    if name == "sun":
        if len(v.attr) < 3 or v.structure["components"] != 1:
            return []
        conf = sum(a[2] for a in v.attr.values()) / len(v.attr)
        pos = sum(1 for a in v.attr.values() if a[0] > 0) / len(v.attr)
        return [()] if conf >= m["min_confidence"] and pos >= m["min_positive"] else []
    if name == "overlap":
        if len(causal_nodes) < 3:
            return []
        return [()] if all(v.kg_out[n] or v.kg_in[n] for n in causal_nodes) else []
    if name == "reach":
        roots = [n for n in causal_nodes if not v.pred[n]]
        sinks = [n for n in causal_nodes if not v.succ[n]]
        isolated = any(not (v.succ[n] or v.pred[n] or v.kg_out[n] or v.kg_in[n]) for n in v.nodes)
        if len(roots) < 2 or len(sinks) < 2 or isolated or v.structure["components"] != 1:
            return []
        desc = v.structure["descendants"]
        share = sum(1 for r in roots for s in sinks if s in desc[r]) / (len(roots) * len(sinks))
        return [()] if share >= m["min_share"] else []
    if name == "inconsistent_paths":
        signs: dict[str, dict[str, int]] = {}  # node -> ancestor -> bitmask (1 = some + path, 2 = some - path)
        for n in v.structure["topo"]:
            mine: dict[str, int] = {}
            for p in v.pred[n]:
                s = v.attr[(p, n)][0]
                for anc, mask in list(signs[p].items()) + [(p, 1)]:
                    mask = mask if s > 0 else ((mask & 1) << 1) | ((mask & 2) >> 1)
                    mine[anc] = mine.get(anc, 0) | mask
            signs[n] = mine
        return [(a, b) for b in v.nodes for a, mask in sorted(signs[b].items()) if mask == 3]
    raise ValueError(f"unknown global motif: {name}")


# ---------- scoring ----------


def _strength(count: int, null: list[int], single: bool) -> float:
    """How far a motif stands out from the shuffle, in 0..1.

    0 if absent or no more common than in the shuffle; otherwise tanh(z/2), so
    only over-representation counts. 1 when the shuffle can't move the count (a
    shape-only motif). Counted motifs are then shrunk by count/(count+1), so a
    single instance is weak evidence. A global property (true or not) and a motif
    marked "single" (one keystone is the whole point) aren't shrunk.
    """
    if count == 0:
        return 0.0
    mean = sum(null) / len(null)
    sd = math.sqrt(sum((c - mean) ** 2 for c in null) / len(null))
    if sd == 0:
        s = 1.0 if count >= mean else 0.0
    else:
        s = max(0.0, math.tanh((count - mean) / sd / 2))
    return s if single else s * count / (count + 1)


def _ego(snap: GraphSnapshot, focus: str, radius: int = 2) -> set[str]:
    g = nx.Graph()
    g.add_nodes_from(snap.nodes)
    g.add_edges_from((s, d) for s, d, _ in snap.kg.edges(keys=True))
    g.add_edges_from(snap.dag.edges())
    return set(nx.single_source_shortest_path_length(g, focus, cutoff=radius))


def _quotes(snap: GraphSnapshot, inst: Instance) -> list[str]:
    provs: list[dict] = []
    for a, b in pairwise(inst):
        for x, y in ((a, b), (b, a)):
            if snap.dag.has_edge(x, y):
                provs += snap.dag.edges[x, y].get("provenance", [])
            if snap.kg.has_edge(x, y):
                for attrs in snap.kg.get_edge_data(x, y).values():
                    provs += attrs.get("provenance", [])
    if len(inst) == 1:
        provs += snap.nodes[inst[0]].get("provenance", [])
    out: list[str] = []
    for p in provs:
        q = p.get("quote") if isinstance(p, dict) else None
        if q and q not in out:
            out.append(q)
    return out[:2]


def match(snap: GraphSnapshot, focus: str | None = None, path: str | Path = DEFAULT_SET,
          samples: int | None = None) -> dict[str, Any]:
    """The archetype the snapshot (or `focus`'s two-hop neighbourhood) most resembles."""
    cset = load_set(str(path))
    cfg = cset["settings"]
    if focus is not None and focus not in snap.nodes:
        raise KeyError(focus)
    keep = _ego(snap, focus) if focus else None
    view = _build(snap, keep, cset)
    unclassified = sorted({k for _, _, k, f in view.kg if f == "other"})
    base = {"version": snap.version, "focus": focus, "set": cset["name"], "unclassified_relations": unclassified}
    if len(view.nodes) < 3:
        return {**base, "archetype": None, "undecided": True, "margin": 0.0, "runners_up": [], "scores": [], "evidence": []}

    # Every motif in both orientations, counted on the real graph and on each shuffle.
    motifs: list[dict[str, Any]] = []
    for card in cset["cards"]:
        for m in card["motifs"]:
            motifs.append(m)
            if m.get("reversible"):
                motifs.append(_flip(m))
    real = [_instances(view, m) for m in motifs]
    rng = random.Random(cfg["seed"])
    nulls: list[list[int]] = [[] for _ in motifs]
    for _ in range(samples or cfg["null_samples"]):
        sv = _shuffled(view, rng, cset)
        for i, m in enumerate(motifs):
            nulls[i].append(len(_instances(sv, m)))

    scores, i = [], 0
    for card in cset["cards"]:
        up, rev, wsum, parts = 0.0, 0.0, 0.0, []
        for m in card["motifs"]:
            w = m.get("weight", 1)
            single = m.get("name") in PREDICATES or m.get("single", False)
            s_up = _strength(len(real[i]), nulls[i], single)
            entry = {"motif": m, "upright": (s_up, real[i])}
            i += 1
            if m.get("reversible"):
                entry["reversed"] = (_strength(len(real[i]), nulls[i], single), real[i])
                i += 1
            s_rev = entry.get("reversed", entry["upright"])[0]
            up, rev, wsum = up + w * s_up, rev + w * s_rev, wsum + w
            parts.append(entry)
        up, rev = up / wsum, rev / wsum
        reversed_ = any(m.get("reversible") for m in card["motifs"]) and rev >= up + cfg["delta"]
        scores.append({"card": card, "score": rev if reversed_ else up, "reversed": reversed_, "parts": parts})

    scores.sort(key=lambda r: (-r["score"], r["card"]["n"]))
    top, second = scores[0], scores[1]
    margin = top["score"] - second["score"]
    undecided = top["score"] < cfg["tau"]  # a close runner-up doesn't matter: the top card is shown

    def card_info(r: dict[str, Any]) -> dict[str, Any]:
        c = r["card"]
        return {k: c[k] for k in ("n", "numeral", "name", "theme", "glyph", "face")} | {
            "reversed": r["reversed"], "score": round(r["score"], 3)}

    evidence = []
    orient = "reversed" if top["reversed"] else "upright"
    for part in top["parts"]:
        s, insts = part.get(orient, part["upright"])
        if not insts:
            continue
        m = part["motif"]
        evidence.append({
            "layer": m["layer"], "text": m["text"], "strength": round(s, 3), "count": len(insts),
            "examples": [{"nodes": list(x), "labels": [snap.nodes[n].get("label", n) for n in x],
                          "quotes": _quotes(snap, x)} for x in insts[:3]],
        })
    return {
        **base,
        "archetype": card_info(top),
        "undecided": undecided,
        "margin": round(margin, 3),
        "runners_up": [card_info(r) for r in scores[1:4]],
        "scores": [card_info(r) for r in sorted(scores, key=lambda r: r["card"]["n"])],
        "evidence": evidence,
    }
