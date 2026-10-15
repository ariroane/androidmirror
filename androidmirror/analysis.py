"""Empirical investigation, interventions, the Breaker and the Chernoff metrics.

Everything here reads a snapshot of the ontology and never changes it.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .ontology import Ontology

POSITIVE_TYPES = {"virtue", "value", "practice"}
NEGATIVE_TYPES = {"vice", "emotion"}


def important_nodes(onto: Ontology, top: int = 10) -> list[dict[str, Any]]:
    """Rank nodes by combined descriptive centrality and causal reach."""
    if not onto.nodes:
        return []
    und = nx.Graph(onto.kg.to_undirected())
    und.add_nodes_from(onto.nodes)
    pr = nx.degree_centrality(und)  # pagerank would need scipy; degree is enough at this size
    reach = {n: _reach(onto, n) for n in onto.nodes}
    max_reach = max(reach.values()) or 1.0
    max_pr = max(pr.values()) or 1.0
    rows = [
        {
            "id": n,
            "label": onto.nodes[n]["label"],
            "centrality": round(pr.get(n, 0.0), 4),
            "causal_reach": round(reach[n], 3),
            "score": round(0.5 * pr.get(n, 0.0) / max_pr + 0.5 * reach[n] / max_reach, 4),
        }
        for n in onto.nodes
    ]
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:top]


def _reach(onto: Ontology, node: str) -> float:
    """Sum of path-weighted influence of `node` on all its descendants."""
    return sum(abs(v) for k, v in propagate(onto, {node: 1.0}).items() if k != node)


def propagate(onto: Ontology, shocks: dict[str, float]) -> dict[str, float]:
    """Linear propagation of shocks through the causal DAG in topological order.

    The effect on a child is sum(parent_effect * sign * strength * confidence).
    """
    effect = {n: 0.0 for n in onto.dag.nodes}
    effect.update(shocks)
    for n in nx.topological_sort(onto.dag):
        if n in shocks:
            continue  # an intervened node is held at its set value (do-operator)
        total = 0.0
        for p in onto.dag.predecessors(n):
            a = onto.dag.edges[p, n]
            total += effect[p] * a.get("sign", 1) * a.get("strength", 0.5) * a.get("confidence", 0.5)
        effect[n] = total
    return effect


def intervene(onto: Ontology, node: str, delta: float) -> list[dict[str, Any]]:
    """do(node += delta): return the nodes it moves, largest effect first."""
    if node not in onto.nodes:
        raise KeyError(node)
    eff = propagate(onto, {node: delta})
    rows = [
        {"id": n, "label": onto.nodes[n]["label"], "effect": round(v, 4)}
        for n, v in eff.items()
        if n != node and abs(v) > 1e-6
    ]
    return sorted(rows, key=lambda r: abs(r["effect"]), reverse=True)


def breaker(onto: Ontology, top: int = 5) -> dict[str, Any]:
    """Shiva: points whose destruction would most fragment the subjective world.

    - cut_points: nodes whose removal disconnects the combined graph
    - load_bearing: nodes whose removal destroys the most causal influence
    - tensions: node pairs the author links with opposing causal signs
    """
    combined = nx.Graph()
    combined.add_nodes_from(onto.nodes)
    combined.add_edges_from((s, d) for s, d, _ in onto.kg.edges(keys=True))
    combined.add_edges_from(onto.dag.edges())
    cut_points = []
    for n in nx.articulation_points(combined):
        g = combined.copy()
        g.remove_node(n)
        sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
        cut_points.append({"id": n, "label": onto.nodes[n]["label"], "fragments": len(sizes), "sizes": sizes[:5]})
    cut_points.sort(key=lambda r: (r["fragments"], -r["sizes"][0] if r["sizes"] else 0), reverse=True)

    base = sum(_reach(onto, n) for n in onto.nodes)
    load = []
    for n in onto.nodes:
        if onto.dag.degree(n) == 0:
            continue
        trial = Ontology()
        trial.nodes = {k: v for k, v in onto.nodes.items() if k != n}
        trial.dag = onto.dag.copy()
        trial.dag.remove_node(n)
        lost = base - sum(_reach(trial, m) for m in trial.nodes)
        load.append({"id": n, "label": onto.nodes[n]["label"], "influence_lost": round(lost, 3)})
    load.sort(key=lambda r: r["influence_lost"], reverse=True)

    tensions = []
    for n in onto.dag.nodes:
        preds = list(onto.dag.predecessors(n))
        signs = {p: onto.dag.edges[p, n].get("sign", 1) for p in preds}
        pos = [p for p, s in signs.items() if s > 0]
        neg = [p for p, s in signs.items() if s < 0]
        for a in pos:
            for b in neg:
                tensions.append({"target": n, "pushes_up": a, "pushes_down": b})
    return {"cut_points": cut_points[:top], "load_bearing": load[:top], "tensions": tensions[:top * 2]}


def metrics(onto: Ontology) -> dict[str, float]:
    """Scalar readings of the Mirror's state, each 0..1, for the Chernoff face."""
    n = len(onto.nodes)
    if n == 0:
        return {"size": 0.0, "coherence": 0.0, "tension": 0.0, "valence": 0.5, "fragility": 0.0, "certainty": 0.0}
    combined = nx.Graph()
    combined.add_nodes_from(onto.nodes)
    combined.add_edges_from((s, d) for s, d, _ in onto.kg.edges(keys=True))
    combined.add_edges_from(onto.dag.edges())
    largest = max((len(c) for c in nx.connected_components(combined)), default=0)
    edges = list(onto.dag.edges(data=True))
    neg = sum(1 for *_, a in edges if a.get("sign", 1) < 0)
    pos_nodes = sum(1 for v in onto.nodes.values() if v.get("type") in POSITIVE_TYPES)
    neg_nodes = sum(1 for v in onto.nodes.values() if v.get("type") in NEGATIVE_TYPES)
    cuts = len(list(nx.articulation_points(combined))) if combined.number_of_edges() else 0
    conf = sum(a.get("confidence", 0.5) for *_, a in edges) / len(edges) if edges else 0.0
    return {
        "size": round(min(1.0, n / 200), 3),
        "coherence": round(largest / n, 3),
        "tension": round(neg / len(edges), 3) if edges else 0.0,
        "valence": round((pos_nodes + 1) / (pos_nodes + neg_nodes + 2), 3),
        "fragility": round(cuts / n, 3),
        "certainty": round(conf, 3),
    }
