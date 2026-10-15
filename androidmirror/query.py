"""The query mechanism: find the relevant part of the ontology, then answer."""

from __future__ import annotations

import re
from typing import Any

from . import llm
from .ontology import Ontology

STOP = set("a an the of to in on and or is are was were what why how who does do did his her he she it about".split())


def match_nodes(onto: Ontology, question: str, limit: int = 8) -> list[str]:
    words = {w for w in re.findall(r"[a-z]+", question.lower()) if w not in STOP and len(w) > 2}
    scored = []
    for nid, node in onto.nodes.items():
        label_words = set(re.findall(r"[a-z]+", node["label"].lower())) | set(nid.split("_"))
        hits = sum(1 for w in words if any(lw.startswith(w[:5]) for lw in label_words))
        if hits:
            scored.append((hits, onto.kg.degree(nid) + onto.dag.degree(nid), nid))
    return [nid for *_, nid in sorted(scored, reverse=True)[:limit]]


def neighborhood_facts(onto: Ontology, nodes: list[str]) -> list[str]:
    """Plain-text facts about the given nodes and their direct neighbours."""
    label = lambda n: onto.nodes[n]["label"]  # noqa: E731
    facts: list[str] = []
    for n in nodes:
        facts.append(f"{label(n)} is a {onto.nodes[n].get('type', 'concept')}.")
        for s, d, rel in list(onto.kg.out_edges(n, keys=True)) + list(onto.kg.in_edges(n, keys=True)):
            facts.append(f"{label(s)} {rel.replace('_', ' ')} {label(d)}.")
        for s, d, a in list(onto.dag.out_edges(n, data=True)) + list(onto.dag.in_edges(n, data=True)):
            verb = "increases" if a.get("sign", 1) > 0 else "decreases"
            mech = f" ({a['mechanism']})" if a.get("mechanism") else ""
            facts.append(f"{label(s)} {verb} {label(d)}, strength {a.get('strength', 0.5):.1f}{mech}.")
    return list(dict.fromkeys(facts))


def ask(onto: Ontology, question: str) -> dict[str, Any]:
    nodes = match_nodes(onto, question)
    facts = neighborhood_facts(onto, nodes)
    result: dict[str, Any] = {"question": question, "nodes": nodes, "facts": facts, "answer": None}
    if not facts:
        result["answer"] = "The Mirror has nothing on that yet."
    elif llm.available():
        out = llm.answer(question, "\n".join(facts))
        result["answer"], result["nodes"] = out["answer"], out["nodes_used"] or nodes
    return result
