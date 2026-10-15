"""How well a design mirrors the subject.

A design is scored on:

- probes: claims about the subject the model should reproduce ("reason calms
  anger", "fortune and providence are distinct"), each tied to a source quote
- fidelity: the share of the current ontology's relations and causal edges it
  still carries (following merges, renames and splits)
- distinctness: penalises folding together concepts the subject names
  differently (merging near-identical labels is free)
- coverage: how much of the current ontology's quoted evidence it keeps
- grounding: the share of its nodes and edges backed by a source quote
- coherence: the share of nodes in its largest connected piece
- parsimony: smaller than the current ontology scores higher, larger lower
- robustness: one minus the share of nodes that are cut points

Structure alone moves the score only a little; a design has to keep what the
subject said, or reproduce more of it, to win.

Each component is 0..1; the total is their weighted mean. Probes are left out
of the mean when there are none.
"""

from __future__ import annotations

import difflib
import itertools
import json
from pathlib import Path
from typing import Literal

import networkx as nx
from pydantic import BaseModel, Field

from .. import analysis
from ..ontology import GraphSnapshot, Ontology, slugify

SEED_PROBES = Path(__file__).resolve().parents[2] / "data" / "probes_meditations.json"

DEFAULT_WEIGHTS = {
    "probes": 0.3,
    "fidelity": 0.2,
    "distinctness": 0.15,
    "coverage": 0.1,
    "grounding": 0.1,
    "coherence": 0.05,
    "parsimony": 0.05,
    "robustness": 0.05,
}

SAME_LABEL = 0.8  # label similarity at or above which a merge costs no distinctness


class Probe(BaseModel):
    """A claim about the subject that a good design should reproduce.

    causes:   raising `a` moves `b` in `direction` (+1 up, -1 down)
    related:  `a` and `b` are linked in either graph
    distinct: `a` and `b` are separate nodes (guards against bad merges)
    exists:   `a` is in the ontology
    Nodes are named by label; aliases left by merges and renames also match.
    """

    id: str = ""
    claim: str = ""
    kind: Literal["causes", "related", "distinct", "exists"]
    a: str
    b: str | None = None
    direction: int = 1
    weight: float = 1.0
    locator: str = ""
    quote: str = ""


class Fitness(BaseModel):
    total: float
    components: dict[str, float]
    passed: list[str] = Field(default_factory=list)  # probe ids
    failed: list[str] = Field(default_factory=list)


def load_probes(path: Path) -> list[Probe]:
    if not path.exists():
        return []
    probes = [Probe.model_validate(p) for p in json.loads(path.read_text(encoding="utf-8"))]
    for i, p in enumerate(probes):
        p.id = p.id or f"probe{i + 1}"
    return probes


def default_probes(data_dir: Path) -> list[Probe]:
    """The user's own probes (data/probes.json) if present, else the Meditations probes."""
    own = data_dir / "probes.json"
    return load_probes(own if own.exists() else SEED_PROBES)


def resolve(onto: Ontology, label: str) -> str | None:
    """Node id for a label, also matching aliases recorded by merges and renames."""
    nid = onto.find_node(label)
    if nid:
        return nid
    want = slugify(label)
    for nid, node in onto.nodes.items():
        aliases = node.get("attrs", {}).get("aliases") or []
        if any(slugify(str(a)) == want for a in aliases):
            return nid
    return None


def resolve_all(onto: Ontology, label: str) -> set[str]:
    """Every live node now standing for `label`: itself, or the parts it was split into."""
    nid = resolve(onto, label)
    if nid is None:
        return set()
    out, todo = set(), [nid]
    while todo:
        n = todo.pop()
        attrs = onto.nodes.get(n, {}).get("attrs", {})
        parts = [p for p in attrs.get("split_into") or [] if p in onto.nodes and p not in out]
        if attrs.get("archived") and parts:
            todo.extend(parts)
        else:
            out.add(n)
    return out


def _fidelity(onto: Ontology, base: Ontology) -> float:
    labels = {n: v["label"] for n, v in base.nodes.items()}
    where = {n: resolve_all(onto, lab) for n, lab in labels.items()}
    facts = kept = 0
    for s, d, k in base.kg.edges(keys=True):
        facts += 1
        kept += any(onto.kg.has_edge(a, b, key=k) for a in where[s] for b in where[d] if a != b)
    for s, d, a in base.dag.edges(data=True):
        facts += 1
        sign = a.get("sign", 1)
        kept += any(onto.dag.has_edge(x, y) and onto.dag.edges[x, y].get("sign", 1) == sign
                    for x in where[s] for y in where[d] if x != y)
    return kept / facts if facts else 1.0


def _distinctness(onto: Ontology, base: Ontology) -> float:
    groups: dict[str, list[str]] = {}
    for n, v in base.nodes.items():
        for r in resolve_all(onto, v["label"]):
            groups.setdefault(r, []).append(v["label"])
    cost = 0.0
    for members in groups.values():
        for a, b in itertools.combinations(sorted(set(members)), 2):
            sim = difflib.SequenceMatcher(None, slugify(a), slugify(b)).ratio()
            cost += max(0.0, SAME_LABEL - sim) / SAME_LABEL
    return 1.0 / (1.0 + cost)


def check(onto: Ontology, probe: Probe) -> bool:
    a = resolve(onto, probe.a)
    if probe.kind == "exists":
        return a is not None
    b = resolve(onto, probe.b or "")
    if a is None or b is None:
        return False
    if probe.kind == "distinct":
        return a != b
    if a == b:
        return False
    if probe.kind == "related":
        return (onto.kg.has_edge(a, b) or onto.kg.has_edge(b, a)
                or nx.has_path(onto.dag, a, b) or nx.has_path(onto.dag, b, a))
    effect = analysis.propagate(onto, {a: 1.0}).get(b, 0.0)
    return abs(effect) > 1e-6 and (effect > 0) == (probe.direction > 0)


def quotes(onto: Ontology) -> set[str]:
    """Every source quote the ontology carries, on nodes and on edges."""
    out: set[str] = set()
    records = list(onto.nodes.values())
    records += [a for *_, a in onto.kg.edges(keys=True, data=True)]
    records += [a for *_, a in onto.dag.edges(data=True)]
    for r in records:
        for prov in r.get("provenance") or []:
            q = (prov or {}).get("quote", "").strip() if isinstance(prov, dict) else ""
            if q:
                out.add(q)
    return out


def _grounding(onto: Ontology) -> float:
    def grounded(r: dict) -> bool:
        return any(isinstance(p, dict) and p.get("quote", "").strip() for p in r.get("provenance") or [])

    records = list(onto.nodes.values())
    records += [a for *_, a in onto.kg.edges(keys=True, data=True)]
    records += [a for *_, a in onto.dag.edges(data=True)]
    return sum(map(grounded, records)) / len(records) if records else 0.0


def size(onto: Ontology) -> int:
    return len(onto.nodes) + onto.kg.number_of_edges() + onto.dag.number_of_edges()


def score(onto: Ontology, probes: list[Probe], base: Ontology | GraphSnapshot,
          weights: dict[str, float] | None = None) -> Fitness:
    """Score `onto` as a design, relative to the `base` ontology it was derived from."""
    if isinstance(base, GraphSnapshot):
        from .search import sandbox

        base = sandbox(base)
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    m = analysis.metrics(onto)
    base_quotes = quotes(base)
    base_size = size(base) or 1
    comp = {
        "fidelity": _fidelity(onto, base),
        "distinctness": _distinctness(onto, base),
        "coverage": len(quotes(onto) & base_quotes) / len(base_quotes) if base_quotes else 1.0,
        "grounding": _grounding(onto),
        "coherence": m["coherence"],
        "parsimony": max(0.0, min(1.0, 0.5 + 0.5 * (base_size - size(onto)) / base_size)),
        "robustness": 1.0 - m["fragility"],
    }
    passed, failed = [], []
    if probes:
        total_w = sum(p.weight for p in probes) or 1.0
        got = 0.0
        for p in probes:
            if check(onto, p):
                passed.append(p.id)
                got += p.weight
            else:
                failed.append(p.id)
        comp["probes"] = got / total_w
    used = {k: weights[k] for k in comp if weights.get(k, 0) > 0}
    total = sum(comp[k] * w for k, w in used.items()) / (sum(used.values()) or 1.0)
    return Fitness(total=round(total, 4), components={k: round(v, 4) for k, v in comp.items()},
                   passed=passed, failed=failed)
