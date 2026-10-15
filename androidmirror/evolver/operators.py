"""Where the Evolver gets its mutations: the refactor, break and chain engines.

An engine plugs in by exposing `propose(snapshot: GraphSnapshot) -> list[Patch]`,
returning candidate patches without queueing or applying them. For engines
whose public API is shaped differently (the Breaker's candidate finders and
`propose_split` / `propose_sever`, the Chainer's `plan_merge`), a small adapter
here calls their own planners. The Evolver only ever calls them on sandbox
snapshots.
"""

from __future__ import annotations

import difflib
import importlib
from types import ModuleType
from typing import Callable

from ..ontology import GraphSnapshot, slugify
from ..patches import Patch

Operator = Callable[[GraphSnapshot], list[Patch]]

# operation family -> module that implements it
ENGINE_MODULES = {
    "refactor": "androidmirror.refactor",
    "break": "androidmirror.breaker",
    "chain": "androidmirror.chainer",
}

TOP = 6  # candidates asked of each engine per design


def _breaker(mod: ModuleType) -> Operator | None:
    if not all(hasattr(mod, f) for f in ("split_candidates", "sever_candidates", "propose_split", "propose_sever")):
        return None

    def propose(snap: GraphSnapshot) -> list[Patch]:
        from .search import sandbox

        onto = sandbox(snap)
        out = []
        for c in mod.split_candidates(onto, top=TOP):
            try:
                out.append(mod.propose_split(onto, c["id"]))
            except (KeyError, ValueError):
                continue  # e.g. ambivalent but one group: the Breaker needs named parts
        for c in mod.sever_candidates(onto, top=TOP):
            try:
                out.append(mod.propose_sever(onto, c["src"], c["dst"]))
            except (KeyError, ValueError):
                continue
        return out

    return propose


def merge_pairs(snap: GraphSnapshot, top: int = TOP) -> list[tuple[str, str, float]]:
    """Same-type node pairs that look like one concept: similar labels or shared neighbours."""
    live = [n for n, v in snap.nodes.items() if not v.get("attrs", {}).get("archived") and v.get("type") != "person"]

    def nbrs(n: str) -> set[str]:
        return set(snap.kg.predecessors(n)) | set(snap.kg.successors(n)) | set(snap.dag.predecessors(n)) | set(
            snap.dag.successors(n))

    near = {n: nbrs(n) for n in live}
    rows = []
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if snap.nodes[a].get("type") != snap.nodes[b].get("type"):
                continue
            label = difflib.SequenceMatcher(None, slugify(snap.nodes[a]["label"]), slugify(snap.nodes[b]["label"])).ratio()
            union = (near[a] | near[b]) - {a, b}
            shared = len((near[a] & near[b]) - {a, b}) / len(union) if union else 0.0
            score = max(label, shared if len((near[a] & near[b]) - {a, b}) >= 2 else 0.0)
            if score >= 0.75:
                rows.append((a, b, round(score, 3)))
    return sorted(rows, key=lambda r: r[2], reverse=True)[:top]


def _chainer(mod: ModuleType) -> Operator | None:
    if not hasattr(mod, "plan_merge"):
        return None
    error = getattr(mod, "ChainError", ValueError)

    def propose(snap: GraphSnapshot) -> list[Patch]:
        out = []
        for a, b, _ in merge_pairs(snap):
            # keep the better-connected node as the survivor
            keep, fold = (a, b) if snap.kg.degree(a) + snap.dag.degree(a) >= snap.kg.degree(b) + snap.dag.degree(b) else (b, a)
            try:
                out.append(mod.plan_merge(snap, keep, [fold]))
            except (error, ValueError):
                continue  # e.g. the merge would create a causal cycle or a sign conflict
        return out

    return propose


ADAPTERS: dict[str, Callable[[ModuleType], Operator | None]] = {"break": _breaker, "chain": _chainer}


def discover() -> dict[str, Operator]:
    """The operation families whose engines are installed, by name."""
    found: dict[str, Operator] = {}
    for name, module in ENGINE_MODULES.items():
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        fn = getattr(mod, "propose", None)
        if not callable(fn) and name in ADAPTERS:
            fn = ADAPTERS[name](mod)
        if callable(fn):
            found[name] = fn
    return found
