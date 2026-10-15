"""Beam search over designs, and handing the winners to the Steward.

A design is the current ontology plus a sequence of steps, each step one patch
proposed by an engine (refactor, break or chain). Every generation, each design
in the beam is offered to every engine; each proposal is tried on a sandbox
copy and scored; the best designs survive. Designs that lose a probe the
current ontology satisfies are dropped, so the search cannot trade away what
the user has said is true of the subject.

Nothing here writes the live ontology. `queue_designs` only proposes.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..ontology import GraphSnapshot, Ontology, OntologyError
from ..patches import Op, Patch, Source
from ..steward import Steward
from .fitness import Fitness, Probe, score
from .operators import Operator, discover


@dataclass
class Design:
    steps: list[tuple[str, Patch]]  # (operation family, the engine's patch)
    fitness: Fitness
    snapshot: GraphSnapshot = field(repr=False)

    @property
    def ops(self) -> list[Op]:
        return [op.model_copy(deep=True) for _, p in self.steps for op in p.ops]

    def summary(self) -> dict[str, Any]:
        return {
            "fitness": self.fitness.model_dump(),
            "steps": [{"family": fam, "kind": p.kind, "rationale": p.rationale, "proposal": p.id,
                       "ops": len(p.ops)} for fam, p in self.steps],
        }


def sandbox(snap: GraphSnapshot) -> Ontology:
    """A private, writable Ontology built from a snapshot. Never saved."""
    o = Ontology("sandbox")
    o.nodes = copy.deepcopy(snap.nodes)
    o.kg, o.dag = snap.kg.copy(), snap.dag.copy()
    o.version = snap.version
    return o


def _signature(o: Ontology) -> str:
    parts = sorted(f"n|{k}|{v.get('label')}|{v.get('type')}" for k, v in o.nodes.items())
    parts += sorted(f"r|{s}|{k}|{d}" for s, d, k in o.kg.edges(keys=True))
    parts += sorted(f"c|{s}|{d}|{a.get('sign', 1)}" for s, d, a in o.dag.edges(data=True))
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()


def evolve(
    onto: Ontology,
    probes: list[Probe] | None = None,
    operators: dict[str, Operator] | None = None,
    generations: int = 3,
    beam: int = 4,
    per_operator: int = 8,
    keep: int = 3,
    weights: dict[str, float] | None = None,
    protect_probes: bool = True,
) -> dict[str, Any]:
    """Search for designs that score better than the current ontology.

    Reads `onto` once (a snapshot) and never changes it. Returns the baseline
    fitness, up to `keep` improving designs (best first) and any engine errors.
    """
    probes = probes or []
    operators = discover() if operators is None else operators
    base_snap = onto.graph_snapshot()
    base = sandbox(base_snap)
    base_fit = score(base, probes, base, weights)
    protected = set(base_fit.passed) if protect_probes else set()

    frontier = [Design([], base_fit, base_snap)]
    seen = {_signature(base)}
    explored = 0
    errors: list[str] = []
    for _ in range(generations):
        children: list[Design] = []
        for parent in frontier:
            for family, propose in operators.items():
                try:
                    proposals = list(propose(sandbox(parent.snapshot).graph_snapshot()))[:per_operator]
                except Exception as e:  # one engine failing must not stop the others
                    errors.append(f"{family}: {e}")
                    continue
                for p in proposals:
                    child = sandbox(parent.snapshot)
                    try:
                        child.apply_ops([op.model_copy(deep=True) for op in p.ops])
                    except OntologyError:
                        continue
                    sig = _signature(child)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    explored += 1
                    fit = score(child, probes, base, weights)
                    if protected - set(fit.passed):
                        continue
                    children.append(Design(parent.steps + [(family, p)], fit, child.graph_snapshot()))
        if not children:
            break
        frontier = sorted(frontier + children, key=lambda d: d.fitness.total, reverse=True)[:beam]

    better = [d for d in frontier if d.steps and d.fitness.total > base_fit.total + 1e-9]
    return {
        "base_version": base_snap.version,
        "base": base_fit.model_dump(),
        "operators": sorted(operators),
        "explored": explored,
        "designs": better[:keep],
        "errors": list(dict.fromkeys(errors)),
    }


def queue_designs(steward: Steward, result: dict[str, Any], probes: list[Probe] | None = None) -> list[Patch]:
    """Offer each design to the Steward as one patch of kind "evolve" for approval.

    The patch refuses to apply if the ontology has moved under it
    (`require_fresh_base`); re-run the Evolver instead.
    """
    by_id = {p.id: p for p in probes or []}
    base = Fitness.model_validate(result["base"])
    queued = []
    for d in result["designs"]:
        gained = [pid for pid in d.fitness.passed if pid not in base.passed]
        families = ", ".join(dict.fromkeys(fam for fam, _ in d.steps))
        lines = [f"{i + 1}. {fam}: {p.rationale or p.kind}" for i, (fam, p) in enumerate(d.steps)]
        rationale = (f"Evolved design ({families}): fitness {base.total:.3f} -> {d.fitness.total:.3f}"
                     + (f", now reproduces: {'; '.join(by_id[g].claim or g for g in gained if g in by_id)}"
                        if gained else "")
                     + "\n" + "\n".join(lines))
        patch = Patch(
            kind="evolve",
            ops=d.ops,
            rationale=rationale,
            source=Source(work="", locator="Evolver"),
            proposed_by="evolver",
            base_version=result["base_version"],
            extra={
                "require_fresh_base": True,
                "fitness": d.fitness.model_dump(),
                "base_fitness": base.model_dump(),
                "probes_gained": gained,
                "steps": d.summary()["steps"],
            },
        )
        queued.append(steward.propose(patch))
    if queued:
        steward.save()
    return queued
