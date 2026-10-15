"""The Ingestor: reads the corpus chunk by chunk and queues proposed patches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import llm
from .corpus import WORK, Chunk
from .ontology import slugify
from .patches import Op, Patch, Source
from .steward import Steward


def to_patch(raw: dict[str, Any], locator: str, proposed_by: str, work: str = WORK) -> Patch:
    """Turn one extracted patch (llm.EXTRACT_SCHEMA item) into a Patch."""
    ops: list[Op] = []
    for n in raw.get("nodes", []):
        ops.append(Op(op="add_node", id=slugify(n.get("id") or n["label"]), label=n["label"], type=n.get("type")))
    for r in raw.get("relations", []):
        ops.append(Op(op="add_relation", src=slugify(r["src"]), rel=slugify(r["rel"]), dst=slugify(r["dst"])))
    for e in raw.get("causal_edges", []):
        ops.append(
            Op(
                op="add_causal_edge",
                src=slugify(e["src"]),
                dst=slugify(e["dst"]),
                sign=e.get("sign", 1),
                strength=e.get("strength"),
                confidence=e.get("confidence"),
                mechanism=e.get("mechanism", ""),
            )
        )
    return Patch(
        kind=raw.get("kind", "descriptive"),
        ops=ops,
        rationale=raw.get("rationale", ""),
        source=Source(work=work, locator=locator, quote=raw.get("quote", "")),
        proposed_by=proposed_by,
    )


def ingest_chunks(
    steward: Steward,
    chunks: list[Chunk],
    extract: Callable[[str, str, str], list[dict[str, Any]]] = llm.extract,
) -> list[Patch]:
    """Run the Cartographer/Causal Modeler over chunks and queue the results.

    Known nodes include those in still-pending proposals, so later chunks reuse
    ids from earlier ones instead of inventing duplicates.
    """
    queued: list[Patch] = []
    pending: dict[str, Op] = {}  # node id -> its add_node op, from proposals not yet approved
    for chunk in chunks:
        known = steward.ontology.summary()
        if pending:
            known += "\n" + "\n".join(f"{k}: {o.label} ({o.type}, pending)" for k, o in pending.items())
        for raw in extract(chunk.text, chunk.locator, known):
            patch = to_patch(raw, chunk.locator, proposed_by="cartographer")
            _self_contain(patch, steward, pending)
            steward.propose(patch)
            for op in patch.ops:
                if op.op == "add_node" and op.id:
                    pending[op.id] = op
            queued.append(patch)
        steward.save()
    return queued


def _self_contain(patch: Patch, steward: Steward, pending: dict[str, Op]) -> None:
    """Add any node a patch references but does not define, so patches can be approved in any order."""
    defined = {op.id for op in patch.ops if op.op == "add_node"}
    for op in list(patch.ops):
        for nid in (op.src, op.dst):
            if nid and nid not in defined and nid not in steward.ontology.nodes and nid in pending:
                patch.ops.insert(0, pending[nid].model_copy())
                defined.add(nid)


def load_seed(steward: Steward, path: Path) -> list[Patch]:
    """Queue hand-authored patches from a JSON file of EXTRACT_SCHEMA items."""
    data = json.loads(path.read_text(encoding="utf-8"))
    queued = []
    pending: dict[str, Op] = {}
    for raw in data["patches"]:
        patch = to_patch(raw, raw.get("locator", ""), proposed_by=data.get("proposed_by", "seed"))
        _self_contain(patch, steward, pending)
        queued.append(steward.propose(patch))
        pending.update({op.id: op for op in patch.ops if op.op == "add_node" and op.id})
    steward.save()
    return queued
