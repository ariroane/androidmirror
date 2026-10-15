"""The patch format: every change to the ontology is a Patch of Ops.

Patches are proposed by agents (or people), held by the Steward, and applied
only after a user approves them. `kind` is an open string so other engines can
add their own patch kinds (for example "refactor") and carry extra fields in
`extra` without changing this module.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

OpType = Literal[
    "add_node",
    "update_node",
    "remove_node",
    "add_relation",
    "remove_relation",
    "add_causal_edge",
    "remove_causal_edge",
]

Status = Literal["proposed", "approved", "rejected", "rolled_back"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Op(BaseModel):
    """One atomic change. Which fields matter depends on `op`.

    add_node / update_node: id, label, type, attrs
    remove_node: id
    add_relation / remove_relation: src, rel, dst
    add_causal_edge / remove_causal_edge: src, dst, sign, strength, confidence, mechanism
    """

    op: OpType
    id: str | None = None
    label: str | None = None
    type: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)
    src: str | None = None
    dst: str | None = None
    rel: str | None = None
    sign: int | None = None
    strength: float | None = None
    confidence: float | None = None
    mechanism: str | None = None


class Source(BaseModel):
    work: str = ""
    locator: str = ""  # e.g. "Book I, 1"
    quote: str = ""  # verbatim text the patch is grounded in


class Approval(BaseModel):
    approver: str
    at: str = Field(default_factory=now)
    note: str = ""


class Patch(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: str = "descriptive"  # descriptive | causal | refactor | ...
    ops: list[Op]
    rationale: str = ""
    source: Source = Field(default_factory=Source)
    proposed_by: str = "unknown"
    proposed_at: str = Field(default_factory=now)
    base_version: int | None = None
    status: Status = "proposed"
    approval: Approval | None = None
    inverse: list[Op] = Field(default_factory=list)  # filled in when applied
    applied_version: int | None = None
    warnings: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
