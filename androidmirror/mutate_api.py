"""HTTP routes for the mutate engine (see mirror/mutate.py).

Every route but approve, reject and undo is read-only on the Mirror: they write
proposals with their previews. Approve is the only route that changes
mirror.json, and it needs the approver's name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import llm
from .mirror.mutate import Engine, MutationError


class Propose(BaseModel):
    corpus: str | None = None
    limit: int = 8
    check: bool = False


class Merge(BaseModel):
    ids: list[str]
    into: str | None = None
    opposite: list[str] = []
    label: str | None = None
    corpus: str | None = None


class Sense(BaseModel):
    label: str
    gloss: str = ""
    aliases: list[str] = []
    passages: list[str] = []


class Split(BaseModel):
    node: str
    senses: list[Sense] | None = None
    claude: bool = False
    corpus: str | None = None


class Decide(BaseModel):
    approver: str = ""
    corpus: str | None = None


def register(app: FastAPI, root_of: Callable[[str | None], Path]) -> None:
    """Add the /api/mutate routes; `root_of(corpus)` is the built Mirror's folder."""

    def engine(corpus: str | None) -> Engine:
        return Engine(root_of(corpus))

    def run(fn):
        try:
            return fn()
        except (MutationError, llm.LLMUnavailable, llm.LLMTruncated) as e:
            raise HTTPException(400, str(e)) from None
        except KeyError as e:
            raise HTTPException(404, f"no such proposal: {e.args[0]}") from None

    @app.get("/api/mutate")
    def listing(corpus: str | None = None) -> dict:
        e = engine(corpus)
        return {"proposals": e.list(), "applied": e.applied()}

    @app.post("/api/mutate/propose")
    def propose(b: Propose) -> dict:
        return {"proposals": run(lambda: engine(b.corpus).propose(b.limit, b.check))}

    @app.post("/api/mutate/merge")
    def merge(b: Merge) -> dict:
        return run(lambda: engine(b.corpus).propose_merge(b.ids, b.into, b.opposite, b.label))

    @app.post("/api/mutate/split")
    def split(b: Split) -> dict:
        senses = [s.model_dump() for s in b.senses] if b.senses else None
        return run(lambda: engine(b.corpus).propose_split(b.node, senses, claude=b.claude))

    @app.post("/api/mutate/undo")
    def undo(b: Decide) -> dict:
        return run(lambda: engine(b.corpus).undo(b.approver or "user"))

    @app.get("/api/mutate/words/{node}")
    def words(node: str, corpus: str | None = None) -> dict:
        return run(lambda: engine(corpus).words(node))

    @app.get("/api/mutate/{pid}")
    def one(pid: str, corpus: str | None = None) -> dict:
        return run(lambda: engine(corpus).get(pid))

    @app.post("/api/mutate/{pid}/check")
    def check(pid: str, b: Decide) -> dict:
        return run(lambda: engine(b.corpus).check(pid))

    @app.post("/api/mutate/{pid}/approve")
    def approve(pid: str, b: Decide) -> dict:
        return run(lambda: engine(b.corpus).approve(pid, b.approver))

    @app.post("/api/mutate/{pid}/reject")
    def reject(pid: str, b: Decide) -> dict:
        return run(lambda: engine(b.corpus).reject(pid, b.approver or "user"))
