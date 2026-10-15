"""HTTP routes and CLI commands for the Breaker. Both only queue proposals.

Kept apart from `breaker.py` so the engine itself stays free of web code, and
apart from `server.py` / `cli.py` so each engine plugs in with one line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import breaker, corpus
from .ontology import Ontology
from .steward import Steward

COMMANDS = ("break-candidates", "break-split", "break-sever")


class SplitReq(BaseModel):
    node: str
    parts: list[dict] | None = None  # [{"label", "type"?, "id"?}]; omit to split by neighbour group
    assign: dict[str, str] | None = None  # link key -> part id or "*"


class SeverReq(BaseModel):
    src: str
    dst: str
    rels: list[str] | None = None
    causal: bool = True


def register(app: FastAPI, steward: Steward) -> None:
    """Add /api/break/* to the app. Proposals go to the Review queue; nothing applies here."""
    onto = steward.ontology

    @app.get("/api/break/candidates")
    def break_candidates() -> dict:
        return {"split": breaker.split_candidates(onto), "sever": breaker.sever_candidates(onto)}

    @app.get("/api/break/links/{node}")
    def break_links(node: str) -> list[dict]:
        return [{k: v for k, v in ln.items() if k != "attrs"} for ln in breaker.links(onto, node)]

    @app.post("/api/break/split")
    def break_split(r: SplitReq) -> dict:
        return _queue(steward, lambda: breaker.propose_split(onto, r.node, r.parts, r.assign))

    @app.post("/api/break/sever")
    def break_sever(r: SeverReq) -> dict:
        return _queue(steward, lambda: breaker.propose_sever(onto, r.src, r.dst, r.rels, r.causal))


def _queue(steward: Steward, build) -> dict:
    try:
        patch = steward.propose(build())
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    steward.save()
    return patch.model_dump()


def add_cli(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("break-candidates", help="list nodes the Breaker would split and links it would sever")
    bs = sub.add_parser("break-split", help="propose splitting a node in both graphs (queued for review)")
    bs.add_argument("node")
    bs.add_argument("--parts", nargs="+", help="part labels; omit to split by neighbour group")
    bv = sub.add_parser("break-sever", help="propose cutting every link between two nodes (queued for review)")
    bv.add_argument("src")
    bv.add_argument("dst")
    bv.add_argument("--rel", dest="rels", nargs="+", help="only these descriptive relations")
    bv.add_argument("--keep-causal", action="store_true", help="leave the causal edge in place")


def run_cli(args: argparse.Namespace, data_dir: Path) -> bool:
    """Run a break-* command. Returns False if `args.cmd` is not one of them."""
    if args.cmd not in COMMANDS:
        return False
    onto = Ontology.load(data_dir / "ontology.json")
    if args.cmd == "break-candidates":
        print(json.dumps({"split": breaker.split_candidates(onto), "sever": breaker.sever_candidates(onto)}, indent=2))
        return True
    st = Steward(onto, data_dir, corpus_text=corpus.load_text(data_dir))
    try:
        if args.cmd == "break-split":
            parts = [{"label": p} for p in args.parts] if args.parts else None
            patch = breaker.propose_split(onto, args.node, parts)
        else:
            patch = breaker.propose_sever(onto, args.src, args.dst, args.rels, not args.keep_causal)
    except (KeyError, ValueError) as e:
        raise SystemExit(f"not queued: {e}")
    st.propose(patch)
    st.save()
    warn = f"  [{'; '.join(patch.warnings)}]" if patch.warnings else ""
    print(f"queued {patch.id}: {patch.rationale}{warn}")
    return True
