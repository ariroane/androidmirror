"""HTTP API and web UI for the Mirror (served from web/index.html).

Read-only over built Mirrors: build them with `python -m androidmirror build`.
Every route takes an optional `corpus` (default: the first one built).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import archetypes, llm, mutate_api
from .mirror import ask as oracle
from .mirror import derange, interview, ooda, question
from .mirror.build import mirror_dir
from .mirror.corpus import SPECS
from .mirror.store import Mirror

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


class Ask(BaseModel):
    question: str
    corpus: str | None = None


class Place(BaseModel):
    concept: str
    tier: str | None = None  # observe, orient, decide, act; None: automatic
    by: str = ""
    corpus: str | None = None


class Transmute(BaseModel):
    a: str
    b: str
    claude: bool = False
    pairs: list[str] = []  # "A=B"
    opposites: list[str] = []  # "A=B": B's concept is the opposite of A's
    drop: list[str] = []
    keep: list[str] = []  # approve: trims to spare, by number or id
    approver: str = ""  # approve: who approved it


def create_app(data_dir: Path | None = None) -> FastAPI:
    data_dir = data_dir or Path(os.environ.get("ANDROIDMIRROR_DATA", ROOT / "data"))
    app = FastAPI(title="Android Mirror")
    cache: dict[str, tuple[float, Mirror]] = {}

    def built() -> list[str]:
        root = data_dir / "mirror"
        return sorted(p.name for p in root.iterdir() if (p / "mirror.json").exists()) if root.exists() else []

    def get(corpus: str | None) -> Mirror:
        names = built()
        name = corpus or os.environ.get("ANDROIDMIRROR_CORPUS") or (names[0] if names else None)
        if not name or name not in names:
            raise HTTPException(404, f"no Mirror built for {name or 'any corpus'}; run: python -m androidmirror build")
        path = mirror_dir(data_dir, name) / "mirror.json"
        mtime = path.stat().st_mtime
        if name not in cache or cache[name][0] != mtime:  # reload after a rebuild
            cache[name] = (mtime, Mirror.load(path.parent))
        return cache[name][1]

    def missing(e: KeyError) -> HTTPException:
        return HTTPException(404, f"not found: {e.args[0]}")

    app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    @app.get("/interview")
    def interview_page() -> FileResponse:
        return FileResponse(WEB / "interview.html")

    @app.get("/api/status")
    def status() -> dict:
        specs = sorted(p.stem for p in SPECS.glob("*.json"))
        return {"built": built(), "corpora": specs, "llm": llm.available(), "model": llm.DEFAULT_MODEL}

    @app.get("/api/meta")
    def meta(corpus: str | None = None) -> dict:
        m = get(corpus)
        return {**m.meta, "stats": m.kb.stats(), "variables": len(m.variables), "concepts": len(m.nodes),
                "judgments": sum(1 for j in m.judgments if j["held"]), "causal": m.causal.get("stats"),
                "values": len(m.values)}

    @app.get("/api/graph")
    def graph(layer: str = "kg", limit: int = 120, corpus: str | None = None) -> dict:
        m = get(corpus)
        return ooda.graph(m) if layer == "ooda" else m.graph(layer, limit)

    @app.get("/api/ooda")
    def ooda_model(corpus: str | None = None) -> dict:
        """The OODA model: concepts in four tiers, built from the two graphs. Read-only."""
        return ooda.model(get(corpus))

    @app.post("/api/ooda/place")
    def ooda_place(p: Place) -> dict:
        """Put a concept in a tier yourself, or back to automatic (tier null). Changes neither graph."""
        m = get(p.corpus)
        if p.concept not in m.nodes:
            raise HTTPException(404, f"not found: {p.concept}")
        try:
            ooda.place(m.root, p.concept, p.tier, p.by)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return ooda.model(m)

    @app.get("/api/node/{nid}")
    def node(nid: str, corpus: str | None = None) -> dict:
        try:
            return get(corpus).node(nid)
        except KeyError as e:
            raise missing(e) from None

    @app.get("/api/passage/{pid}")
    def passage(pid: str, corpus: str | None = None) -> dict:
        try:
            return get(corpus).passage(pid)
        except KeyError as e:
            raise missing(e) from None

    @app.get("/api/knowledge")
    def knowledge(corpus: str | None = None) -> dict:
        return get(corpus).knowledge()

    @app.get("/api/holds")
    def holds(s: str, r: str, o: str, corpus: str | None = None) -> dict:
        m = get(corpus)
        return {**m.holds(s, r, o), "s_label": m.label(s), "o_label": m.label(o)}

    @app.get("/api/causal")
    def causal(corpus: str | None = None) -> dict:
        return get(corpus).causal_view()

    @app.get("/api/edge")
    def edge(src: str, dst: str, corpus: str | None = None) -> dict:
        try:
            return get(corpus).edge(src, dst)
        except KeyError as e:
            raise missing(e) from None

    @app.get("/api/explain")
    def explain(x: str, y: str, corpus: str | None = None) -> dict:
        m = get(corpus)
        if x not in m.nodes or y not in m.nodes:
            raise HTTPException(400, "both ends must be concepts of the Mirror")
        return m.explain(x, y)

    @app.get("/api/derange")
    def derange_(corpus: str | None = None) -> dict:
        m = get(corpus)
        if not hasattr(m, "derange"):
            root = mirror_dir(data_dir, m.meta["corpus"]["id"])
            m.derange = derange.saved(root) or derange.report(root, m)
        return {**m.derange, "labels": {n: m.label(n) for n in m.nodes}}

    @app.post("/api/ask")
    def ask(a: Ask) -> dict:
        try:
            return oracle.ask(get(a.corpus), a.question)
        except llm.LLMUnavailable as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/archetype")
    def archetype(corpus: str | None = None) -> dict:
        """The archetype the whole Mirror most resembles. Read-only."""
        return archetypes.match(get(corpus).snapshot())

    @app.get("/api/concepts")
    def concepts(corpus: str | None = None) -> list[dict]:
        m = get(corpus)
        return sorted(({"id": k, "label": v["label"], "kind": v["kind"], "support": v.get("support", 0)}
                       for k, v in m.nodes.items()), key=lambda r: -r["support"])

    mutate_api.register(app, lambda c: mirror_dir(data_dir, get(c).meta["corpus"]["id"]))
    app.include_router(question.routes(get))  # the Question engine: questions, answers, proposals to approve
    app.include_router(interview.routes(data_dir))  # the interview questionnaires: answers saved as they come, then a corpus

    @app.get("/api/transmutate")
    def transmutation(a: str, b: str) -> dict:
        from .mirror import transmutate as tm

        p = tm._read(tm.proposal_dir(data_dir, a, b) / "proposal.json")
        if p is None:
            raise HTTPException(404, f"no proposal to combine {a} and {b}")
        return p

    @app.post("/api/transmutate/propose")
    def transmutate_propose(t: Transmute) -> dict:
        """Only scores and saves a proposal: no Mirror is written until it is approved."""
        from .mirror import transmutate as tm

        try:
            add = [tm.parse_pair(x) for x in t.pairs] + [tm.parse_pair(x, -1) for x in t.opposites]
            return tm.propose(data_dir, t.a, t.b, t.claude, add, t.drop, log=lambda *_: None)
        except (FileNotFoundError, RuntimeError, KeyError, ValueError, llm.LLMUnavailable) as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/transmutate/compare")
    def transmutate_compare(t: Transmute) -> dict:
        """How two Mirrors differ in knowledge, causes and OODA. Saves nothing."""
        from .mirror import transmutate as tm

        try:
            add = [tm.parse_pair(x) for x in t.pairs] + [tm.parse_pair(x, -1) for x in t.opposites]
            return tm.compare_only(data_dir, t.a, t.b, t.claude, add, t.drop)
        except (FileNotFoundError, RuntimeError, KeyError, ValueError, llm.LLMUnavailable) as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/transmutate/approve")
    def transmutate_approve(t: Transmute) -> dict:
        from .mirror import transmutate as tm

        try:
            m = tm.approve(data_dir, t.a, t.b, t.keep, t.approver or "you", log=lambda *_: None)
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            raise HTTPException(400, str(e)) from None
        return {"built": m["meta"]["corpus"]["id"], "transmutation": m["meta"]["transmutation"]}

    return app
