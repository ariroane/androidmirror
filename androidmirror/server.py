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

from . import archetypes, llm
from .mirror import ask as oracle
from .mirror import derange
from .mirror.build import mirror_dir
from .mirror.corpus import SPECS
from .mirror.store import Mirror

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


class Ask(BaseModel):
    question: str
    corpus: str | None = None


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
        return get(corpus).graph(layer, limit)

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
    def archetype(focus: str | None = None, corpus: str | None = None) -> dict:
        snap = get(corpus).snapshot()
        try:
            return archetypes.match(snap, focus=focus)
        except KeyError:
            raise HTTPException(404, f"no such node: {focus}") from None

    @app.get("/api/concepts")
    def concepts(corpus: str | None = None) -> list[dict]:
        m = get(corpus)
        return sorted(({"id": k, "label": v["label"], "kind": v["kind"], "support": v.get("support", 0)}
                       for k, v in m.nodes.items()), key=lambda r: -r["support"])

    return app
