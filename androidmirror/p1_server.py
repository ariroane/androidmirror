"""HTTP API and the web UI (served from web/)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, archetypes, corpus, evolver, llm, query
from .ingest import ingest_chunks
from .ontology import Ontology
from .steward import GovernanceError, Steward, normalize

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


class Decision(BaseModel):
    approver: str
    note: str = ""


class IngestReq(BaseModel):
    book: int = 1
    start: int = 0
    count: int = 3


class Ask(BaseModel):
    question: str


class EvolveReq(BaseModel):
    generations: int = 3
    beam: int = 4
    keep: int = 3
    queue: bool = True


class RefactorQueueReq(BaseModel):
    keys: list[str] | None = None  # None queues every suggestion


class Intervention(BaseModel):
    node: str
    delta: float = 1.0
    retell: bool = False


def create_app(data_dir: Path | None = None) -> FastAPI:
    data_dir = data_dir or Path(os.environ.get("ANDROIDMIRROR_DATA", ROOT / "data"))
    onto = Ontology.load(data_dir / "ontology.json")
    steward = Steward(onto, data_dir, corpus_text=corpus.load_text(data_dir))
    app = FastAPI(title="Android Mirror")
    app.state.steward = steward
    from .breaker_api import register as register_breaker  # Breaker routes: proposals only

    register_breaker(app, steward)

    def fail(e: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(e))

    app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "p1" / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        return {
            "version": onto.version,
            "nodes": len(onto.nodes),
            "proposed": len(steward.list("proposed")),
            "llm": llm.available(),
            "model": llm.DEFAULT_MODEL,
            "corpus": steward.corpus_norm is not None,
        }

    @app.get("/api/graph")
    def graph() -> dict:
        return onto.snapshot()

    @app.get("/api/patches")
    def patches(status: str | None = "proposed") -> list[dict]:
        return [p.model_dump() for p in steward.list(status or None)]

    @app.post("/api/patches/{patch_id}/approve")
    def approve(patch_id: str, d: Decision) -> dict:
        try:
            return steward.approve(patch_id, d.approver, d.note).model_dump()
        except GovernanceError as e:
            raise fail(e)

    @app.post("/api/patches/{patch_id}/reject")
    def reject(patch_id: str, d: Decision) -> dict:
        try:
            return steward.reject(patch_id, d.approver, d.note).model_dump()
        except GovernanceError as e:
            raise fail(e)

    @app.post("/api/patches/{patch_id}/rollback")
    def rollback(patch_id: str, d: Decision) -> dict:
        try:
            return steward.rollback(patch_id, d.approver).model_dump()
        except GovernanceError as e:
            raise fail(e)

    @app.post("/api/ingest")
    def ingest(r: IngestReq) -> dict:
        try:
            path = corpus.download(data_dir)
        except Exception as e:  # network trouble on the user's machine
            raise HTTPException(status_code=502, detail=f"could not download the corpus: {e}")
        text = path.read_text(encoding="utf-8", errors="replace")
        steward.corpus_norm = steward.corpus_norm or normalize(text)
        chunks = [c for c in corpus.split_sections(text) if c.book == r.book][r.start : r.start + r.count]
        if not chunks:
            raise fail(ValueError("no sections in that range"))
        try:
            queued = ingest_chunks(steward, chunks)
        except llm.LLMUnavailable as e:
            raise fail(e)
        return {"chunks": [c.locator for c in chunks], "queued": len(queued)}

    @app.post("/api/query")
    def ask(a: Ask) -> dict:
        try:
            return query.ask(onto, a.question)
        except llm.LLMUnavailable as e:
            raise fail(e)

    @app.post("/api/intervene")
    def intervene(i: Intervention) -> dict:
        try:
            effects = analysis.intervene(onto, i.node, i.delta)
        except KeyError:
            raise fail(ValueError(f"unknown node {i.node}"))
        out: dict = {"node": i.node, "delta": i.delta, "effects": effects, "story": None}
        if i.retell and llm.available():
            label = onto.nodes[i.node]["label"]
            verb = "more" if i.delta > 0 else "less"
            lines = "\n".join(f"{e['label']}: {e['effect']:+.2f}" for e in effects[:12])
            facts = "\n".join(query.neighborhood_facts(onto, [i.node] + [e["id"] for e in effects[:6]]))
            out["story"] = llm.retell(f"{verb} {label} (delta {i.delta:+.1f})", lines, facts)
        return out

    @app.get("/api/investigate")
    def investigate() -> dict:
        return {"important": analysis.important_nodes(onto), "breaker": analysis.breaker(onto)}

    @app.get("/api/probes")
    def probes() -> list[dict]:
        return [p.model_dump() for p in evolver.default_probes(data_dir)]

    @app.post("/api/evolve")
    def evolve(r: EvolveReq) -> dict:
        """Search for better designs; queue them as proposals. Never applies anything."""
        probes = evolver.default_probes(data_dir)
        result = evolver.evolve(onto, probes, generations=r.generations, beam=r.beam, keep=r.keep)
        queued = evolver.queue_designs(steward, result, probes) if r.queue else []
        return {**{k: v for k, v in result.items() if k != "designs"},
                "designs": [d.summary() for d in result["designs"]],
                "queued": [p.id for p in queued]}

    @app.get("/api/metrics")
    def metrics() -> dict:
        return analysis.metrics(onto)

    @app.get("/api/archetype")
    def archetype(focus: str | None = None) -> dict:
        """The archetype the Mirror (or one node's neighbourhood) most resembles. Read-only."""
        try:
            return archetypes.match(onto.graph_snapshot(), focus=focus)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no such node: {focus}") from None

    from . import chainer, refactor

    chainer.register(app, steward)

    def _pending_refactor_keys() -> set:
        return {p.extra.get("key") for p in steward.list("proposed") if p.kind == "refactor"}

    @app.get("/api/refactor")
    def refactor_suggest() -> dict:
        """The Refactorer's suggestions for the current graphs. Nothing is queued."""
        result = refactor.suggest(onto.graph_snapshot())
        pending = _pending_refactor_keys()
        return {"base_version": result.base_version, "metrics": result.metrics,
                "findings": [vars(f) for f in result.findings],
                "patches": [{**p.model_dump(), "queued": p.extra["key"] in pending} for p in result.patches]}

    @app.post("/api/refactor/queue")
    def refactor_queue(r: RefactorQueueReq) -> dict:
        """Queue suggestions by key (or all of them) for review. Never applies anything."""
        pending = _pending_refactor_keys()
        queued = []
        for patch in refactor.suggest(onto.graph_snapshot()).patches:
            key = patch.extra["key"]
            if key in pending or (r.keys is not None and key not in r.keys):
                continue
            queued.append(steward.propose(patch).id)
        steward.save()
        return {"queued": queued}

    @app.get("/api/chain/candidates")
    def chain_candidates() -> list[dict]:
        """Pairs of nodes that look like one concept, for the Merge button."""
        label = lambda n: onto.nodes[n]["label"]
        return [{"survivor": s, "absorbed": a, "survivor_label": label(s), "absorbed_label": label(a), "why": why}
                for s, a, why in chainer.merge_candidates(onto.graph_snapshot())]

    return app
