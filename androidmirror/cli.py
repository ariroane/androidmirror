"""Command line: `python -m androidmirror <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import analysis, corpus, llm
from .ingest import ingest_chunks, load_seed
from .ontology import Ontology
from .server import ROOT
from .steward import Steward

DATA = ROOT / "data"


def _steward(data_dir: Path) -> Steward:
    return Steward(Ontology.load(data_dir / "ontology.json"), data_dir, corpus_text=corpus.load_text(data_dir))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="androidmirror", description="Android Mirror prototype")
    ap.add_argument("--data", type=Path, default=DATA, help="data directory (default: ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--port", type=int, default=8000)

    sub.add_parser("download", help="download the Meditations text")

    i = sub.add_parser("ingest", help="have Claude read sections and propose patches")
    i.add_argument("--book", type=int, default=1)
    i.add_argument("--start", type=int, default=0)
    i.add_argument("--count", type=int, default=3)

    sd = sub.add_parser("seed", help="queue hand-authored patches from a JSON file")
    sd.add_argument("file", type=Path, nargs="?", default=DATA / "seed_book1.json")

    p = sub.add_parser("pending", help="list proposed patches")
    p.add_argument("--status", default="proposed")

    a = sub.add_parser("approve", help="approve patches by id (or 'all')")
    a.add_argument("ids", nargs="+")
    a.add_argument("--as", dest="approver", required=True, help="your name, recorded on the approval")

    from .breaker_api import add_cli as add_breaker_cli, run_cli as run_breaker_cli

    add_breaker_cli(sub)
    sub.add_parser("investigate", help="print important nodes and Breaker findings")

    args = ap.parse_args(argv)
    data_dir: Path = args.data

    if args.cmd == "serve":
        import uvicorn

        from .server import create_app

        print(f"Android Mirror on http://localhost:{args.port}  (Claude: {'on' if llm.available() else 'off, no key'})")
        uvicorn.run(create_app(data_dir), host="127.0.0.1", port=args.port)
    elif args.cmd == "download":
        print(corpus.download(data_dir))
    elif args.cmd == "ingest":
        st = _steward(data_dir)
        text = corpus.download(data_dir).read_text(encoding="utf-8", errors="replace")
        st = Steward(st.ontology, data_dir, corpus_text=text)
        chunks = [c for c in corpus.split_sections(text) if c.book == args.book][args.start : args.start + args.count]
        queued = ingest_chunks(st, chunks)
        print(f"read {', '.join(c.locator for c in chunks)}; queued {len(queued)} patches")
    elif args.cmd == "seed":
        queued = load_seed(_steward(data_dir), args.file)
        print(f"queued {len(queued)} patches from {args.file}")
    elif args.cmd == "pending":
        for patch in _steward(data_dir).list(args.status):
            ops = ", ".join(o.op for o in patch.ops)
            warn = f"  [{'; '.join(patch.warnings)}]" if patch.warnings else ""
            print(f"{patch.id}  {patch.source.locator:<14} {patch.rationale[:70]}  ({ops}){warn}")
    elif args.cmd == "approve":
        st = _steward(data_dir)
        ids = [p.id for p in st.list("proposed")] if args.ids == ["all"] else args.ids
        for pid in ids:
            st.approve(pid, args.approver)
        print(f"approved {len(ids)}; ontology is now version {st.ontology.version}")
    elif run_breaker_cli(args, data_dir):
        pass
    elif args.cmd == "investigate":
        onto = Ontology.load(data_dir / "ontology.json")
        print(json.dumps({"important": analysis.important_nodes(onto), "breaker": analysis.breaker(onto),
                          "metrics": analysis.metrics(onto)}, indent=2))
