"""Command line: `python -m androidmirror <command>`.

Prototype 2 commands build and serve a Mirror from a whole corpus. The paused
Prototype 1 app (patch review and the four engines) is still available as
`python -m androidmirror p1 <command>`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import llm
from .mirror import ask as oracle
from .mirror.build import build, mirror_dir, rediscover
from .mirror.corpus import CorpusSpec
from .mirror.store import Mirror
from .server import ROOT

DATA = ROOT / "data"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["p1"]:
        from .p1_cli import main as p1_main

        return p1_main(argv[1:])

    ap = argparse.ArgumentParser(prog="androidmirror", description="Android Mirror, Prototype 2")
    ap.add_argument("--data", type=Path, default=DATA, help="data directory (default: ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="have Claude read the whole corpus and build the Mirror")
    b.add_argument("--corpus", default="meditations", help="a spec in data/corpora/ (meditations, fomc) or a JSON path")
    b.add_argument("--limit", type=int, help="only the first N passages (a cheap trial run)")
    b.add_argument("--workers", type=int, default=6, help="Claude calls in parallel")
    b.add_argument("--model", help=f"Claude model for coding (default {llm.DEFAULT_MODEL})")
    b.add_argument("--no-measure", action="store_true",
                   help="skip the second pass; measure variables from the first pass's mentions")

    c = sub.add_parser("causal", help="re-run causal discovery on a built Mirror (no Claude calls)")
    c.add_argument("--corpus", default="meditations")
    c.add_argument("--alpha", type=float)
    c.add_argument("--depth", type=int, help="largest conditioning set")
    c.add_argument("--max-vars", type=int)
    c.add_argument("--min-support", type=int)
    c.add_argument("--bootstrap", type=int, help="resamples for edge stability (0 to skip)")
    c.add_argument("--no-claims", action="store_true", help="do not use stated claims as background knowledge")
    c.add_argument("--tiers", type=Path, help="JSON list of lists of variable ids, earliest tier first")

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--port", type=int, default=8000)

    q = sub.add_parser("ask", help="ask the Mirror a question")
    q.add_argument("question")
    q.add_argument("--corpus", default="meditations")

    sub.add_parser("p1", help="the paused Prototype 1 app: python -m androidmirror p1 --help")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        spec = CorpusSpec.load(args.corpus)
        if not llm.available():
            raise SystemExit("building needs Claude: put ANTHROPIC_API_KEY in your local .env")
        if args.model:
            llm.DEFAULT_MODEL = args.model
        build(spec, args.data, limit=args.limit, workers=args.workers, measure=not args.no_measure)
    elif args.cmd == "causal":
        settings: dict = {}
        if args.alpha is not None:
            settings["alpha"] = args.alpha
        if args.depth is not None:
            settings["max_depth"] = args.depth
        if args.bootstrap is not None:
            settings["bootstrap"] = args.bootstrap
        if args.no_claims:
            settings["claims_as_pins"] = False
        if args.tiers:
            settings["tiers"] = json.loads(args.tiers.read_text(encoding="utf-8"))
        sel = {k: v for k, v in (("max_vars", args.max_vars), ("min_support", args.min_support)) if v is not None}
        if sel:
            settings["selection"] = sel
        mirror = rediscover(args.data, CorpusSpec.load(args.corpus).id, settings)
        c = mirror["causal"]
        size = c.get("class_size", {})
        print(f"{len(c.get('variables', []))} variables, {len(c.get('edges', []))} edges; class size "
              f"{size.get('data', {}).get('count')} from the data, {size.get('with_pins', {}).get('count')} with pins")
    elif args.cmd == "serve":
        import uvicorn

        from .server import create_app

        print(f"Android Mirror on http://localhost:{args.port}  (Claude: {'on' if llm.available() else 'off, no key'})")
        uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port)
    elif args.cmd == "ask":
        m = Mirror.load(mirror_dir(args.data, CorpusSpec.load(args.corpus).id))
        if m is None:
            raise SystemExit("no Mirror built yet: python -m androidmirror build")
        r = oracle.ask(m, args.question)
        print(r["answer"] or "\n".join(r["facts"]))
