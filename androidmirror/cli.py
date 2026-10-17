"""Command line: `python -m androidmirror <command>`.

The commands build and serve a Mirror from a whole corpus (Prototype 3). The paused
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
from .mirror.build import build, mirror_dir, refit
from .mirror.corpus import CorpusSpec
from .mirror.store import Mirror
from .server import ROOT

DATA = ROOT / "data"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["p1"]:
        from .p1_cli import main as p1_main

        return p1_main(argv[1:])

    ap = argparse.ArgumentParser(
        prog="androidmirror", description="Android Mirror, Prototype 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""usual order:
  build --corpus meditations --limit 30   a cheap trial (about 2 Claude calls per passage)
  serve                                   the web app on http://localhost:8000
  build --corpus meditations              the whole corpus; cached passages cost nothing
  causal --min-weight 2                   re-fit the causal model with other settings, free

free: causal, derange, benchmark (without --claude), serve. Full guide: docs/commands.md""")
    ap.add_argument("--data", type=Path, default=DATA, help="data directory (default: ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="have Claude read the whole corpus and build the Mirror")
    b.add_argument("--corpus", default="meditations", help="a spec in data/corpora/ (meditations, fomc) or a JSON path")
    b.add_argument("--limit", type=int, help="only the first N passages (a cheap trial run)")
    b.add_argument("--workers", type=int, default=6, help="Claude calls in parallel")
    b.add_argument("--model", help=f"Claude model for reading (default {llm.DEFAULT_MODEL})")

    c = sub.add_parser("causal", help="re-fit the subject's causal model on a built Mirror (no Claude calls)")
    c.add_argument("--corpus", default="meditations")
    c.add_argument("--min-weight", type=int, help="fewest passages a judgment must be made in to count (default 1)")
    c.add_argument("--folds", type=int, help="blocks of passages for the held-out check (default 5)")

    d = sub.add_parser("derange", help="find where the worldview rests on least, under what matters most (no Claude calls)")
    d.add_argument("--corpus", default="meditations")
    d.add_argument("--top", type=int, default=10, help="findings to print per layer (default 10)")
    d.add_argument("--knockouts", type=int, help="candidate bases to knock out and re-check, per layer (default 30)")

    bm = sub.add_parser("benchmark", help="test the causal fit on a subject whose model we planted")
    bm.add_argument("--n", type=int, default=300, help="passages to generate (offline) or to have Claude write (--claude)")
    bm.add_argument("--seeds", type=int, default=5)
    bm.add_argument("--claude", action="store_true", help="have Claude write the passages and run the real reading pipeline")

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
        build(spec, args.data, limit=args.limit, workers=args.workers)
    elif args.cmd == "causal":
        settings = {k: v for k, v in (("min_weight", args.min_weight), ("folds", args.folds)) if v is not None}
        try:
            mirror = refit(args.data, CorpusSpec.load(args.corpus).id, settings)
        except (RuntimeError, FileNotFoundError) as e:
            raise SystemExit(str(e)) from None
        st, h = mirror["causal"]["stats"], mirror["causal"]["holdout"]
        print(f"{st['held_constraints']} judgments held by the subject, {st['arrows']} arrows ({st['open']} open), "
              f"{st['rejected']} in tension, {st['denials_held']} denials; held out: {h['positive']} {h['negative']}")
    elif args.cmd == "derange":
        from .mirror import derange

        root = mirror_dir(args.data, CorpusSpec.load(args.corpus).id)
        m = Mirror.load(root)
        if m is None:
            raise SystemExit("no Mirror built yet: python -m androidmirror build")
        settings = {"knockouts": args.knockouts} if args.knockouts else None
        derange.print_report(derange.report(root, m, settings), args.top)
    elif args.cmd == "benchmark":
        from .mirror import benchmark

        if args.claude:
            if not llm.available():
                raise SystemExit("--claude needs ANTHROPIC_API_KEY in your local .env")
            benchmark.run_with_claude(args.data, n=args.n)
        else:
            benchmark.print_report(benchmark.run_offline(n=args.n, seeds=args.seeds))
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
