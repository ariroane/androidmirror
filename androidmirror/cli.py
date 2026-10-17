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

    mu = sub.add_parser("mutate", help="merge or split concepts and re-evaluate both graphs; applies only what you approve",
                        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""examples:
  mutate propose                              find merges and splits the Mirror points to, with previews
  mutate merge anger wrath                    preview merging wrath into anger
  mutate merge anger calm --opposite calm     calm is anger's opposite: merged at -1
  mutate split pleasure                       split along the evidence (called good and bad, ...)
  mutate split pleasure --sense "joy: joy, gladness" --sense "pleasure: pleasure, delight"
  mutate split pleasure --claude              Claude reads the senses (one call per 50 passages)
  mutate approve m-1a2b3c4d                   apply it; mutate undo takes the last one back""")
    mu.add_argument("action", choices=["propose", "list", "show", "merge", "split", "approve", "reject", "undo"])
    mu.add_argument("args", nargs="*", help="concept ids (merge, split) or a proposal id (show, approve, reject)")
    mu.add_argument("--corpus", default="meditations")
    mu.add_argument("--into", help="merge: the concept the others merge into (default: the most read)")
    mu.add_argument("--opposite", action="append", default=[], help="merge: a concept that is the other's opposite")
    mu.add_argument("--label", help="merge: a new label for the merged concept")
    mu.add_argument("--sense", action="append", default=[], help='split: "label: word, word" (its words in the corpus)')
    mu.add_argument("--claude", action="store_true", help="split: have Claude read the senses")
    mu.add_argument("--by", default="user (command line)", help="approve, reject, undo: who decides")
    mu.add_argument("--limit", type=int, default=8, help="propose: how many candidates")

    qu = sub.add_parser("question", help="answer the Question engine's questions; answers become proposals you approve")
    qu.add_argument("--corpus", default="meditations")
    qu.add_argument("--limit", type=int, default=10, help="questions to ask (default 10)")
    qu.add_argument("--kind", help="only one kind: tension, sign, open, weak, link, conflict, evaluate, want, basis")
    qu.add_argument("--list", action="store_true", help="only list the questions")
    qu.add_argument("--pending", action="store_true", help="list proposals waiting for approval")
    qu.add_argument("--approve", metavar="ID", help="approve a proposal (needs --by)")
    qu.add_argument("--reject", metavar="ID", help="reject a proposal")
    qu.add_argument("--revoke", metavar="ID", help="take an approved answer back out")
    qu.add_argument("--by", default="", help="your name, for --approve")

    t = sub.add_parser("transmutate", help="combine two built Mirrors and trim to what is strong (writes only on approve)")
    t.add_argument("action", choices=["propose", "approve"])
    t.add_argument("a", help="a built Mirror (a folder in data/mirror/)")
    t.add_argument("b", help="the Mirror to combine with it")
    t.add_argument("--claude", action="store_true", help="propose: also have Claude pair concepts by meaning (1 call)")
    t.add_argument("--pair", action="append", metavar="A=B", help="propose: pair A's concept with B's")
    t.add_argument("--opposite", action="append", metavar="A=B", help="propose: B's concept is the opposite of A's")
    t.add_argument("--drop", action="append", metavar="CONCEPT", help="propose: leave this concept (or A=B pair) unpaired")
    t.add_argument("--share", type=float, help="propose: share of passages a fitness sample reads (default 0.5)")
    t.add_argument("--samples", type=int, help="propose: fitness samples (default 50)")
    t.add_argument("--keep-at", type=float, help="propose: survival rate below which an item is trimmed (default 0.6)")
    t.add_argument("--keep", action="append", metavar="N", help="approve: spare trim N (number or id) from the proposal")

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
              f"{st['rejected']} in tension ({st.get('common_causes', 0)} drawn as shared causes), {st['denials_held']} denials; held out: {h['positive']} {h['negative']}")
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
    elif args.cmd == "mutate":
        _mutate(args)
    elif args.cmd == "question":
        from .mirror import question

        root = mirror_dir(args.data, CorpusSpec.load(args.corpus).id)
        try:
            if args.pending:
                for p in question.proposals(root, "proposed"):
                    print(f"{p['id']}  {p['question']}\n    answer: {p['answer']}\n    if approved: {p['preview']['summary']}")
            elif args.approve:
                question.approve(root, args.approve, args.by)
                print("approved; the Mirror is re-fitted")
            elif args.reject:
                question.reject(root, args.reject)
            elif args.revoke:
                question.revoke(root, args.revoke)
                print("revoked; the Mirror is re-fitted")
            else:
                question.cli(root, limit=args.limit, kind=args.kind, listing=args.list)
        except (KeyError, ValueError) as e:
            raise SystemExit(str(e)) from None
    elif args.cmd == "transmutate":
        from .mirror import transmutate

        transmutate.cli(args, args.data)
    elif args.cmd == "ask":
        m = Mirror.load(mirror_dir(args.data, CorpusSpec.load(args.corpus).id))
        if m is None:
            raise SystemExit("no Mirror built yet: python -m androidmirror build")
        r = oracle.ask(m, args.question)
        print(r["answer"] or "\n".join(r["facts"]))
        for u in r["uses"]:
            print(f"\nWay to use this: change {u['lever']} to move {u['outcome']}\n  {u['how']}\n  Assumes: {u['assumes']}"
                  f"\n  Based on {' '.join(u['citations'])}")


def _mutate(args: argparse.Namespace) -> None:
    from .mirror import mutate

    eng = mutate.Engine(mirror_dir(args.data, CorpusSpec.load(args.corpus).id))
    a, rest = args.action, args.args
    try:
        if a == "propose":
            found = eng.propose(args.limit)
            for p in found:
                print("\n".join(mutate.describe(p)) + "\n")
            print(f"{len(found)} proposals; nothing has changed. mutate show ID adds the held-out check; "
                  "mutate approve ID applies one" if found
                  else "nothing to propose")
        elif a == "list":
            for p in eng.list():
                print(f"{p['id']}  [{p['status']}]  {p['preview']['summary']}")
        elif a == "show":  # with the held-out check, if it has not been run yet
            p = eng.get(rest[0])
            print("\n".join(mutate.describe(eng.check(rest[0]) if p["preview"]["causal"]["holdout_after"] is None else p)))
        elif a == "merge":
            print("\n".join(mutate.describe(eng.propose_merge(rest, args.into, args.opposite, args.label))))
        elif a == "split":
            senses = []
            for s in args.sense:
                label, _, words = s.partition(":")
                senses.append({"label": label.strip(), "aliases": [w.strip() for w in words.split(",") if w.strip()]})
            if args.claude and not llm.available():
                raise SystemExit("--claude needs ANTHROPIC_API_KEY in your local .env")
            print("\n".join(mutate.describe(eng.propose_split(rest[0], senses or None, claude=args.claude))))
        elif a == "approve":
            r = eng.approve(rest[0], args.by)
            print({"applied": "applied.", "stale": "the Mirror changed since; recomputed, review it and approve again:",
                   "invalid": "no longer applies:"}[r["status"]])
            if r["status"] != "applied":
                print("\n".join(mutate.describe(r["proposal"])))
        elif a == "reject":
            eng.reject(rest[0], args.by)
            print("rejected; propose will not bring it back")
        elif a == "undo":
            r = eng.undo(args.by)
            print(f"undid {r['id']}" + ("" if r["how"] == "restored" else
                                         "; the Mirror was rebuilt since, so run build again (cached, free) to drop it"))
    except IndexError:
        raise SystemExit(f"mutate {a} needs {'concept ids' if a in ('merge', 'split') else 'a proposal id'}") from None
    except (mutate.MutationError, FileNotFoundError, KeyError, llm.LLMUnavailable) as e:
        raise SystemExit(f"mutate {a}: {e}") from None
