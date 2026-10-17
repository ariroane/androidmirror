"""The Question engine: ask the user what the Mirror's structures leave open, and turn answers into proposals.

The engine reads a built Mirror for the places where the subject's words do
not settle the model, and asks about each in turn, most important first:

causal model
- tension: a judgment the fit rejected (it would close a loop, break a
  denial, or is outweighed); which side does the subject hold?
- sign: the subject says x raises y in some passages and lowers it in others;
- open: an arrow the rest of the model also explains; direct, or only through?
- weak: an arrow only one passage makes; does the subject really hold it?
- link: two concepts the knowledge graph relates (x requires y, x opposes y,
  x expresses y) or two of the most central ones, with no causal judgment
  between them; does one make a difference to the other?

knowledge graph
- conflict: an inconsistency the reasoner found (a thing both good and bad,
  a denied fact that follows, valued and rejected, each a kind of the other);
- evaluate: a central disposition, emotion or practice the subject never
  calls good, bad or indifferent;

values
- want: the subject urges more of something in some passages and less in others.

from the Derange engine
- basis: a weak basis Derange found (an arrow or fact said in few passages that
  much of the important subgraphs rests on), read from its saved report
  (derange.json) when that report was made from the Mirror as it is now.

Each question is built from the structures alone (no Claude call), and each
option lists the exact changes it would make. Answering never changes the
Mirror: it records a proposal, with a preview of what the fit would do with
it. Only `approve`, with the approver's name, adds it to the Mirror; `revoke`
takes it out again. Free-text answers are read into an option by Claude when a
key is present.

Importance: a concept's weight is log(1 + passages that mention it) plus
log(1 + passages behind its causal judgments); a question's score is its
kind's base times the mean weight of the concepts it is about, times
log(2 + the passages on the line). Unsettled questions about the concepts the
subject returns to most come first; each further question of a kind already
asked has its score divided by 1 + 0.5 x (questions of that kind before it).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .. import llm
from . import answers
from .causal import fit
from .kg import KnowledgeBase, fact_key
from .schema import EVALUATIONS, SELF, VARIABLE_KINDS
from .store import Mirror

BASE = {"tension": 3.0, "conflict": 3.0, "sign": 2.5, "open": 2.0, "want": 1.5, "link": 1.5, "weak": 1.2,
        "evaluate": 0.8, "basis": 2.5}
SKIP = {"key": "skip", "label": "Not sure / skip", "changes": []}
KG_LINKS = {  # relation -> (ask about the pair which way round, what the relation hints)
    "requires": ("o", "s"), "opposes": ("s", "o"), "expresses": ("o", "s"), "accords_with": ("s", "o"),
}
MAX_PER_KIND = 25


def _qid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


def _plural(n: int, word: str = "passage") -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


class Asker:
    """Builds the questions for one Mirror."""

    def __init__(self, m: Mirror) -> None:
        self.m = m
        self.subject = m.meta["corpus"]["subject"]
        weight: Counter = Counter()
        for c in m.causal.get("constraints", []):
            if c["held"]:
                for n in (c["x"], c["y"], c["z"]):
                    if n:
                        weight[n] += c["weight"]
        self.imp = {n: math.log1p(m.nodes.get(n, {}).get("support", 0)) + math.log1p(weight[n]) for n in m.nodes}

    def L(self, n: str) -> str:
        return self.m.label(n)

    def q(self, kind: str, key: str, text: str, about: list[str], options: list[dict[str, Any]], why: str,
          evidence: list[dict[str, Any]], passages: int = 1) -> dict[str, Any]:
        about = [n for n in dict.fromkeys(about) if n]
        mean = sum(self.imp.get(n, 0) for n in about) / max(1, len(about))
        return {"id": _qid(kind, key), "kind": kind, "text": text, "about": about,
                "labels": {n: self.L(n) for n in about}, "why": why, "evidence": evidence[:4],
                "options": options + [SKIP], "score": round(BASE[kind] * (0.5 + mean) * math.log(2 + passages), 3)}

    def quotes(self, c: dict[str, Any]) -> list[dict[str, Any]]:
        return [{k: e[k] for k in ("passage", "locator", "quote")} for e in c.get("evidence", []) if e.get("held")]

    # ---------- causal ----------

    def tensions(self) -> list[dict[str, Any]]:
        out = []
        cons = self.m.constraint
        for c in self.m.causal.get("constraints", []):
            if c["status"] != "rejected" or not c["held"] or c.get("pinned"):
                continue
            others = [cons[i] for i in c.get("conflicts", []) if i in cons]
            if not others:
                continue
            said = self.m.say(c)
            against = "; ".join(self.m.say(o) for o in others)
            how = {"cycle": "it would close a loop with", "denied": "it breaks the denial",
                   "outweighed": "it is outweighed by"}.get(c.get("reason"), "it clashes with")
            text = (f"{self.subject} says {said} ({_plural(c['weight'])}), but {how}: {against}. "
                    f"Which does {self.subject} hold?")
            opts = [{"key": "this", "label": said, "changes": [{"type": "ruling", "target": c["id"], "ruling": "affirm",
                                                                "says": f"{said} (settled)"}]},
                    {"key": "other", "label": against,
                     "changes": [{"type": "ruling", "target": c["id"], "ruling": "retract", "says": f"not: {said}"}]
                     + [{"type": "ruling", "target": o["id"], "ruling": "affirm", "says": f"{self.m.say(o)} (settled)"}
                        for o in others]}]
            if c.get("reason") == "cycle":
                opts.append({"key": "neither", "label": "Neither: one of them is a misreading",
                             "changes": [{"type": "ruling", "target": c["id"], "ruling": "retract", "says": f"not: {said}"}]})
            ev = self.quotes(c) + [e for o in others for e in self.quotes(o)[:1]]
            out.append(self.q("tension", c["id"], text, [c["x"], c["y"], c["z"]] + [o["x"] for o in others], opts,
                              "the fit had to drop one of the subject's judgments to stay consistent", ev,
                              c["weight"] + sum(o["weight"] for o in others)))
        return out

    def signs(self) -> list[dict[str, Any]]:
        out = []
        for t in self.m.causal.get("tensions", []):
            c = self.m.constraint.get(t["id"])
            if not c or c.get("pinned") or c.get("ruling"):
                continue
            x, y = self.L(t["x"]), self.L(t["y"])
            plus, minus = t["signs"].get("1", 0), t["signs"].get("-1", 0)
            text = (f"{self.subject} says {x} raises {y} in {_plural(plus)} and lowers it in {_plural(minus)}. "
                    f"Which is it?")
            opts = [{"key": "raises", "label": f"{x} raises {y}",
                     "changes": [{"type": "ruling", "target": t["id"], "ruling": "affirm", "sign": 1,
                                  "says": f"{x} raises {y}"}]},
                    {"key": "lowers", "label": f"{x} lowers {y}",
                     "changes": [{"type": "ruling", "target": t["id"], "ruling": "affirm", "sign": -1,
                                  "says": f"{x} lowers {y}"}]}]
            out.append(self.q("sign", t["id"], text, [t["x"], t["y"]], opts,
                              "the passages disagree on the direction", self.quotes(c), plus + minus))
        return out

    def opens(self) -> list[dict[str, Any]]:
        out = []
        for e in self.m.causal.get("edges", []):
            c = self.m.constraint.get(e["constraint"])
            if not e["open"] or not c or c.get("pinned"):
                continue
            x, y = self.L(e["src"]), self.L(e["dst"])
            via = " → ".join(self.L(n) for n in e["open"]["through"])
            verb = "raises" if e["sign"] > 0 else "lowers"
            text = f"{self.subject} says {x} {verb} {y}. Directly, or only by way of {via}?"
            opts = [{"key": "direct", "label": f"Directly (whatever else it does through {via})",
                     "changes": [{"type": "ruling", "target": c["id"], "ruling": "affirm", "direct": True,
                                  "says": f"{x} {verb} {y} directly"}]},
                    {"key": "through", "label": f"Only by way of {via}",
                     "changes": [{"type": "ruling", "target": c["id"], "ruling": "affirm", "indirect": True,
                                  "says": f"{x} {verb} {y} only through {via}"}]}]
            out.append(self.q("open", c["id"], text, [e["src"], e["dst"]] + e["open"]["through"], opts,
                              "the subject's other judgments also explain it, so the arrow may be a shortcut",
                              self.quotes(c), e["weight"]))
        return out

    def weak(self) -> list[dict[str, Any]]:
        out = []
        for e in self.m.causal.get("edges", []):
            c = self.m.constraint.get(e["constraint"])
            if e["weight"] != 1 or e["open"] or not c or c.get("pinned"):
                continue
            said = self.m.say(c)
            text = f"Only one passage says {said}. Does {self.subject} hold it?"
            opts = [{"key": "yes", "label": "Yes, it holds",
                     "changes": [{"type": "ruling", "target": c["id"], "ruling": "affirm", "says": f"{said} (settled)"}]},
                    {"key": "no", "label": "No, that reads too much into the passage",
                     "changes": [{"type": "ruling", "target": c["id"], "ruling": "retract", "says": f"not: {said}"}]}]
            out.append(self.q("weak", c["id"], text, [e["src"], e["dst"]], opts,
                              "an arrow resting on a single passage is the easiest to misread", self.quotes(c), 1))
        return out

    def links(self) -> list[dict[str, Any]]:
        m = self.m
        said = set()
        for c in m.causal.get("constraints", []):
            if c["held"] or c.get("ruling"):
                said |= {(c["x"], c["y"]), (c["y"], c["x"])}
                if c["z"]:
                    said |= {(c["x"], c["z"]), (c["z"], c["x"]), (c["y"], c["z"]), (c["z"], c["y"])}
        g = m.graph_

        def eligible(n: str) -> bool:
            node = m.nodes.get(n, {})
            return n not in (SELF, *EVALUATIONS) and not n.endswith("~") and (
                node.get("variable") or node.get("kind") in VARIABLE_KINDS or n in m.variables)

        def unsaid(a: str, b: str) -> bool:
            return a != b and (a, b) not in said and not g.reaches(a, b) and not g.reaches(b, a)

        out, seen = [], set()
        for f in sorted(m.facts, key=lambda f: -f["support"]):
            if not f["held"] or f["neg"] or f["r"] not in KG_LINKS:
                continue
            first, second = KG_LINKS[f["r"]]
            a, b = f[first], f[second]
            if not (eligible(a) and eligible(b) and unsaid(a, b)) or frozenset((a, b)) in seen:
                continue
            seen.add(frozenset((a, b)))
            hint = f"{self.subject} says {self.L(f['s'])} {f['r'].replace('_', ' ')} {self.L(f['o'])}"
            out.append(self._link(a, b, f"{hint}. ", "the knowledge graph relates them, the causal model does not",
                                  [{k: e[k] for k in ("passage", "locator", "quote")} for e in f["evidence"][:2]],
                                  f["support"]))
        central = sorted((n for n in m.nodes if eligible(n)), key=lambda n: -self.imp.get(n, 0))[:8]
        for a, b in combinations(central, 2):
            if unsaid(a, b) and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                out.append(self._link(a, b, "", "two of the concepts the subject returns to most, with no judgment "
                                      "between them", [], 1))
        return out

    def _link(self, a: str, b: str, lead: str, why: str, ev: list[dict[str, Any]], n: int) -> dict[str, Any]:
        x, y = self.L(a), self.L(b)
        text = f"{lead}Does {x} make a difference to {y}, in {self.subject}'s view?"

        def j(kind: str, sign: int, says: str) -> list[dict[str, Any]]:
            return [{"type": "judgment", "kind": kind, "x": a, "y": b, "sign": sign, "scope": "overall", "says": says}]

        opts = [{"key": "raises", "label": f"More {x} means more {y}", "changes": j("causes", 1, f"{x} raises {y}")},
                {"key": "lowers", "label": f"More {x} means less {y}", "changes": j("causes", -1, f"{x} lowers {y}")},
                {"key": "none", "label": f"{x} makes no difference to {y}",
                 "changes": j("independent", 0, f"{x} makes no difference to {y}")},
                {"key": "reverse", "label": f"The other way round: {y} moves {x}",
                 "changes": [{"type": "judgment", "kind": "causes", "x": b, "y": a, "sign": 1, "scope": "overall",
                              "says": f"{y} raises {x}"}]}]
        return self.q("link", f"{a}|{b}", text, [a, b], opts, why, ev, n)

    # ---------- knowledge graph ----------

    def weakest(self, t: tuple[str, str, str]) -> dict[str, Any] | None:
        """The least-supported stated fact a closure fact rests on (on a tie, the one about its subject)."""
        leaves: list[dict[str, Any]] = []

        def walk(p: dict[str, Any]) -> None:
            if p.get("rule") == "stated":
                leaves.append(p)
            for q in p.get("from", []):
                walk(q)

        walk(self.m.kb.explain(t))
        leaves = [x for x in leaves if x["fact"] in {fact_key(f["s"], f["r"], f["o"]) for f in self.m.facts}]
        if not leaves:
            return None
        return min(leaves, key=lambda x: (x.get("support", 0), x["s"] != t[0], x["fact"]))

    def drop(self, t: tuple[str, str, str]) -> list[dict[str, Any]]:
        w = self.weakest(t)
        if not w:
            return []
        return [{"type": "retract_fact", "key": w["fact"],
                 "says": f"no longer held: {self.L(w['s'])} {w['r'].replace('_', ' ')} {self.L(w['o'])}"}]

    def conflicts(self) -> list[dict[str, Any]]:
        out = []
        S = self.subject
        for v in self.m.kb.violations:
            t = v["type"]
            if t == "disjoint_evaluations":
                n = v["node"]
                x = self.L(n)
                opts = []
                for e in v["evaluations"]:
                    ch = [{"type": "fact", "s": n, "r": "is_a", "o": e, "says": f"{x} is {e}"}]
                    for other in v["evaluations"]:
                        if other != e:
                            ch += self.drop((n, "is_a", other))
                    opts.append({"key": e, "label": f"{x} is {e}", "changes": ch})
                text = f"The Mirror finds {S} calling {x} {' and '.join(v['evaluations'])}. Which is it?"
                out.append(self.q("conflict", f"eval|{n}", text, [n], opts, "the evaluations good, bad and "
                                  "indifferent exclude each other", self._fact_quotes(v["facts"]), 2))
            elif t == "denied_but_follows":
                s, r, o = v["fact"].split("|")
                said = f"{self.L(s)} {r.replace('_', ' ')} {self.L(o)}"
                opts = [{"key": "holds", "label": f"It holds: {said}",
                         "changes": [{"type": "retract_fact", "key": fact_key(s, r, o, True),
                                      "says": f"no longer denied: {said}"}]},
                        {"key": "denied", "label": "It does not: the denial stands",
                         "changes": self.drop((s, r, o))}]
                text = f"{S} denies that {said}, yet it follows from other things {S} says. Which holds?"
                out.append(self.q("conflict", f"denied|{v['fact']}", text, [s, o], opts,
                                  "a denied fact that the reasoner derives anyway", self._fact_quotes(v["facts"]), 2))
            elif t == "values_and_rejects":
                n = v["node"]
                x = self.L(n)
                opts = [{"key": "values", "label": f"{S} values {x}", "changes": self.drop((v["agent"], "rejects", n))},
                        {"key": "rejects", "label": f"{S} rejects {x}", "changes": self.drop((v["agent"], "values", n))}]
                text = f"The Mirror finds {self.L(v['agent'])} both valuing and rejecting {x}. Which is it?"
                out.append(self.q("conflict", f"vr|{v['agent']}|{n}", text, [n], opts, "values and rejects exclude "
                                  "each other", self._fact_quotes(v["facts"]), 2))
            elif t == "taxonomy_cycle":
                a, b = v["nodes"]
                x, y = self.L(a), self.L(b)
                opts = [{"key": "same", "label": f"{x} and {y} are the same thing",
                         "changes": [{"type": "fact", "s": a, "r": "same_as", "o": b, "says": f"{x} is another name for {y}"}]},
                        {"key": "a_in_b", "label": f"{x} is a kind of {y}", "changes": self.drop((b, "is_a", a))},
                        {"key": "b_in_a", "label": f"{y} is a kind of {x}", "changes": self.drop((a, "is_a", b))}]
                text = f"The Mirror finds {x} a kind of {y} and {y} a kind of {x}. Which is it?"
                out.append(self.q("conflict", f"tax|{a}|{b}", text, [a, b], opts, "each is a kind of the other",
                                  self._fact_quotes(v["facts"]), 2))
        return [q for q in out if all(o["changes"] for o in q["options"] if o["key"] != "skip")]

    def _fact_quotes(self, keys: list[str]) -> list[dict[str, Any]]:
        by = {fact_key(f["s"], f["r"], f["o"], f["neg"]): f for f in self.m.facts}
        out = []
        for k in keys:
            proof = by.get(k)
            if proof is None and k.count("|") == 2:
                for leaf in self._leaves(tuple(k.split("|"))):
                    if leaf in by:
                        out += by[leaf]["evidence"][:1]
            elif proof:
                out += proof["evidence"][:1]
        return [{k: e[k] for k in ("passage", "locator", "quote")} for e in out if e.get("passage") != "codebook"]

    def _leaves(self, t: tuple) -> list[str]:
        out: list[str] = []

        def walk(p: dict[str, Any]) -> None:
            if p.get("rule") == "stated":
                out.append(p["fact"])
            for q in p.get("from", []):
                walk(q)

        walk(self.m.kb.explain(t))
        return out

    def evaluations(self) -> list[dict[str, Any]]:
        out = []
        for n, node in self.m.nodes.items():
            if node.get("kind") not in ("disposition", "emotion", "practice", "state", "belief") or n.endswith("~"):
                continue
            if node.get("support", 0) < 2 or self.m.evaluation(n) is not None:
                continue
            x = self.L(n)
            opts = [{"key": e, "label": f"{x} is {e}",
                     "changes": [{"type": "fact", "s": n, "r": "is_a", "o": e, "says": f"{x} is {e}"}]}
                    for e in EVALUATIONS]
            text = f"{self.subject} speaks of {x} in {_plural(node['support'])} but never says whether it is good, " \
                   f"bad or indifferent. Which?"
            ev = [{k: mm[k] for k in ("passage", "locator", "quote")} for mm in node.get("mentions", [])[:3]]
            out.append(self.q("evaluate", n, text, [n], opts, "an unevaluated concept the subject returns to", ev,
                              node["support"]))
        return out

    # ---------- values ----------

    def wants(self) -> list[dict[str, Any]]:
        out = []
        for v in self.m.values:
            if not (v["more"] and v["less"]) or "answered" in v:
                continue
            if min(v["more"], v["less"]) * 3 < max(v["more"], v["less"]):
                continue  # a clear majority: not worth asking
            x = self.L(v["concept"])
            text = f"{self.subject} urges more {x} in {_plural(v['more'])} and less in {_plural(v['less'])}. " \
                   f"Which does {self.subject} want?"
            opts = [{"key": "more", "label": f"More {x}", "changes": [{"type": "value", "concept": v["concept"],
                                                                         "direction": 1, "says": f"wants more {x}"}]},
                    {"key": "less", "label": f"Less {x}", "changes": [{"type": "value", "concept": v["concept"],
                                                                         "direction": -1, "says": f"wants less {x}"}]}]
            ev = [{k: e[k] for k in ("passage", "locator", "quote")} for e in v["evidence"][:3]]
            out.append(self.q("want", v["concept"], text, [v["concept"]], opts, "the passages urge both ways", ev,
                              v["more"] + v["less"]))
        return out

    # ---------- from Derange ----------

    def derange(self) -> list[dict[str, Any]]:
        """Derange's findings as questions: is the thin basis really what the subject holds?"""
        root = self.m.root
        if root is None or not (root / "derange.json").exists():
            return []
        import json

        rep = json.loads((root / "derange.json").read_text(encoding="utf-8"))
        if rep.get("mirror_mtime") != (root / "mirror.json").stat().st_mtime:
            return []  # made from an older Mirror: run Derange again
        out = []
        for f in (rep.get("causal") or {}).get("findings", []):
            basis = [b for b in f.get("basis", []) if b.get("constraint") in self.m.constraint]
            if not basis or any(self.m.constraint[b["constraint"]].get("pinned") for b in basis):
                continue
            said = "; ".join(b["text"] for b in basis)
            opts = [{"key": "yes", "label": "Yes, it holds",
                     "changes": [{"type": "ruling", "target": b["constraint"], "ruling": "affirm",
                                  "says": f"{b['text']} (settled)"} for b in basis]},
                    {"key": "no", "label": "No, that reads too much into the passages",
                     "changes": [{"type": "ruling", "target": b["constraint"], "ruling": "retract",
                                  "says": f"not: {b['text']}"} for b in basis]}]
            about = [b["src"] for b in basis] + [b["dst"] for b in basis] + [h["end"] for h in f.get("ends", [])]
            ev = [{k: e[k] for k in ("passage", "locator", "quote")} for b in basis for e in b.get("evidence", [])]
            out.append(self.q("basis", f["id"], f.get("probe") or f"Does {self.subject} hold that {said}?", about, opts,
                              f"Derange: much of what matters rests on {said}", ev, f.get("strength", 1)))
        held = {fact_key(x["s"], x["r"], x["o"]) for x in self.m.facts if x["held"] and not x.get("retracted")}
        for f in (rep.get("conceptual") or {}).get("findings", []):
            if f.get("fact") not in held:
                continue
            opts = [{"key": "yes", "label": "Yes, it holds",
                     "changes": [{"type": "fact", "s": f["s"], "r": f["r"], "o": f["o"], "says": f"{f['text']} (settled)"}]},
                    {"key": "no", "label": "No, the subject does not hold it",
                     "changes": [{"type": "retract_fact", "key": f["fact"], "says": f"no longer held: {f['text']}"}]}]
            ev = [{k: e[k] for k in ("passage", "locator", "quote")} for e in f.get("evidence", [])]
            out.append(self.q("basis", f["id"], f.get("probe") or f"Does {self.subject} hold that {f['text']}?",
                              [f["s"], f["o"]] + [h["concept"] for h in f.get("concepts", [])], opts,
                              f"Derange: much of what is held about it rests on {f['text']}", ev, f.get("strength", 1)))
        return out

    def all(self) -> list[dict[str, Any]]:
        out = []
        for make in (self.tensions, self.signs, self.opens, self.weak, self.links, self.conflicts, self.evaluations,
                     self.wants, self.derange):
            out += sorted(make(), key=lambda q: -q["score"])[:MAX_PER_KIND]
        # most important first, but each further question of a kind already asked counts for less,
        # so a session does not spend itself on one kind
        out.sort(key=lambda q: (-q["score"], q["id"]))
        ranked, seen = [], Counter()
        while out:
            best = max(range(len(out)), key=lambda i: out[i]["score"] / (1 + 0.5 * seen[out[i]["kind"]]))
            q = out.pop(best)
            seen[q["kind"]] += 1
            ranked.append(q)
        return ranked


# ---------- the engine ----------


def questions(m: Mirror, limit: int | None = 20, kinds: set[str] | None = None,
              include_answered: bool = False) -> list[dict[str, Any]]:
    """The questions the Mirror's structures raise, most important first.

    Questions with a proposal waiting or approved are left out unless `include_answered`."""
    book = answers.load(m.root) if m.root else {"proposals": []}
    busy = {p["question_id"] for p in book["proposals"] if p["status"] in ("proposed", "approved")}
    qs = [q for q in Asker(m).all() if (not kinds or q["kind"] in kinds) and (include_answered or q["id"] not in busy)]
    return qs[:limit] if limit else qs


def find(m: Mirror, question_id: str) -> dict[str, Any]:
    q = next((q for q in Asker(m).all() if q["id"] == question_id), None)
    if q is None:
        raise KeyError(f"question {question_id} (the Mirror may have changed since it was asked)")
    return q


INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {"option": {"type": "string"}, "note": {"type": "string"}},
    "required": ["option", "note"],
    "additionalProperties": False,
}

INTERPRET_SYSTEM = """You read a person's free-text answer to a question about the model the Android Mirror holds of {subject}'s views.
Pick the one option the answer means. Pick "skip" if it fits none, is unsure, or says "it depends" without choosing.
Do not answer the question yourself; only read what the person said.
note: anything in the answer the option does not capture (a condition, a caveat), in a short phrase of their own words, or "" if nothing."""


def interpret(m: Mirror, q: dict[str, Any], text: str) -> dict[str, str]:
    """Read a free-text answer into one of the question's options (one Claude call)."""
    schema = {**INTERPRET_SCHEMA, "properties": {**INTERPRET_SCHEMA["properties"],
                                                 "option": {"type": "string", "enum": [o["key"] for o in q["options"]]}}}
    opts = "\n".join(f"{o['key']}: {o['label']}" for o in q["options"])
    system = INTERPRET_SYSTEM.replace("{subject}", m.meta["corpus"]["subject"])
    r = llm._json_call(system, f"Question: {q['text']}\n\nOptions:\n{opts}\n\nAnswer: {text}", schema, effort="low",
                       max_tokens=2000)
    return {"option": r["option"], "note": r["note"]}


def _raw(root: Path) -> dict[str, Any]:
    import json

    return json.loads((root / "mirror.json").read_text(encoding="utf-8"))


def _state(raw: dict[str, Any], book: dict[str, Any]) -> dict[str, Any]:
    over = answers.overlay(raw, book)
    model = fit(over["judgments"], (raw.get("causal") or {}).get("settings"))
    kb = KnowledgeBase(over["nodes"], over["facts"])
    return {"model": model, "kb": kb, "values": {v["concept"]: v["direction"] for v in over["values"]}}


def preview(root: Path, changes: list[dict[str, Any]], book: dict[str, Any] | None = None) -> dict[str, Any]:
    """What approving `changes` would do: the arrows, judgments, evaluations and values that would differ."""
    raw = _raw(root)
    book = book or answers.load(root)
    trial = {"proposals": book["proposals"] + [{"id": "preview", "question": "", "answer": "", "status": "approved",
                                                "changes": changes}]}
    before, after = _state(raw, book), _state(raw, trial)
    label = lambda n: raw["nodes"].get(n, {}).get("label", n)  # noqa: E731
    arrows = lambda s: {(e["src"], e["dst"]): e["sign"] for e in s["model"]["edges"]}  # noqa: E731
    a0, a1 = arrows(before), arrows(after)
    word = lambda sign: "raises" if sign > 0 else "lowers"  # noqa: E731
    out: dict[str, Any] = {
        "arrows_added": [f"{label(a)} {word(s)} {label(b)}" for (a, b), s in a1.items() if (a, b) not in a0],
        "arrows_removed": [f"{label(a)} {word(s)} {label(b)}" for (a, b), s in a0.items() if (a, b) not in a1],
        "arrows_flipped": [f"{label(a)} now {word(s)} {label(b)}" for (a, b), s in a1.items() if a0.get((a, b), s) != s],
    }
    st0 = {c["id"]: c["status"] for c in before["model"]["constraints"]}
    out["judgments"] = [{"id": c["id"], "before": st0.get(c["id"], "new"), "after": c["status"]}
                        for c in after["model"]["constraints"] if st0.get(c["id"], "new") != c["status"]]
    ev0 = {n for n in raw["nodes"]}
    evals = []
    for n in ev0:
        e0 = [e for e in EVALUATIONS if (n, "is_a", e) in before["kb"].closure]
        e1 = [e for e in EVALUATIONS if (n, "is_a", e) in after["kb"].closure]
        if e0 != e1:
            evals.append(f"{label(n)}: {', '.join(e0) or 'none'} → {', '.join(e1) or 'none'}")
    out["evaluations"] = sorted(evals)
    out["inferred"] = {"before": len(before["kb"].closure), "after": len(after["kb"].closure)}
    out["inconsistencies"] = {"before": len(before["kb"].violations), "after": len(after["kb"].violations)}
    out["values"] = [f"{label(c)}: {before['values'].get(c, 0)} → {d}" for c, d in after["values"].items()
                     if before["values"].get(c, 0) != d]
    out["summary"] = _summary(out)
    return out


def _summary(p: dict[str, Any]) -> str:
    bits = []
    for k, w in (("arrows_added", "arrow added"), ("arrows_removed", "arrow removed"), ("arrows_flipped", "arrow flipped")):
        if p[k]:
            bits.append(_plural(len(p[k]), w))
    if p["judgments"]:
        bits.append(_plural(len(p["judgments"]), "judgment") + " settled differently")
    if p["evaluations"]:
        bits.append(_plural(len(p["evaluations"]), "evaluation") + " changed")
    d = p["inconsistencies"]["after"] - p["inconsistencies"]["before"]
    if d:
        bits.append(f"{abs(d)} {'fewer' if d < 0 else 'more'} inconsistenc{'y' if abs(d) == 1 else 'ies'}")
    if p["values"]:
        bits.append(_plural(len(p["values"]), "value") + " settled")
    return "; ".join(bits) or "the answer is recorded; the model's arrows and evaluations stay as they are"


def answer(m: Mirror, question_id: str, option: str | None = None, note: str = "", text: str = "") -> dict[str, Any]:
    """Record an answer as a proposal (nothing is applied). With `text` and no `option`, Claude reads the text."""
    if m.root is None:
        raise RuntimeError("this Mirror was not loaded from disk")
    q = find(m, question_id)
    if not option:
        if not text.strip():
            raise ValueError("give an option or a written answer")
        if not llm.available():
            raise llm.LLMUnavailable("reading a written answer needs Claude; pick one of the options instead")
        r = interpret(m, q, text)
        option, note = r["option"], note or r["note"]
    opt = next((o for o in q["options"] if o["key"] == option), None)
    if opt is None:
        raise ValueError(f"no option {option!r}; choose one of {[o['key'] for o in q['options']]}")
    if not opt["changes"]:
        return {"status": "skipped", "question_id": question_id}
    book = answers.load(m.root)
    for p in book["proposals"]:  # a new answer replaces one still waiting on the same question
        if p["question_id"] == question_id and p["status"] == "proposed":
            p["status"] = "superseded"
    prop = {"id": _qid(question_id, option, note, str(time.time())), "question_id": question_id, "kind": q["kind"],
            "question": q["text"], "about": q["about"], "option": option, "answer": opt["label"], "note": note,
            "written": text, "changes": opt["changes"], "status": "proposed",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    prop["preview"] = preview(m.root, opt["changes"], book)
    book["proposals"].append(prop)
    answers.save(m.root, book)
    return prop


def proposals(root: Path, status: str | None = None) -> list[dict[str, Any]]:
    return [p for p in answers.load(root)["proposals"] if status is None or p["status"] == status]


def _set(root: Path, pid: str, frm: set[str], to: str, **extra: Any) -> dict[str, Any]:
    book = answers.load(root)
    p = next((p for p in book["proposals"] if p["id"] == pid), None)
    if p is None:
        raise KeyError(f"proposal {pid}")
    if p["status"] not in frm:
        raise ValueError(f"proposal {pid} is {p['status']}")
    p.update(status=to, **extra)
    answers.save(root, book)
    return p


def approve(root: Path, pid: str, approver: str) -> dict[str, Any]:
    """The only way an answer changes the Mirror: a named person approves it, and the causal model is re-fitted."""
    if not approver.strip():
        raise ValueError("approving needs the approver's name")
    p = _set(root, pid, {"proposed"}, "approved", approved_by=approver.strip(),
             approved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _refit(root)
    return p


def reject(root: Path, pid: str) -> dict[str, Any]:
    return _set(root, pid, {"proposed"}, "rejected")


def revoke(root: Path, pid: str) -> dict[str, Any]:
    """Take an approved answer back out of the Mirror."""
    p = _set(root, pid, {"approved"}, "revoked", revoked_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _refit(root)
    return p


def _refit(root: Path) -> None:
    from .build import finalize

    finalize(_raw(root), root, log=lambda *_: None)


# ---------- the web API ----------


class AnswerIn(BaseModel):
    question_id: str
    option: str | None = None
    note: str = ""
    text: str = ""
    corpus: str | None = None


class ApproveIn(BaseModel):
    approver: str = ""
    corpus: str | None = None


def routes(get: Any) -> Any:
    """The engine's HTTP routes; `get(corpus)` returns the loaded Mirror."""
    from fastapi import APIRouter, HTTPException

    r = APIRouter()

    def guard(fn: Any) -> Any:
        try:
            return fn()
        except KeyError as e:
            raise HTTPException(404, f"not found: {e.args[0]}") from None
        except (ValueError, llm.LLMUnavailable) as e:
            raise HTTPException(400, str(e)) from None

    @r.get("/api/questions")
    def list_questions(limit: int = 20, kind: str | None = None, corpus: str | None = None) -> dict:
        m = get(corpus)
        qs = questions(m, limit=None, kinds={kind} if kind else None)
        return {"questions": qs[:limit], "total": len(qs), "counts": dict(Counter(q["kind"] for q in qs)),
                "proposals": proposals(m.root)[::-1], "llm": llm.available()}

    @r.post("/api/questions/answer")
    def post_answer(a: AnswerIn) -> dict:
        return guard(lambda: answer(get(a.corpus), a.question_id, a.option, a.note, a.text))

    @r.post("/api/proposals/{pid}/approve")
    def post_approve(pid: str, a: ApproveIn) -> dict:
        return guard(lambda: approve(get(a.corpus).root, pid, a.approver))

    @r.post("/api/proposals/{pid}/reject")
    def post_reject(pid: str, a: ApproveIn) -> dict:
        return guard(lambda: reject(get(a.corpus).root, pid))

    @r.post("/api/proposals/{pid}/revoke")
    def post_revoke(pid: str, a: ApproveIn) -> dict:
        return guard(lambda: revoke(get(a.corpus).root, pid))

    return r


# ---------- the command line ----------


def cli(root: Path, limit: int = 10, kind: str | None = None, listing: bool = False, inp: Any = input,
        out: Any = print) -> None:
    """`python -m androidmirror question`: answer questions in the terminal, then approve or leave each proposal."""
    m = Mirror.load(root)
    if m is None:
        raise SystemExit("no Mirror built yet: python -m androidmirror build")
    qs = questions(m, limit=limit, kinds={kind} if kind else None)
    if not qs:
        out("No open questions.")
    for i, q in enumerate(qs, 1):
        out(f"\n[{i}/{len(qs)}] ({q['kind']}) {q['text']}")
        for e in q["evidence"][:2]:
            out(f"    [{e['locator']}] \"{e['quote']}\"")
        if listing:
            continue
        for n, o in enumerate(q["options"], 1):
            out(f"  {n}. {o['label']}")
        raw = inp(f"answer (1-{len(q['options'])}{', or write it out' if llm.available() else ''}; q to stop): ").strip()
        if raw.lower() == "q":
            break
        if not raw:
            continue
        try:
            if raw.isdigit() and 1 <= int(raw) <= len(q["options"]):
                p = answer(m, q["id"], q["options"][int(raw) - 1]["key"])
            else:
                p = answer(m, q["id"], text=raw)
        except (ValueError, llm.LLMUnavailable) as e:
            out(f"  {e}")
            continue
        if p["status"] == "skipped":
            continue
        out(f"  proposal {p['id']}: {p['answer']}" + (f" ({p['note']})" if p["note"] else ""))
        for ch in p["changes"]:
            out(f"    - {ch['says']}")
        out(f"  if approved: {p['preview']['summary']}")
        name = inp("  approve? type your name to approve, n to reject, enter to leave it proposed: ").strip()
        if name.lower() == "n":
            reject(root, p["id"])
        elif name:
            approve(root, p["id"], name)
            m = Mirror.load(root)
            out("  approved; the Mirror is re-fitted")
    waiting = proposals(root, "proposed")
    if waiting:
        out(f"\n{len(waiting)} proposal(s) waiting for approval; approve them in the Questions tab or with "
            f"`question --approve ID --by NAME`.")
