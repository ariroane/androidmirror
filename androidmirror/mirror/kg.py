"""The descriptive ontology as a knowledge base that supports inference.

Stated facts come from the corpus, one record per (subject, relation, object,
negated) with every passage that states it (evidence accumulates instead of
being overwritten). Only facts the subject holds (its own assertions and
self-prescriptions, not quoted views, questions or hypotheses) enter
reasoning; the rest are kept as reported speech.

Inference follows the two families in CS520's notes:

- taxonomic: `is_a` is transitive, so membership and evaluation are inherited
  ("justice is_a virtue", "virtue is_a good" gives "justice is_a good"); some
  relations are inherited down the hierarchy (`opposes`, `requires`,
  `accords_with` pass from a class to its members; `values` and `rejects`
  pass from a class to its members on the object side).
- rule-based: transitivity, symmetry and `same_as` substitution, computed
  bottom-up (a chase to a fixpoint). Each derived fact keeps the rule and
  premises it came from, so every answer has a proof that bottoms out in quotes.

Constraint checks make inconsistencies visible instead of silently merging
them: the evaluation partition (good / bad / indifferent are disjoint), values
versus rejects, a stated denial of something that follows, taxonomy cycles, and
domain/range violations. The world is open: a fact that is neither stated nor
derived is "not found", never false.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import EVALUATIONS, HELD_MODALITIES, RELATIONS, SELF, check_fact

Triple = tuple[str, str, str]
MAX_ROUNDS = 12
MAX_DERIVED = 50_000


def fact_key(s: str, r: str, o: str, neg: bool = False) -> str:
    return f"{s}|{r}|{o}" + ("|not" if neg else "")


class KnowledgeBase:
    def __init__(self, nodes: dict[str, dict[str, Any]], facts: list[dict[str, Any]]) -> None:
        self.nodes = nodes
        self.facts = {fact_key(f["s"], f["r"], f["o"], f.get("neg", False)): f for f in facts}
        self.violations: list[dict[str, Any]] = []
        self.closure: dict[Triple, dict[str, Any]] = {}  # triple -> {"rule", "from": [triples]} or {"rule": "stated"}
        self._run()

    # ---------- inference ----------

    def _held(self) -> list[Triple]:
        out = []
        for f in self.facts.values():
            if f.get("neg") or not f.get("held"):
                continue
            kind = lambda n: self.nodes.get(n, {}).get("kind", "concept")  # noqa: E731
            msg = check_fact(f["r"], kind(f["s"]), kind(f["o"]), f["o"])
            if msg:
                self.violations.append({"type": "domain_range", "fact": fact_key(f["s"], f["r"], f["o"]), "why": msg})
                continue
            out.append((f["s"], f["r"], f["o"]))
        return out

    def _run(self) -> None:
        known: dict[Triple, dict[str, Any]] = {t: {"rule": "stated"} for t in self._held()}
        by_rel_src: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        by_rel_dst: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

        def index(t: Triple) -> None:
            s, r, o = t
            by_rel_src[r][s].add(o)
            by_rel_dst[r][o].add(s)

        for t in known:
            index(t)
        delta = list(known)
        for _ in range(MAX_ROUNDS):
            new: dict[Triple, dict[str, Any]] = {}

            def emit(t: Triple, rule: str, premises: list[Triple]) -> None:
                if t[0] == t[2] and t[1] != "opposes":  # a is_a a etc. carry no information
                    return
                if t not in known and t not in new:
                    new[t] = {"rule": rule, "from": premises}

            for t in delta:
                s, r, o = t
                spec = RELATIONS.get(r, {})
                if spec.get("symmetric"):
                    emit((o, r, s), f"{r} is symmetric", [t])
                if spec.get("transitive"):
                    for o2 in list(by_rel_src[r].get(o, ())):
                        emit((s, r, o2), f"{r} is transitive", [t, (o, r, o2)])
                    for s0 in list(by_rel_dst[r].get(s, ())):
                        emit((s0, r, o), f"{r} is transitive", [(s0, r, s), t])
                if r == "is_a":
                    # members inherit what their class opposes, requires, accords with
                    for p, pspec in RELATIONS.items():
                        if pspec.get("inheritable"):
                            for o2 in list(by_rel_src[p].get(o, ())):
                                emit((s, p, o2), f"{p} is inherited along is_a", [t, (o, p, o2)])
                        if pspec.get("down_object"):
                            for x in list(by_rel_dst[p].get(o, ())):
                                emit((x, p, s), f"{p} passes from a class to its members", [(x, p, o), t])
                if spec.get("inheritable"):
                    for m in list(by_rel_dst["is_a"].get(s, ())):
                        emit((m, r, o), f"{r} is inherited along is_a", [(m, "is_a", s), t])
                if spec.get("down_object"):
                    for m in list(by_rel_dst["is_a"].get(o, ())):
                        emit((s, r, m), f"{r} passes from a class to its members", [t, (m, "is_a", o)])
                if r == "same_as":
                    for rel in list(by_rel_src):
                        if rel == "same_as":
                            continue
                        for o2 in list(by_rel_src[rel].get(o, ())):
                            emit((s, rel, o2), "same_as substitution", [t, (o, rel, o2)])
                        for s2 in list(by_rel_dst[rel].get(o, ())):
                            emit((s2, rel, s), "same_as substitution", [t, (s2, rel, o)])
            if not new or len(known) + len(new) > MAX_DERIVED:
                break
            for t, proof in new.items():
                known[t] = proof
                index(t)
            delta = list(new)
        self.closure = known
        self._check()

    def _check(self) -> None:
        c = self.closure
        for x in self.nodes:
            evs = [e for e in EVALUATIONS if (x, "is_a", e) in c]
            if len(evs) > 1:
                self.violations.append({"type": "disjoint_evaluations", "node": x, "evaluations": evs,
                                        "facts": [fact_key(x, "is_a", e) for e in evs]})
        for (s, r, o) in list(c):
            if r == "values" and (s, "rejects", o) in c:
                self.violations.append({"type": "values_and_rejects", "agent": s, "node": o,
                                        "facts": [fact_key(s, "values", o), fact_key(s, "rejects", o)]})
            if r == "is_a" and (o, "is_a", s) in c and s < o and o not in EVALUATIONS:
                self.violations.append({"type": "taxonomy_cycle", "nodes": [s, o],
                                        "facts": [fact_key(s, "is_a", o), fact_key(o, "is_a", s)],
                                        "why": "each is a kind of the other; perhaps the same concept"})
            if r == "opposes" and s == o:
                self.violations.append({"type": "self_opposition", "node": s, "facts": [fact_key(s, r, o)]})
        for f in self.facts.values():
            if f.get("neg") and f.get("held") and (f["s"], f["r"], f["o"]) in c:
                self.violations.append({"type": "denied_but_follows", "fact": fact_key(f["s"], f["r"], f["o"]),
                                        "facts": [fact_key(f["s"], f["r"], f["o"]), fact_key(f["s"], f["r"], f["o"], True)]})

    # ---------- queries ----------

    def holds(self, s: str, r: str, o: str) -> dict[str, Any]:
        """Whether the subject holds s-r-o: stated, inferred (with proof), denied, or not found."""
        t = (s, r, o)
        neg = self.facts.get(fact_key(s, r, o, True))
        if t in self.closure:
            out = {"status": "stated" if self.closure[t]["rule"] == "stated" else "inferred", "proof": self.explain(t)}
            if neg and neg.get("held"):
                out["conflict"] = "also denied"
            return out
        if neg and neg.get("held"):
            return {"status": "denied", "evidence": neg["evidence"]}
        return {"status": "not found"}

    def query(self, s: str | None = None, r: str | None = None, o: str | None = None,
              limit: int = 200) -> list[dict[str, Any]]:
        """All facts in the closure matching a pattern (None is a variable)."""
        out = []
        for (a, b, c), proof in self.closure.items():
            if (s is None or a == s) and (r is None or b == r) and (o is None or c == o):
                out.append({"s": a, "r": b, "o": c, "status": "stated" if proof["rule"] == "stated" else "inferred",
                            "rule": proof["rule"]})
                if len(out) >= limit:
                    break
        return out

    def explain(self, t: Triple, depth: int = 0) -> dict[str, Any]:
        """The proof tree for a fact in the closure, down to the quotes it rests on."""
        proof = self.closure.get(t)
        node: dict[str, Any] = {"fact": fact_key(*t), "s": t[0], "r": t[1], "o": t[2]}
        if proof is None:
            return {**node, "rule": "missing"}
        if proof["rule"] == "stated":
            f = self.facts.get(fact_key(*t), {})
            return {**node, "rule": "stated", "evidence": f.get("evidence", [])[:5], "support": f.get("support", 0)}
        node["rule"] = proof["rule"]
        node["from"] = [self.explain(p, depth + 1) for p in proof["from"]] if depth < 8 else []
        return node

    def stated_support(self, t: Triple) -> set[str]:
        """Every passage a derived fact ultimately rests on."""
        proof = self.closure.get(t)
        if proof is None:
            return set()
        if proof["rule"] == "stated":
            return {e["passage"] for e in self.facts[fact_key(*t)]["evidence"]}
        return set().union(*(self.stated_support(p) for p in proof["from"]))

    def stats(self) -> dict[str, int]:
        stated = sum(1 for p in self.closure.values() if p["rule"] == "stated")
        return {"stated_facts": len(self.facts), "held": stated, "inferred": len(self.closure) - stated,
                "violations": len(self.violations),
                "reported_not_held": sum(1 for f in self.facts.values() if not f.get("held"))}


def aggregate(stmts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold per-passage statements into one record per fact, with all their evidence."""
    out: dict[str, dict[str, Any]] = {}
    for st in stmts:
        k = fact_key(st["s"], st["r"], st["o"], st["neg"])
        f = out.setdefault(k, {"s": st["s"], "r": st["r"], "o": st["o"], "neg": st["neg"], "evidence": [],
                               "held": False, "qualifiers": []})
        ev = {k2: st[k2] for k2 in ("passage", "locator", "quote", "modality", "attribution") if k2 in st}
        f["evidence"].append(ev)
        if st.get("qualifier") and st["qualifier"] not in f["qualifiers"]:
            f["qualifiers"].append(st["qualifier"])
        if st.get("wording"):
            f.setdefault("wordings", [])
            if st["wording"] not in f["wordings"]:
                f["wordings"].append(st["wording"])
        if st["attribution"] == SELF and st["modality"] in HELD_MODALITIES:
            f["held"] = True
    for f in out.values():
        f["support"] = len({e["passage"] for e in f["evidence"]})
        f["modalities"] = sorted({e["modality"] for e in f["evidence"]})
    return list(out.values())
