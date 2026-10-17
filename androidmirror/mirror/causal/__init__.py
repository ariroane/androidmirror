"""The subject's causal model, fitted to their own causal judgments.

The data are the constraints in judgments.py, each weighted by the number of
distinct passages that assert it: x reaches y (with a sign), x has no path to y,
x reaches z only through m. `fit` finds the smallest graph that satisfies as
much of that weight as it can:

1. Constraints are taken strongest first (on a tie, a denial before a claim,
   so the model never has more arrows than the subject's words require; then
   the one said more often, repetitions included; then by name).
2. A claim x -> y that the graph already entails (a path from x to y of the
   same sign) adds nothing. A claim the subject does not call direct waits
   until the rest of the graph is in, since their other judgments may give it
   a path. Otherwise it adds the arrow x -> y, unless that
   would make a cycle or break a denial already taken; then it is rejected and
   the constraints it ran into are recorded. A cycle is the subject's own
   feedback loop or contradiction, and is reported as such, never hidden.
3. A denial is taken if the graph so far satisfies it, and rejected (with the
   claims that outweigh it) if not.

Every arrow keeps the judgments behind it, so each edge in the app goes back
to its quotes. An arrow added for an "overall" claim that the rest of the
graph also explains is marked open: the subject may mean it only through the
other path. That, and the rejected constraints, are what the subject's words
leave undecided.

Arrows come only from the subject: no co-occurrence counts, no simulated
answers. `holdout` checks the model against held-out passages: does the rest
of the corpus imply what a passage says, contradict it, or say nothing?
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..judgments import constraints as build_constraints

DEFAULTS = {"min_weight": 1, "folds": 5}
RANK = {"no_path": 0, "only_through": 1, "reaches": 2}
MAX_PATHS = 64
MAX_LEN = 8


class Graph:
    """A directed graph with an edge record per arrow."""

    def __init__(self) -> None:
        self.succ: dict[str, dict[str, dict[str, Any]]] = {}

    def add(self, a: str, b: str, rec: dict[str, Any]) -> None:
        self.succ.setdefault(a, {})[b] = rec
        self.succ.setdefault(b, {})

    def remove(self, a: str, b: str) -> dict[str, Any]:
        return self.succ[a].pop(b)

    def edges(self) -> list[dict[str, Any]]:
        return [rec for a in sorted(self.succ) for b, rec in sorted(self.succ[a].items())]

    def path(self, a: str, b: str, skip: str | None = None) -> list[str] | None:
        """A shortest directed path from a to b (avoiding `skip`), or None."""
        if a == b or a == skip or b == skip:
            return None
        prev = {a: None}
        todo = deque([a])
        while todo:
            n = todo.popleft()
            for m in self.succ.get(n, {}):
                if m in prev or m == skip:
                    continue
                prev[m] = n
                if m == b:
                    out = [b]
                    while prev[out[-1]] is not None:
                        out.append(prev[out[-1]])
                    return out[::-1]
                todo.append(m)
        return None

    def reaches(self, a: str, b: str, skip: str | None = None) -> bool:
        return self.path(a, b, skip) is not None

    def would_reach(self, a: str, b: str, x: str, y: str, skip: str | None = None) -> bool:
        """Would a reach b (avoiding `skip`) once the arrow x -> y is added?"""
        if self.reaches(a, b, skip):
            return True
        if skip in (x, y):
            return False
        return (a == x or self.reaches(a, x, skip)) and (y == b or self.reaches(y, b, skip))

    def signed_paths(self, a: str, b: str) -> list[tuple[list[str], int]]:
        """Simple directed paths from a to b with the product of their signs (at most MAX_PATHS)."""
        out: list[tuple[list[str], int]] = []

        def walk(n: str, path: list[str], sign: int) -> None:
            if len(out) >= MAX_PATHS or len(path) > MAX_LEN:
                return
            for m, rec in self.succ.get(n, {}).items():
                if m in path:
                    continue
                if m == b:
                    out.append((path + [m], sign * rec["sign"]))
                else:
                    walk(m, path + [m], sign * rec["sign"])

        if a != b:
            walk(a, [a], 1)
        return out

    def path_constraints(self, path: list[str]) -> list[str]:
        return [self.succ[a][b]["constraint"] for a, b in zip(path, path[1:])]


def fit(judgments: list[dict[str, Any]], settings: dict[str, Any] | None = None,
        exclude: set[str] | frozenset = frozenset()) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(settings or {})}
    allc = build_constraints(judgments, exclude)
    used = [c for c in allc if c["weight"] >= cfg["min_weight"]]
    used.sort(key=lambda c: (-c["weight"], RANK[c["type"]], -c["mentions"], c["id"]))
    g = Graph()
    for c in used:
        for n in (c["x"], c["y"], c["z"]):
            if n:
                g.succ.setdefault(n, {})
    accepted_neg: list[dict[str, Any]] = []
    status: dict[str, dict[str, Any]] = {}

    def blocked(x: str, y: str) -> dict[str, Any] | None:
        """Why the arrow x -> y cannot be added, or None if it can."""
        back = g.path(y, x)
        if back is not None or x == y:
            return {"reason": "cycle", "path": back or [y], "conflicts": g.path_constraints(back or [y])}
        return denied(x, y)

    def denied(x: str, y: str) -> dict[str, Any] | None:
        for n in accepted_neg:
            target, skip = _target(n)
            if g.would_reach(n["x"], target, x, y, skip):
                return {"reason": "denied", "conflicts": [n["id"]]}
        return None

    def add_edge(c: dict[str, Any], x: str, y: str, note: str | None = None) -> None:
        g.add(x, y, {"src": x, "dst": y, "sign": c["sign"], "constraint": c["id"], "weight": c["weight"],
                     "direct": c["direct"], "kinds": c["kinds"], "supports": [], "note": note, "open": None})

    def claim(c: dict[str, Any], defer: bool) -> bool:
        """Take a claim x -> y; returns False if it was deferred."""
        x, y = c["x"], c["y"]
        paths = g.signed_paths(x, y)
        same = [p for p, s in paths if s == c["sign"]]
        if same:
            p = min(same, key=len)
            for a, b in zip(p, p[1:]):
                g.succ[a][b]["supports"].append(c["id"])
            status[c["id"]] = {"status": "follows", "path": p}
            return True
        if defer and not paths:
            return False
        # with only a path of the other sign, the direct arrow is added beside it, unless a denial forbids it
        why = denied(x, y) if paths else blocked(x, y)
        if why:
            status[c["id"]] = {"status": "rejected", **why}
        else:
            add_edge(c, x, y, "beside a path of the other sign" if paths else None)
            status[c["id"]] = {"status": "arrow"}
        return True

    # An "overall" claim that nothing explains yet waits until the end: the subject's other judgments may yet give it
    # a path, and only if none does is it drawn as an arrow of its own. A denial is still weighed against it now.
    deferred: list[dict[str, Any]] = []
    for c in used:
        if c["type"] == "reaches":
            if not claim(c, defer=not c["direct"]):
                deferred.append(c)
            continue
        target, skip = _target(c)
        p = g.path(c["x"], target, skip)
        stronger = next((d for d in deferred if c["type"] == "no_path" and (d["x"], d["y"]) == (c["x"], c["y"])
                         and d["weight"] > c["weight"]), None)
        if p is not None:
            status[c["id"]] = {"status": "rejected", "reason": "outweighed", "path": p, "conflicts": g.path_constraints(p)}
        elif stronger:
            status[c["id"]] = {"status": "rejected", "reason": "outweighed", "conflicts": [stronger["id"]]}
        else:
            accepted_neg.append(c)
            status[c["id"]] = {"status": "holds"}
    for c in deferred:
        claim(c, defer=False)

    # arrows the rest of the graph also explains: the subject may mean them only through the other path
    for rec in g.edges():
        if rec["direct"]:
            continue
        a, b = rec["src"], rec["dst"]
        g.remove(a, b)
        alt = [p for p, s in g.signed_paths(a, b) if s == rec["sign"]]
        g.add(a, b, rec)
        if alt:
            rec["open"] = {"through": min(alt, key=len)}

    by_id = {c["id"]: c for c in allc}
    pairs = {(c["x"], c["y"]) for c in used if c["type"] == "reaches"}
    feedback = sorted({tuple(sorted((x, y))) for x, y in pairs if (y, x) in pairs})
    rows = []
    for c in allc:
        st = status.get(c["id"], {"status": "not used" if c["weight"] else "not held"})
        rows.append({**{k: c[k] for k in ("id", "type", "x", "y", "z", "sign", "weight", "mentions", "kinds", "direct",
                                          "sign_conflict", "held")}, **st, "evidence": c["evidence"][:12],
                     "n_evidence": len(c["evidence"])})
    edges = g.edges()
    nodes = sorted(n for n in g.succ if g.succ[n] or any(n in g.succ[m] for m in g.succ)) if edges else []
    isolated = sorted(n for n in g.succ if n not in set(nodes))
    count = lambda s: sum(1 for r in rows if r["status"] == s)  # noqa: E731
    return {
        "settings": cfg, "nodes": nodes, "isolated": isolated, "edges": edges, "constraints": rows,
        "tensions": [{"id": c["id"], "x": c["x"], "y": c["y"], "signs": c["signs"]} for c in allc if c["sign_conflict"]],
        "feedback": [{"pair": list(p), "status": [status.get(f"reaches|{p[0]}|{p[1]}", {}).get("status"),
                                                  status.get(f"reaches|{p[1]}|{p[0]}", {}).get("status")]} for p in feedback],
        "stats": {"judgments": len(judgments), "held_judgments": sum(1 for j in judgments if j["held"]),
                  "constraints": len(allc), "held_constraints": sum(1 for c in allc if c["held"]),
                  "arrows": len(edges), "open": sum(1 for e in edges if e["open"]), "follows": count("follows"),
                  "rejected": count("rejected"), "denials_held": count("holds")},
        "_graph": g, "_by_id": by_id,
    }


def _target(c: dict[str, Any]) -> tuple[str, str | None]:
    """A denial's end and the node it lets paths through: (y, None) for no_path, (z, y) for only_through."""
    return (c["z"], c["y"]) if c["type"] == "only_through" else (c["y"], None)


def public(model: dict[str, Any]) -> dict[str, Any]:
    """The model without its working objects, for saving."""
    return {k: v for k, v in model.items() if not k.startswith("_")}


def graph_of(model: dict[str, Any]) -> Graph:
    g = Graph()
    for n in model.get("nodes", []) + model.get("isolated", []):
        g.succ.setdefault(n, {})
    for e in model.get("edges", []):
        g.add(e["src"], e["dst"], e)
    return g


def explain(model: dict[str, Any], x: str, y: str) -> dict[str, Any]:
    """What the subject's model says changing x does to y, and on which judgments that rests."""
    g = model.get("_graph") or graph_of(model)
    paths = g.signed_paths(x, y)
    by_id = {c["id"]: c for c in model.get("constraints", [])}
    denial = by_id.get(f"no_path|{x}|{y}")
    out = {"x": x, "y": y, "paths": [{"nodes": p, "sign": s, "steps": [
        {"src": a, "dst": b, "sign": g.succ[a][b]["sign"], "constraint": g.succ[a][b]["constraint"],
         "open": g.succ[a][b]["open"], "evidence": (by_id.get(g.succ[a][b]["constraint"]) or {}).get("evidence", [])[:2]}
        for a, b in zip(p, p[1:])]} for p, s in sorted(paths, key=lambda ps: len(ps[0]))[:12]],
        "reverse": bool(g.signed_paths(y, x)), "denial": denial,
        "claims": [c for c in model.get("constraints", []) if {c["x"], c["y"]} == {x, y} or (c["type"] == "only_through" and {c["x"], c["z"]} == {x, y})]}
    signs = {s for _, s in paths}
    if signs == {1}:
        out["verdict"] = "raises"
    elif signs == {-1}:
        out["verdict"] = "lowers"
    elif signs:
        out["verdict"] = "mixed: some paths raise it, some lower it"
    elif denial and denial["status"] == "holds":
        out["verdict"] = "makes no difference (the subject says so)"
    else:
        out["verdict"] = "not said: the subject's judgments do not connect them"
    return out


def holdout(judgments: list[dict[str, Any]], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit on all but one block of passages and ask what the model says about each held-out judgment.

    Blocks are runs of consecutive passages, so a held-out passage's neighbours (which often repeat it) are held
    out with it. "restated" means the rest of the corpus says the same thing; "follows" means it is implied
    without being said; "new" means the rest is silent; "contradicted" means the rest implies the opposite."""
    cfg = {**DEFAULTS, **(settings or {})}
    order = list(dict.fromkeys(j["passage"] for j in judgments if j["held"]))
    k = max(2, min(cfg["folds"], len(order)))
    if len(order) < 2:
        return {"folds": 0, "positive": {}, "negative": {}, "examples": []}
    size = -(-len(order) // k)
    pos = dict.fromkeys(["restated", "follows", "new", "sign differs", "contradicted"], 0)
    neg = dict.fromkeys(["restated", "consistent", "contradicted"], 0)
    examples = []
    for f in range(k):
        held = set(order[f * size:(f + 1) * size])
        if not held:
            continue
        model = fit(judgments, cfg, exclude=held)
        g, train = model["_graph"], model["_by_id"]
        accepted = {c["id"] for c in model["constraints"] if c["status"] == "holds"}
        seen = set()
        for c in build_constraints([j for j in judgments if j["passage"] in held]):
            if not c["held"]:
                continue
            for pid in {e["passage"] for e in c["evidence"] if e["held"]}:
                if (pid, c["id"]) in seen:
                    continue
                seen.add((pid, c["id"]))
                again = train.get(c["id"])
                if c["type"] == "reaches":
                    signs = {s for _, s in g.signed_paths(c["x"], c["y"])}
                    if again and again["weight"]:
                        res = "restated"
                    elif c["sign"] in signs:
                        res = "follows"
                    elif signs:
                        res = "sign differs"
                    elif g.reaches(c["y"], c["x"]) or f"no_path|{c['x']}|{c['y']}" in accepted:
                        res = "contradicted"
                    else:
                        res = "new"
                    pos[res] += 1
                else:
                    target, skip = _target(c)
                    if again and again["weight"]:
                        res = "restated"
                    elif g.reaches(c["x"], target, skip):
                        res = "contradicted"
                    else:
                        res = "consistent"
                    neg[res] += 1
                if res in ("follows", "contradicted", "sign differs") and len(examples) < 20:
                    ev = next(e for e in c["evidence"] if e["passage"] == pid)
                    examples.append({"result": res, "constraint": c["id"], "type": c["type"], "x": c["x"], "y": c["y"],
                                     "z": c["z"], "passage": pid, "locator": ev["locator"], "quote": ev["quote"]})
    return {"folds": k, "positive": pos, "negative": neg, "examples": examples}
