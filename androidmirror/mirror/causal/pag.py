"""The subject's causal model as a PAG: what every hidden-cause graph that fits their commitments has in common.

The fitted model is a mixed graph: one-way arrows (x drives y) and two-way links (x <-> y, "something the subject
never names drives both"). Read each two-way link as a hidden common cause and the model is a graph with hidden
variables. A PAG (partial ancestral graph) is the standard summary of such a graph: it keeps a mark only where
every graph with the same pattern of "can be separated / cannot be separated" agrees, and puts a circle ("could be
either") everywhere else. Getting there takes two steps:

1. `to_mag` turns the model into a maximal ancestral graph (MAG). On each pair the model links, the mark follows the
   chains of arrows: x -> y if x leads to y, y -> x if y leads to x, x <-> y if neither does. So a two-way link
   beside a chain of arrows becomes a plain arrow (in a MAG an arrow means "leads to, perhaps through other
   ideas", not "acts directly"). A MAG must also link every pair that no set of ideas can screen off from each
   other. The model leaves such pairs unlinked, because the subject never connects them; under the hidden-cause
   reading they are connected anyway (x -> c <-> y with c leading on to y: holding c fixed blocks x's chain to y,
   and opens the route through the hidden cause). They are found (`inseparable`) but left out unless asked for:
   adding a link the subject never states is the opposite of the Mirror's rule that silence means no link. Left
   out, the result is not a valid PAG on those pairs, and says so (`valid`).
2. `pag` runs the orientation rules of FCI (Zhang 2008: R0-R4, R8-R10; no selection bias) on the MAG. With
   `background`, the directions the subject's arrows imply are fed in first as background knowledge (x leads to y:
   a tail at x, an arrowhead at y), as are the "makes no difference" claims that hold (x does not lead to y: an
   arrowhead at x). The rules are sound with background knowledge, but not known to be complete, so a few circles
   may remain that every graph agrees on.

Nothing here changes the fitted model; it is a view of it. Effects are still traced on the fitted model.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Any

ARROW, TAIL, CIRCLE = ">", "-", "o"
NAMES = {ARROW: "arrow", TAIL: "tail", CIRCLE: "circle"}
BUDGET = 20000  # path steps per search in R9/R10; past it the rule is skipped, which leaves a circle (always sound)


def admg(model: dict[str, Any]) -> tuple[list[str], set[tuple[str, str]], set[frozenset]]:
    """The fitted model's nodes, arrows and two-way links (those not explained by a named shared cause)."""
    directed = {(e["src"], e["dst"]) for e in model.get("edges", [])}
    bidirected = {frozenset((b["a"], b["b"])) for b in model.get("common_causes", []) if not b.get("via")}
    nodes = sorted(set(model.get("nodes", [])) | {n for e in directed for n in e} | {n for b in bidirected for n in b})
    return nodes, directed, bidirected


def descendants(nodes: list[str], directed: set[tuple[str, str]]) -> dict[str, set[str]]:
    succ: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in directed:
        succ[a].append(b)
    out = {}
    for n in nodes:
        seen, todo = set(), [n]
        while todo:
            for m in succ[todo.pop()]:
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        out[n] = seen
    return out


def _chain(directed: set[tuple[str, str]], a: str, b: str) -> list[str]:
    succ: dict[str, list[str]] = {}
    for u, v in sorted(directed):
        succ.setdefault(u, []).append(v)
    prev, todo = {a: None}, deque([a])
    while todo:
        n = todo.popleft()
        for m in succ.get(n, []):
            if m not in prev:
                prev[m] = n
                if m == b:
                    out = [b]
                    while prev[out[-1]] is not None:
                        out.append(prev[out[-1]])
                    return out[::-1]
                todo.append(m)
    return []


def inducing_path(a: str, b: str, directed: set[tuple[str, str]], bidirected: set[frozenset],
                  desc: dict[str, set[str]], into: dict[str, set[str]], spouses: dict[str, set[str]]) -> list[str]:
    """A path a *-> c1 <-> ... <-> ck <-* b whose middle ideas all lead to a or b, or [] if there is none.

    Every middle idea is a collider (an arrowhead meets it from both sides), so with the two-way links read as
    hidden causes, no set of ideas screens a off from b: leaving a middle idea out of the set leaves open its chain
    on to a or b, and putting it in opens the route through it. These are the pairs a MAG must link (Richardson
    and Spirtes 2002, inducing paths)."""
    allowed = {c for c in desc if c not in (a, b) and (a in desc[c] or b in desc[c])}
    # first step: an edge from a with its arrowhead at c
    start = [c for c in allowed if (a, c) in directed or frozenset((a, c)) in bidirected]
    prev: dict[str, str | None] = {c: None for c in start}
    todo = deque(start)
    while todo:
        c = todo.popleft()
        if b in into[c] or b in spouses[c]:  # last step: b -> c or b <-> c
            out = [c]
            while prev[out[-1]] is not None:
                out.append(prev[out[-1]])
            return [a] + out[::-1] + [b]
        for d in spouses[c]:
            if d in allowed and d not in prev:
                prev[d] = c
                todo.append(d)
    return []


def to_mag(model: dict[str, Any], inseparable: bool = False) -> dict[str, Any]:
    """The fitted model as an ancestral graph: its links with the marks the chains of arrows imply.

    Returns the nodes, `marks` ({(u, v): mark at v} for every linked pair, both ways round), the two-way links that
    became arrows (`converted`), and the pairs nothing can screen off (`inseparable`, each with the path that ties
    it). Those pairs are linked in `marks` only when `inseparable` is True; otherwise the graph is ancestral but not
    maximal, and `maximal` is False when there are any."""
    nodes, directed, bidirected = admg(model)
    desc = descendants(nodes, directed)
    into: dict[str, set[str]] = {n: set() for n in nodes}
    spouses: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in directed:
        into[b].add(a)
    for pair in bidirected:
        a, b = tuple(pair)
        spouses[a].add(b)
        spouses[b].add(a)
    adjacent = {frozenset(e) for e in directed} | bidirected

    def orient(a: str, b: str) -> tuple[str, str]:
        """(mark at a, mark at b) on the a - b link."""
        if b in desc[a]:
            return TAIL, ARROW
        if a in desc[b]:
            return ARROW, TAIL
        return ARROW, ARROW

    marks: dict[tuple[str, str], str] = {}
    converted = []
    for pair in sorted(adjacent, key=sorted):
        a, b = sorted(pair)
        ma, mb = orient(a, b)
        marks[(b, a)], marks[(a, b)] = ma, mb
        if pair in bidirected and (ma, mb) != (ARROW, ARROW):
            src, dst = (a, b) if ma == TAIL else (b, a)
            converted.append({"src": src, "dst": dst, "chain": _chain(directed, src, dst),
                              "also_arrow": (src, dst) in directed})
    # only an idea that a two-way link reaches (directly or through others) can tie a non-linked pair
    reach: dict[str, set[str]] = {}
    for n in nodes:
        first = {c for c in nodes if (n, c) in directed or frozenset((n, c)) in bidirected}
        seen, todo = set(first), list(first)
        while todo:
            for d in spouses[todo.pop()]:
                if d not in seen:
                    seen.add(d)
                    todo.append(d)
        reach[n] = {m for c in seen for m in into[c] | spouses[c]} if any(spouses[c] for c in seen) else set()
    found = []
    for a, b in combinations(nodes, 2):
        if frozenset((a, b)) in adjacent or b not in reach[a]:
            continue
        p = inducing_path(a, b, directed, bidirected, desc, into, spouses)
        if p:
            ma, mb = orient(a, b)
            found.append({"a": a, "b": b, "marks": [NAMES[ma], NAMES[mb]], "path": p})
            if inseparable:
                marks[(b, a)], marks[(a, b)] = ma, mb
    return {"nodes": nodes, "marks": marks, "converted": converted, "inseparable": found,
            "maximal": inseparable or not found, "_desc": desc}


class _Pag:
    """Circle/arrow/tail marks on a fixed skeleton, with the orientation rules. m[(u, v)] is the mark at v."""

    def __init__(self, nodes: list[str], mag: dict[tuple[str, str], str]) -> None:
        self.mag = mag
        self.m = {k: CIRCLE for k in mag}
        self.adj: dict[str, set[str]] = {n: set() for n in nodes}
        for u, v in mag:
            self.adj[u].add(v)
        self.changed = False

    def is_adj(self, a: str, b: str) -> bool:
        return b in self.adj[a]

    def set(self, u: str, v: str, mark: str) -> None:
        if self.m[(u, v)] != mark:
            self.m[(u, v)] = mark
            self.changed = True

    def into(self, u: str, v: str) -> bool:  # u *-> v
        return self.m.get((u, v)) == ARROW

    def directed(self, u: str, v: str) -> bool:  # u -> v
        return self.m.get((v, u)) == TAIL and self.m.get((u, v)) == ARROW

    def mag_collider(self, a: str, b: str, c: str) -> bool:
        return self.mag[(a, b)] == ARROW and self.mag[(c, b)] == ARROW

    # R0: unshielded colliders, as in the MAG (every graph in the class shares them)
    def r0(self) -> None:
        for b in self.adj:
            for a, c in combinations(sorted(self.adj[b]), 2):
                if not self.is_adj(a, c) and self.mag_collider(a, b, c):
                    self.set(a, b, ARROW)
                    self.set(c, b, ARROW)

    def r1_to_r3(self) -> None:
        for b in self.adj:
            nb = sorted(self.adj[b])
            for a in nb:
                for c in nb:
                    if a == c:
                        continue
                    # R1: a *-> b o-* c, a and c not adjacent: b -> c
                    if self.into(a, b) and self.m[(c, b)] == CIRCLE and not self.is_adj(a, c):
                        self.set(c, b, TAIL)
                        self.set(b, c, ARROW)
                    # R2: a -> b *-> c or a *-> b -> c, and a *-o c: a *-> c
                    if self.is_adj(a, c) and self.m[(a, c)] == CIRCLE and (
                            (self.directed(a, b) and self.into(b, c)) or (self.into(a, b) and self.directed(b, c))):
                        self.set(a, c, ARROW)
            # R3: a *-> b <-* c, a *-o d o-* c, a and c not adjacent, d *-o b: d *-> b
            for a, c in combinations(nb, 2):
                if self.is_adj(a, c) or not (self.into(a, b) and self.into(c, b)):
                    continue
                for d in nb:
                    if d in (a, c) or self.m[(d, b)] != CIRCLE:
                        continue
                    if self.is_adj(a, d) and self.is_adj(c, d) and self.m[(a, d)] == CIRCLE and self.m[(c, d)] == CIRCLE:
                        self.set(d, b, ARROW)

    def r4(self) -> None:
        """Discriminating paths <t, ..., a, b, c> for b, with b o-* c: oriented as b is a collider or not in the MAG."""
        for b in self.adj:
            for c in sorted(self.adj[b]):
                if self.m[(c, b)] != CIRCLE:
                    continue
                for a in sorted(self.adj[b]):
                    if a == c or not self.directed(a, c) or not self.into(b, a):
                        continue
                    # walk back from a through colliders that are parents of c, to a node not adjacent to c
                    prev = {a: b}
                    todo = deque([a])
                    hit = False
                    while todo and not hit:
                        v = todo.popleft()
                        for t in sorted(self.adj[v]):
                            if t in prev or t in (b, c) or not self.into(t, v):
                                continue
                            if not self.is_adj(t, c):
                                hit = True
                                break
                            if self.directed(t, c) and self.into(v, t):
                                prev[t] = v
                                todo.append(t)
                    if hit:
                        if self.mag_collider(a, b, c):
                            for u, v in ((a, b), (b, a), (c, b), (b, c)):
                                self.set(u, v, ARROW)
                        else:
                            self.set(c, b, TAIL)
                            self.set(b, c, ARROW)
                        break

    def _pd(self, u: str, v: str) -> bool:
        """Can the u - v edge point from u to v (no arrowhead at u, no tail at v)?"""
        return self.m[(v, u)] != ARROW and self.m[(u, v)] != TAIL

    def _upd_firsts(self, a: str, target: str, skip: str | None = None) -> set[str]:
        """Second nodes of uncovered potentially directed paths from a to target (not using the edge a - skip)."""
        out, steps = set(), [0]

        def walk(path: list[str]) -> bool:
            steps[0] += 1
            if steps[0] > BUDGET:
                return False
            v = path[-1]
            for w in sorted(self.adj[v]):
                if w in path or not self._pd(v, w) or (len(path) >= 2 and self.is_adj(path[-2], w)):
                    continue
                if w == target:
                    return True
                if walk(path + [w]):
                    return True
            return False

        for mu in sorted(self.adj[a]):
            if mu == skip or not self._pd(a, mu):
                continue
            if mu == target or walk([a, mu]):
                out.add(mu)
            if steps[0] > BUDGET:
                return set()  # unfinished: claim nothing
        return out

    def r8_to_r10(self) -> None:
        for a in self.adj:
            for c in sorted(self.adj[a]):
                if not (self.m[(c, a)] == CIRCLE and self.m[(a, c)] == ARROW):  # a o-> c
                    continue
                # R8: a -> b -> c, or a -o b -> c
                if any(self.directed(b, c) and self.m[(b, a)] == TAIL and self.m[(a, b)] in (ARROW, CIRCLE)
                       for b in self.adj[a] & self.adj[c]):
                    self.set(c, a, TAIL)
                    continue
                # R9: an uncovered potentially directed path a, b, ..., c with b not adjacent to c
                if any(not self.is_adj(b, c) for b in self._upd_firsts(a, c, skip=c) if b != c):
                    self.set(c, a, TAIL)
                    continue
                # R10: b -> c <- t, with uncovered p.d. paths from a to b and a to t starting at non-adjacent mu, omega
                parents = [b for b in self.adj[c] if b != a and self.directed(b, c)]
                done = False
                for b, t in combinations(sorted(parents), 2):
                    s1, s2 = self._upd_firsts(a, b, skip=c), self._upd_firsts(a, t, skip=c)
                    if any(mu != om and not self.is_adj(mu, om) for mu in s1 for om in s2):
                        self.set(c, a, TAIL)
                        done = True
                        break
                if done:
                    continue

    def close(self) -> None:
        while True:
            self.changed = False
            self.r1_to_r3()
            self.r4()
            self.r8_to_r10()
            if not self.changed:
                return


def pag(model: dict[str, Any], background: bool = True, inseparable: bool = False) -> dict[str, Any]:
    """The PAG of the fitted model, read with two-way links as hidden causes.

    `background`: use the directions the subject's arrows and held "makes no difference" claims imply.
    `inseparable`: link the pairs nothing can screen off (the subject never links them; off by default)."""
    mag = to_mag(model, inseparable)
    nodes, desc = mag["nodes"], mag["_desc"]
    p = _Pag(nodes, mag["marks"])
    p.r0()
    known: set[frozenset] = set()
    if background:
        denials = {(c["x"], c["y"]) for c in model.get("constraints", [])
                   if c.get("type") == "no_path" and c.get("status") == "holds"}
        for u, v in mag["marks"]:
            if v in desc[u]:  # u leads to v: u -> v
                p.set(v, u, TAIL)
                p.set(u, v, ARROW)
                known.add(frozenset((u, v)))
            elif (u, v) in denials:  # u does not lead to v: an arrowhead at u
                p.set(v, u, ARROW)
                known.add(frozenset((u, v)))
    p.close()
    _, directed, bidirected = admg(model)
    linked = {frozenset(e) for e in directed} | bidirected
    edges = []
    for (u, v) in sorted(mag["marks"]):
        if u < v:
            a_mark, b_mark = p.m[(v, u)], p.m[(u, v)]
            edges.append({"a": u, "b": v, "marks": [NAMES[a_mark], NAMES[b_mark]],
                          "mag": [NAMES[mag["marks"][(v, u)]], NAMES[mag["marks"][(u, v)]]],
                          "inseparable": frozenset((u, v)) not in linked})
            edges[-1]["kind"] = kind(edges[-1]["marks"])
    kinds: dict[str, int] = {}
    for e in edges:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    return {"background": background, "include_inseparable": inseparable, "valid": mag["maximal"],
            "edges": edges, "converted": mag["converted"], "inseparable": mag["inseparable"],
            "stats": {"links": len(edges), "converted": len(mag["converted"]),
                      "inseparable": len(mag["inseparable"]), "background_marks": len(known),
                      **kinds}}


def kind(marks: list[str]) -> str:
    """leads_to (a -> b either way round), hidden_cause (<->), may_lead (o->), open (o-o)."""
    s = sorted(marks)
    return {("arrow", "tail"): "leads_to", ("arrow", "arrow"): "hidden_cause", ("arrow", "circle"): "may_lead",
            ("circle", "circle"): "open"}.get(tuple(s), "other")
