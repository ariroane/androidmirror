"""Partially directed graphs: CPDAGs, Meek's rules, extension to a DAG, class size.

A CPDAG stands for a whole Markov equivalence class: its directed edges are
compelled (the same in every member) and its undirected edges are reversible.
Every orientation records why it was made (a v-structure from a CI test, one of
Meek's rules from other orientations, or a pin), so any arrow can be traced back
to the tests and passages that hold it up.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from typing import Any

Edge = tuple[str, str]


class PDAG:
    def __init__(self, nodes: list[str]) -> None:
        self.nodes = list(nodes)
        self.directed: set[Edge] = set()
        self.undirected: set[frozenset] = set()
        self.why: dict[Edge, dict[str, Any]] = {}

    # ---------- structure ----------

    def copy(self) -> "PDAG":
        g = PDAG(self.nodes)
        g.directed, g.undirected, g.why = set(self.directed), set(self.undirected), dict(self.why)
        return g

    def add_undirected(self, a: str, b: str) -> None:
        self.undirected.add(frozenset((a, b)))

    def remove(self, a: str, b: str) -> None:
        self.undirected.discard(frozenset((a, b)))
        self.directed.discard((a, b))
        self.directed.discard((b, a))

    def is_undirected(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.undirected

    def adjacent(self, a: str, b: str) -> bool:
        return self.is_undirected(a, b) or (a, b) in self.directed or (b, a) in self.directed

    def adj(self, x: str) -> set[str]:
        return {n for n in self.nodes if n != x and self.adjacent(x, n)}

    def und_nbrs(self, x: str) -> set[str]:
        return {next(iter(e - {x})) for e in self.undirected if x in e}

    def parents(self, x: str) -> set[str]:
        return {a for a, b in self.directed if b == x}

    def children(self, x: str) -> set[str]:
        return {b for a, b in self.directed if a == x}

    def orient(self, a: str, b: str, why: dict[str, Any]) -> bool:
        """Turn a - b into a -> b. False if it is not undirected."""
        if not self.is_undirected(a, b):
            return False
        self.undirected.discard(frozenset((a, b)))
        self.directed.add((a, b))
        self.why[(a, b)] = why
        return True

    def edges(self) -> list[dict[str, Any]]:
        out = [{"src": a, "dst": b, "directed": True, "why": self.why.get((a, b))} for a, b in sorted(self.directed)]
        out += [{"src": a, "dst": b, "directed": False} for a, b in sorted(tuple(sorted(e)) for e in self.undirected)]
        return out

    def has_directed_cycle(self) -> bool:
        succ: dict[str, set[str]] = {n: set() for n in self.nodes}
        for a, b in self.directed:
            succ[a].add(b)
        state: dict[str, int] = {}

        def visit(n: str) -> bool:
            state[n] = 1
            for m in succ[n]:
                if state.get(m) == 1 or (m not in state and visit(m)):
                    return True
            state[n] = 2
            return False

        return any(n not in state and visit(n) for n in self.nodes)

    # ---------- Meek's rules ----------

    def meek(self, rules: tuple[int, ...] = (1, 2, 3, 4)) -> list[Edge]:
        """Apply Meek's orientation rules to a fixpoint. Returns the edges oriented."""
        done: list[Edge] = []
        changed = True
        while changed:
            changed = False
            for e in sorted(self.undirected, key=sorted):  # a fixed order keeps results reproducible
                a, b = sorted(e)
                for x, y in ((a, b), (b, a)):
                    if not self.is_undirected(x, y):
                        break
                    hit = self._meek_one(x, y, rules)
                    if hit and self.orient(x, y, hit):
                        done.append((x, y))
                        changed = True
                        break
        return done

    def _meek_one(self, a: str, b: str, rules: tuple[int, ...]) -> dict[str, Any] | None:
        """Why a - b must become a -> b, or None."""
        if 1 in rules:
            for c in sorted(self.parents(a)):  # c -> a - b, c not adjacent b
                if c != b and not self.adjacent(c, b):
                    return {"rule": "R1", "text": f"{c} -> {a} and {c} is not adjacent to {b}: {b} -> {a} would make a new collider",
                            "via": [[c, a]]}
        if 2 in rules:
            for c in sorted(self.children(a)):  # a -> c -> b
                if (c, b) in self.directed:
                    return {"rule": "R2", "text": f"{a} -> {c} -> {b}: {b} -> {a} would close a cycle", "via": [[a, c], [c, b]]}
        if 3 in rules:
            ps = sorted(c for c in self.parents(b) if self.is_undirected(a, c))
            for c, d in combinations(ps, 2):  # a - c -> b, a - d -> b, c not adj d
                if not self.adjacent(c, d):
                    return {"rule": "R3", "text": f"{c} -> {b} <- {d} with {a} adjacent to both", "via": [[c, b], [d, b]]}
        if 4 in rules:
            for l in sorted(self.parents(b)):  # a adj l, k -> l -> b, a adj k, k not adj b
                if not self.adjacent(a, l):
                    continue
                for k in sorted(self.parents(l)):
                    if k != a and self.adjacent(a, k) and not self.adjacent(k, b):
                        return {"rule": "R4", "text": f"{k} -> {l} -> {b} with {a} adjacent to {k} and {l}",
                                "via": [[k, l], [l, b]]}
        return None

    # ---------- members of the class ----------

    def extension(self) -> list[Edge] | None:
        """One DAG in the class (Dor and Tarsi 1992), or None if there is none."""
        g = self.copy()
        remaining = set(g.nodes)
        dag = set(g.directed)
        while remaining:
            for x in sorted(remaining):
                if any(a == x and b in remaining for a, b in g.directed):
                    continue  # not a sink
                nb = {n for n in g.und_nbrs(x) if n in remaining}
                adj = {n for n in g.adj(x) if n in remaining}
                if all(all(g.adjacent(y, z) for z in adj if z != y) for y in nb):
                    for y in nb:
                        dag.add((y, x))
                        g.undirected.discard(frozenset((x, y)))
                    remaining.discard(x)
                    g.directed = {(a, b) for a, b in g.directed if a != x and b != x}
                    g.undirected = {e for e in g.undirected if x not in e}
                    break
            else:
                return None
        return sorted(dag)

    def chain_components(self) -> list[set[str]]:
        seen: set[str] = set()
        out = []
        for n in self.nodes:
            if n in seen or not self.und_nbrs(n):
                continue
            comp, stack = set(), [n]
            while stack:
                m = stack.pop()
                if m in comp:
                    continue
                comp.add(m)
                stack.extend(self.und_nbrs(m) - comp)
            seen |= comp
            out.append(comp)
        return out

    def class_size(self, max_component: int = 22) -> dict[str, Any]:
        """How many DAGs the class holds, counted exactly per chain component (He, Jia and Yu 2015)."""
        comps = self.chain_components()
        sizes, log10 = [], 0.0
        exact = True
        for comp in comps:
            if len(comp) > max_component:
                sizes.append({"nodes": sorted(comp), "count": None})
                exact = False
                continue
            edges = frozenset(e for e in self.undirected if e <= comp)
            n = _count_amo(frozenset(comp), edges)
            sizes.append({"nodes": sorted(comp), "count": n})
            log10 += math.log10(n)
        return {"log10": round(log10, 3) if exact else None, "count": round(10 ** log10) if exact and log10 < 15 else None,
                "components": sizes, "undirected_edges": len(self.undirected), "directed_edges": len(self.directed)}

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "directed": [[a, b, self.why.get((a, b))] for a, b in sorted(self.directed)],
                "undirected": [sorted(e) for e in sorted(self.undirected, key=sorted)]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PDAG":
        g = cls(d["nodes"])
        for a, b, why in d["directed"]:
            g.directed.add((a, b))
            if why:
                g.why[(a, b)] = why
        for a, b in d["undirected"]:
            g.add_undirected(a, b)
        return g


@lru_cache(maxsize=4096)
def _count_amo(nodes: frozenset, edges: frozenset) -> int:
    """Acyclic moral orientations of a connected chordal graph: sum over roots."""
    if len(nodes) <= 1:
        return 1
    if len(edges) == len(nodes) * (len(nodes) - 1) // 2:
        return math.factorial(len(nodes))  # a clique: every order works
    total = 0
    for v in nodes:
        g = PDAG(sorted(nodes))
        g.undirected = set(edges)
        for e in edges:
            if v in e:
                g.orient(v, next(iter(e - {v})), {"rule": "root"})
        g.meek(rules=(1, 2, 3))
        prod = 1
        for comp in g.chain_components():
            prod *= _count_amo(frozenset(comp), frozenset(e for e in g.undirected if e <= comp))
        total += prod
    return total
