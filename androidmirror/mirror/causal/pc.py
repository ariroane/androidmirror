"""The PC algorithm (order-independent "PC-stable" variant) with provenance.

From the passage-by-variable table, PC removes every edge whose two variables
test as independent given some set of their neighbours, finds the colliders
(x -> z <- y where z did not separate x and y), and applies Meek's rules. The
result is the CPDAG of the Markov equivalence class the data supports, under
the usual assumptions (causal sufficiency, faithfulness, independent units).

Every removed edge keeps the test that removed it, every surviving edge keeps
its weakest test (largest p-value seen), and every arrow keeps the rule that
made it, so the class can be inspected without listing its members.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .citest import Data, as_arrays, g2_test
from .pdag import PDAG


def pc(data: Data, variables: list[str], alpha: float = 0.05, max_depth: int = 3,
       rows: list[int] | None = None, colliders: str = "majority", untestable: str = "keep",
       require_signature: bool = True) -> dict[str, Any]:
    """PC-stable. `untestable` says what a conditional test with too little data does. With "keep"
    (the default), an edge needs a testable dependence to exist at all (a marginal test with too few
    rows removes it, as untestable), but a conditional test with too few rows cannot show
    independence, so it leaves the edge in place. "remove" is pcalg's convention throughout, which
    with unknown values deletes every edge whose conditional tests run short of rows.

    `require_signature` orients a collider x -> z <- y only if x and y also test as dependent given
    z (plus the separating set), on enough rows: the collider's own fingerprint. With complete data
    this rarely changes anything; with unknown values it stops a collider from being read off two
    tests that saw different situations."""
    vs = sorted(variables)
    data = as_arrays({v: data[v] for v in vs})
    if rows is not None:  # a resample: take its rows once, not in every test
        idx = list(rows)
        data = {v: col[idx] for v, col in data.items()}
        rows = None
    adj = {v: set(vs) - {v} for v in vs}
    sepset: dict[frozenset, dict[str, Any]] = {}
    kept: dict[frozenset, dict[str, Any]] = {}
    n_tests, depth = 0, 0
    while True:
        snapshot = {v: set(adj[v]) for v in vs}
        tested = False
        for x in vs:
            for y in sorted(snapshot[x]):
                if y not in adj[x]:
                    continue
                cands = sorted(snapshot[x] - {y})
                if len(cands) < depth:
                    continue
                tested = True
                for s in combinations(cands, depth):
                    t = g2_test(data, x, y, s, rows)
                    n_tests += 1
                    key = frozenset((x, y))
                    info = kept.setdefault(key, {"max_p": 0.0, "tests": 0, "weakest": None})
                    info["tests"] += 1
                    if not t.enough_data and untestable == "keep" and depth > 0:
                        info["untestable"] = info.get("untestable", 0) + 1
                        continue
                    if t.p > info["max_p"] or info["weakest"] is None:
                        info["max_p"], info["weakest"] = t.p, t.to_dict()
                    if t.p > alpha:
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sepset[key] = t.to_dict()
                        break
        if not tested or depth >= max_depth:
            break
        depth += 1

    g = PDAG(vs)
    for x in vs:
        for y in adj[x]:
            if x < y:
                g.add_undirected(x, y)

    # colliders. "majority" (Colombo and Maathuis 2014) re-tests each unshielded triple against every
    # separating set drawn from the final neighbourhoods, which keeps one unlucky test from
    # orienting edges; "standard" uses the first separating set found.
    arrows: dict[tuple[str, str], dict[str, Any]] = {}
    ambiguous, untested, unconfirmed = [], [], []
    sep_cache: dict[frozenset, list[dict[str, Any]]] = {}
    for z in vs:
        for x, y in combinations(sorted(adj[z]), 2):
            if y in adj[x]:
                continue
            sep = sepset.get(frozenset((x, y)))
            if sep is None:
                continue
            if not sep.get("enough_data", True):  # never shown independent, so no evidence for a collider
                untested.append({"triple": [x, z, y], "pair": sorted((x, y))})
                continue
            if colliders == "majority":
                key = frozenset((x, y))
                if key not in sep_cache:
                    sep_cache[key] = _separating_sets(data, x, y, adj, alpha, max_depth, rows, untestable) or [sep]
                seps = sep_cache[key]
                with_z = sum(1 for t in seps if z in t["given"])
                if 2 * with_z == len(seps):
                    ambiguous.append({"triple": [x, z, y], "separating_sets": len(seps), "containing": with_z})
                    continue
                if 2 * with_z > len(seps):
                    continue
                share = f"{len(seps) - with_z} of {len(seps)} separating sets leave {z} out"
                sep = max((t for t in seps if z not in t["given"]), key=lambda t: t["p"])
            else:
                if z in sep["given"]:
                    continue
                share = "the separating set leaves it out"
            # the collider's own signature: x and y, independent without z, become dependent given z.
            # With unknown values each test sees different rows, so this is checked, not assumed.
            sig = g2_test(data, x, y, tuple(sorted(set(sep["given"]) | {z})), rows)
            if require_signature and not (sig.enough_data and sig.p <= alpha):
                unconfirmed.append({"triple": [x, z, y], "test": sig.to_dict()})
                continue
            given = ", ".join(sep["given"]) or "nothing"
            why = {"rule": "v-structure", "collider": [x, z, y], "test": sep, "signature": sig.to_dict(),
                   "text": f"{x} and {y} are independent given {given} (p = {sep['p']}), and {z} is not needed "
                           f"to separate them ({share})"}
            arrows.setdefault((x, z), why)
            arrows.setdefault((y, z), why)
    conflicts = []
    for t in ambiguous + untested + unconfirmed:
        x, z, y = t["triple"]
        g.ambiguous.add((frozenset((x, y)), z))
    for (a, b), why in arrows.items():
        if (b, a) in arrows:
            if a < b:
                conflicts.append({"edge": [a, b], "colliders": [why["collider"], arrows[(b, a)]["collider"]]})
            continue
        g.orient(a, b, why)
    g.meek()
    if g.has_directed_cycle() or g.extension() is None:
        conflicts.append({"edge": None, "note": "the colliders found are not jointly consistent: no DAG fits them all"})

    edge_support = {"|".join(sorted(k)): v for k, v in kept.items() if g.adjacent(*sorted(k))}
    removed = [{"pair": sorted(k), **v} for k, v in sorted(sepset.items(), key=lambda kv: sorted(kv[0]))]
    return {"cpdag": g, "edge_support": edge_support, "removed": removed, "conflicts": conflicts, "ambiguous": ambiguous, "untested_triples": untested, "unconfirmed_colliders": unconfirmed,
            "n_tests": n_tests, "depth_reached": depth, "alpha": alpha, "max_depth": max_depth}


def _separating_sets(data, x, y, adj, alpha, max_depth, rows, untestable="keep") -> list[dict[str, Any]]:
    out = []
    seen = set()
    for side in (adj[x] - {y}, adj[y] - {x}):
        for k in range(min(max_depth, len(side)) + 1):
            for s in combinations(sorted(side), k):
                if s in seen:
                    continue
                seen.add(s)
                t = g2_test(data, x, y, s, rows)
                if t.p > alpha and (t.enough_data or untestable == "remove"):
                    out.append(t.to_dict())
    return out


def apply_pins(cpdag: PDAG, pins: list[dict[str, Any]], removed: list[dict[str, Any]]) -> tuple[PDAG, list[dict[str, Any]]]:
    """Add background knowledge (required orientations) one at a time, strongest first.

    A pin is kept only if the class still has a member afterwards: it may orient
    a reversible edge, and Meek's rules then carry it further, but it can never
    reverse a compelled edge or create a collider the data did not find.
    """
    g = cpdag.copy()
    sep = {frozenset(r["pair"]): r for r in removed}
    out = []
    for pin in sorted(pins, key=lambda p: -p.get("support", 1)):
        a, b = pin["src"], pin["dst"]
        rec = dict(pin)
        if a not in g.nodes or b not in g.nodes:
            rec["status"] = "outside the model"
        elif (a, b) in g.directed:
            rec["status"] = "agrees" if (a, b) in cpdag.directed else "already implied"
        elif (b, a) in g.directed:
            rec["status"] = "contradicted by the data" if (b, a) in cpdag.directed else "contradicted by a stronger pin"
            rec["why_not"] = g.why.get((b, a))
        elif not g.adjacent(a, b):
            why_not = sep.get(frozenset((a, b)))
            untestable = why_not is not None and not why_not.get("enough_data", True)
            rec["status"] = "untestable: too few situations state both" if untestable else "no link in the data"
            rec["why_not"] = why_not
        else:
            trial = g.copy()
            trial.orient(a, b, {"rule": "pin", "text": pin.get("text", "background knowledge"), "pin": pin})
            implied = trial.meek()
            if trial.has_directed_cycle() or trial.extension() is None:
                rec["status"] = "rejected: no DAG in the class fits it"
            else:
                g = trial
                rec["status"] = "applied"
                rec["implied"] = [list(e) for e in implied]
        out.append(rec)
    return g, out


def tiers_to_pins(tiers: list[list[str]], cpdag: PDAG) -> list[dict[str, Any]]:
    """An ordering of variables (earlier tiers can cause later ones, never the reverse) as pins."""
    rank = {v: i for i, tier in enumerate(tiers) for v in tier}
    pins = []
    for e in sorted(cpdag.undirected | {frozenset(d) for d in cpdag.directed}, key=sorted):
        a, b = sorted(e)
        if a in rank and b in rank and rank[a] != rank[b]:
            src, dst = (a, b) if rank[a] < rank[b] else (b, a)
            pins.append({"src": src, "dst": dst, "source": "tier", "support": 0,
                         "text": f"tier {rank[src] + 1} comes before tier {rank[dst] + 1}"})
    return pins
