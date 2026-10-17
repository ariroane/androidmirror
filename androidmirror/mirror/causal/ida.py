"""Possible causal effects across the equivalence class, without listing it (local IDA).

Maathuis, Kalisch and Buhlmann (2009): in each DAG of the class, the effect of
X on Y can be estimated by adjusting for X's parents. X's possible parent sets
are its compelled parents plus any clique of its undirected neighbours, so the
set of possible effects comes from X's neighbourhood alone.

The effect is the adjusted contrast E[Y | do(X = high)] - E[Y | do(X = low)],
on the -1..1 scale, within strata of the parent set: how differently Y stands
in the subject's situations when X is high than when it is low. Rows with an
unknown value are left out, strata where one side is empty are skipped, and
`coverage` reports the share of rows that could be used.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from .citest import Data
from .pdag import PDAG

MAX_NBRS = 10


def adjusted_effect(data: Data, x: str, y: str, adjust: list[str], rows: list[int] | None = None,
                    hi: int = 1, lo: int = -1) -> dict[str, Any]:
    idx = list(rows) if rows is not None else list(range(len(data[x])))
    strata: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i in idx:
        if data[x][i] is None or data[y][i] is None or any(data[v][i] is None for v in adjust):
            continue
        strata[tuple(data[v][i] for v in adjust)][data[x][i]].append(data[y][i])
    total, used, est = len(idx), 0, 0.0
    for cell in strata.values():
        a, b = cell.get(hi), cell.get(lo)
        if not a or not b:
            continue
        w = sum(len(v) for v in cell.values())
        used += w
        est += w * (sum(a) / len(a) - sum(b) / len(b))
    if not used:
        return {"effect": None, "coverage": 0.0, "adjust": adjust}
    return {"effect": round(est / used, 4), "coverage": round(used / total, 3), "adjust": adjust}


def possible_parent_sets(g: PDAG, x: str) -> list[list[str]]:
    pa = sorted(g.parents(x))
    und = sorted(g.und_nbrs(x))[:MAX_NBRS]
    out = []
    for k in range(len(und) + 1):
        for s in combinations(und, k):
            if all(g.adjacent(a, b) for a, b in combinations(s, 2)):
                out.append(sorted(pa + list(s)))
    return out


def possible_effects(data: Data, g: PDAG, x: str, y: str, rows: list[int] | None = None) -> dict[str, Any]:
    results = []
    for ps in possible_parent_sets(g, x):
        if y in ps:
            results.append({"effect": 0.0, "coverage": 1.0, "adjust": ps, "note": f"{y} is a parent of {x} here"})
        else:
            results.append(adjusted_effect(data, x, y, ps, rows))
    vals = [r["effect"] for r in results if r["effect"] is not None]
    summary: dict[str, Any] = {"x": x, "y": y, "members": results, "n_parent_sets": len(results)}
    if vals:
        lo, hi = min(vals), max(vals)
        summary.update({"min": lo, "max": hi, "min_abs": min(abs(v) for v in vals),
                        "sign": "+" if lo > 0 else "-" if hi < 0 else "mixed",
                        "identified": len(set(vals)) == 1})
    return summary
