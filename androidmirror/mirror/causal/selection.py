"""Choosing the causal variables: which concepts become DAG nodes, and why.

PC's answer is only as good as its variables. A candidate is kept only if:

1. it is a quantity: something that can be more or less present in a passage
   (people, places, events and abstract entities cannot be intervened on);
2. it has support: engaged in enough passages for tests to have any power;
3. it varies: not engaged in nearly every passage;
4. it is not a near-duplicate of a stronger variable: near-deterministic pairs
   are usually one idea under two names or a definition, which breaks
   faithfulness and belongs in the knowledge graph;
5. it ranks in the top `max_vars` by support plus the subject's own causal
   claims about it, which keeps the class small enough to read.

Every candidate gets a recorded decision and reason, shown in the Causal tab.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any

DEFAULTS = {"min_support": 8, "min_share": 0.015, "max_share": 0.6, "max_vars": 25, "duplicate_v": 0.8,
            "claim_weight": 3}


def cramers_v(a: list[int], b: list[int]) -> float:
    n = len(a)
    joint = Counter(zip(a, b))
    ca, cb = Counter(a), Counter(b)
    if len(ca) < 2 or len(cb) < 2:
        return 0.0
    chi2 = sum((joint.get((x, y), 0) - ca[x] * cb[y] / n) ** 2 / (ca[x] * cb[y] / n) for x in ca for y in cb)
    return math.sqrt(chi2 / (n * (min(len(ca), len(cb)) - 1)))


def select(matrix: dict[str, list[int]], concepts: dict[str, dict[str, Any]], claim_counts: dict[str, int],
           settings: dict[str, Any] | None = None, pinned: list[str] | None = None) -> list[dict[str, Any]]:
    cfg = {**DEFAULTS, **(settings or {})}
    n = len(next(iter(matrix.values()))) if matrix else 0
    floor = max(cfg["min_support"], math.ceil(cfg["min_share"] * n))
    rows: dict[str, dict[str, Any]] = {}
    for cid, col in matrix.items():
        c = concepts.get(cid, {})
        support = sum(1 for v in col if v)
        rec = {"id": cid, "label": c.get("label", cid), "kind": c.get("kind", "concept"), "support": support,
               "share": round(support / n, 3) if n else 0.0, "negative": sum(1 for v in col if v < 0),
               "claims": claim_counts.get(cid, 0), "status": "excluded", "reason": ""}
        rec["score"] = support + cfg["claim_weight"] * rec["claims"]
        if not c.get("variable"):
            rec["reason"] = f"not a quantity ({rec['kind']}): it cannot be more or less present in a passage"
        elif support < floor:
            rec["reason"] = f"too rare: engaged in {support} passages, needs {floor}"
        elif rec["share"] > cfg["max_share"]:
            rec["reason"] = f"barely varies: engaged in {rec['share']:.0%} of passages"
        else:
            rec["status"] = "candidate"
        rows[cid] = rec
    pinned_set = set(pinned or [])
    cands = sorted((r for r in rows.values() if r["status"] == "candidate"),
                   key=lambda r: (r["id"] not in pinned_set, -r["score"], r["id"]))
    dropped: set[str] = set()
    for a, b in combinations([r["id"] for r in cands], 2):
        if a in dropped or b in dropped:
            continue
        v = cramers_v(matrix[a], matrix[b])
        if v >= cfg["duplicate_v"]:  # `a` ranks higher, so it stays
            dropped.add(b)
            rows[b]["status"] = "excluded"
            rows[b]["reason"] = (f"near-duplicate of {rows[a]['label']} (Cramér's V {v:.2f}): likely one idea or a "
                                 "definition, which would break faithfulness")
            rows[b]["duplicate_of"] = a
    kept = 0
    for r in cands:
        if r["id"] in dropped:
            continue
        if kept < cfg["max_vars"] or r["id"] in pinned_set:
            r["status"], r["reason"] = "selected", ("chosen by you" if r["id"] in pinned_set else
                                                    f"engaged in {r['support']} passages; {r['claims']} stated causal claims")
            kept += 1
        else:
            r["status"], r["reason"] = "excluded", f"outside the top {cfg['max_vars']} by support and stated claims"
    return sorted(rows.values(), key=lambda r: (r["status"] != "selected", -r["score"]))
