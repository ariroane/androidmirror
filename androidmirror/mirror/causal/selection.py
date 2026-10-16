"""Choosing the causal variables: which concepts become DAG nodes, and why.

Every candidate is a concept from the knowledge graph that the codebook marks
as a variable. PC's answer is only as good as its variables, so a candidate is
kept only if:

1. it is a quantity: something that can stand higher or lower in a situation
   (people, places, events and abstract entities cannot be intervened on);
2. it has support: known (stated, or implied by the subject's beliefs) in
   enough situations for tests to have any power;
3. it varies: its known values are not nearly all at one level;
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

DEFAULTS = {"min_support": 8, "min_share": 0.015, "min_minority": 0.1, "max_vars": 25, "duplicate_v": 0.8,
            "duplicate_overlap": 15, "claim_weight": 3}


def cramers_v(a: list, b: list) -> tuple[float, int]:
    """Cramér's V over the rows where both are known, and how many rows that is."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if not n:
        return 0.0, 0
    joint = Counter(pairs)
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    if len(ca) < 2 or len(cb) < 2:
        return 0.0, n
    chi2 = sum((joint.get((x, y), 0) - ca[x] * cb[y] / n) ** 2 / (ca[x] * cb[y] / n) for x in ca for y in cb)
    return math.sqrt(chi2 / (n * (min(len(ca), len(cb)) - 1))), n


def select(matrix: dict[str, list], concepts: dict[str, dict[str, Any]], claim_counts: dict[str, int],
           settings: dict[str, Any] | None = None, pinned: list[str] | None = None) -> list[dict[str, Any]]:
    cfg = {**DEFAULTS, **(settings or {})}
    n = len(next(iter(matrix.values()))) if matrix else 0
    floor = max(cfg["min_support"], math.ceil(cfg["min_share"] * n))
    rows: dict[str, dict[str, Any]] = {}
    for cid, col in matrix.items():
        c = concepts.get(cid, {})
        known = [v for v in col if v is not None]
        levels = Counter(known)
        support = len(known)
        minority = (support - max(levels.values())) / support if support else 0.0
        rec = {"id": cid, "label": c.get("label", cid), "kind": c.get("kind", "concept"), "support": support,
               "share": round(support / n, 3) if n else 0.0, "levels": {str(k): v for k, v in sorted(levels.items())},
               "claims": claim_counts.get(cid, 0), "status": "excluded", "reason": ""}
        rec["score"] = support + cfg["claim_weight"] * rec["claims"]
        if not c.get("variable"):
            rec["reason"] = f"not a quantity ({rec['kind']}): it cannot stand higher or lower in a situation"
        elif support < floor:
            rec["reason"] = f"too rare: known in {support} situations, needs {floor}"
        elif minority < cfg["min_minority"]:
            rec["reason"] = f"barely varies: {1 - minority:.0%} of the situations where it is known have the same level"
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
        v, overlap = cramers_v(matrix[a], matrix[b])
        if overlap >= cfg["duplicate_overlap"] and v >= cfg["duplicate_v"]:  # `a` ranks higher, so it stays
            dropped.add(b)
            rows[b]["status"] = "excluded"
            rows[b]["reason"] = (f"near-duplicate of {rows[a]['label']} (Cramér's V {v:.2f} over {overlap} situations): "
                                 "likely one idea or a definition, which would break faithfulness")
            rows[b]["duplicate_of"] = a
    kept = 0
    for r in cands:
        if r["id"] in dropped:
            continue
        if kept < cfg["max_vars"] or r["id"] in pinned_set:
            r["status"], r["reason"] = "selected", ("chosen by you" if r["id"] in pinned_set else
                                                    f"known in {r['support']} situations; {r['claims']} stated causal claims")
            kept += 1
        else:
            r["status"], r["reason"] = "excluded", f"outside the top {cfg['max_vars']} by support and stated claims"
    return sorted(rows.values(), key=lambda r: (r["status"] != "selected", -r["score"]))
