"""Causal discovery over the corpus treated as an observational dataset.

`discover` runs the whole causal layer from a built Mirror's measurements:
select variables, run PC to get the data's CPDAG (its Markov equivalence
class), apply the subject's stated causal claims (and any tiers) as background
knowledge, count the class before and after, and resample passages to see how
stable each edge is. Nothing here calls Claude, so it can be re-run cheaply
with different settings.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .ida import adjusted_effect, possible_effects
from .pc import apply_pins, pc, tiers_to_pins
from .pdag import PDAG
from .selection import DEFAULTS as SELECTION_DEFAULTS
from .selection import select

DEFAULTS = {"alpha": 0.05, "max_depth": 3, "bootstrap": 20, "block": 5, "seed": 7, "claims_as_pins": True,
            "pin_bases": ["explicit", "temporal"], "tiers": [], "colliders": "majority", "selection": SELECTION_DEFAULTS}


def claim_pins(claims: list[dict[str, Any]], selected: set[str], bases: list[str]) -> tuple[list[dict], list[dict]]:
    """Turn the subject's stated, grounded causal claims into pins; report two-way claims as feedback."""
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for c in claims:
        if c["cause"] in selected and c["effect"] in selected and c.get("held"):
            grounded = sum(n for b, n in c["bases"].items() if b in bases)
            if grounded:
                by_pair[(c["cause"], c["effect"])] = {**c, "grounded": grounded}
    pins, feedback = [], []
    for (a, b), c in by_pair.items():
        back = by_pair.get((b, a))
        if back:
            if a < b:
                feedback.append({"pair": [a, b], "support": [c["grounded"], back["grounded"]]})
            if back["grounded"] >= c["grounded"]:
                continue
        pins.append({"src": a, "dst": b, "source": "stated claim", "support": c["grounded"], "sign": c["sign"],
                     "text": f"stated in {c['support']} passage(s), {c['grounded']} with explicit or temporal wording",
                     "evidence": c["evidence"][:3]})
    return pins, feedback


def block_bootstrap(n: int, block: int, rng: random.Random) -> list[int]:
    starts = list(range(0, max(1, n - block + 1)))
    rows: list[int] = []
    while len(rows) < n:
        s = rng.choice(starts)
        rows.extend(range(s, min(n, s + block)))
    return rows[:n]


def discover(matrix: dict[str, list[int]], concepts: dict[str, dict[str, Any]], claims: list[dict[str, Any]],
             settings: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(settings or {})}
    cfg["selection"] = {**SELECTION_DEFAULTS, **(cfg.get("selection") or {})}
    claim_counts: Counter = Counter()
    for c in claims:
        claim_counts[c["cause"]] += c["support"]
        claim_counts[c["effect"]] += c["support"]
    report = select(matrix, concepts, claim_counts, cfg["selection"], cfg.get("variables"))
    variables = [r["id"] for r in report if r["status"] == "selected"]
    data = {v: matrix[v] for v in variables}
    n = len(next(iter(matrix.values()))) if matrix else 0
    result: dict[str, Any] = {"settings": cfg, "selection": report, "variables": variables, "n_rows": n}
    if len(variables) < 2:
        result["empty"] = True
        return result

    found = pc(data, variables, cfg["alpha"], cfg["max_depth"], colliders=cfg["colliders"])
    cpdag: PDAG = found["cpdag"]
    pins, feedback = claim_pins(claims, set(variables), cfg["pin_bases"]) if cfg["claims_as_pins"] else ([], [])
    pins += tiers_to_pins(cfg["tiers"], cpdag) if cfg["tiers"] else []
    pins += [{**p, "source": "you", "support": 10**6} for p in cfg.get("pins", [])]
    model, pin_report = apply_pins(cpdag, pins, found["removed"])

    stability = _bootstrap(data, variables, cfg) if cfg["bootstrap"] else {}
    dag = model.extension() or []
    result.update({
        "cpdag": cpdag.to_dict(), "model": model.to_dict(), "removed": found["removed"],
        "conflicts": found["conflicts"], "ambiguous": found["ambiguous"], "n_tests": found["n_tests"], "depth_reached": found["depth_reached"],
        "class_size": {"data": cpdag.class_size(), "with_pins": model.class_size()},
        "pins": pin_report, "feedback_claims": feedback, "stability": stability, "dag": [list(e) for e in dag],
        "edges": _edge_table(data, cpdag, model, dag, found["edge_support"], stability, claims),
    })
    return result


def _bootstrap(data: dict[str, list[int]], variables: list[str], cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    rng = random.Random(cfg["seed"])
    n = len(data[variables[0]])
    tally: dict[str, Counter] = defaultdict(Counter)
    for _ in range(cfg["bootstrap"]):
        rows = block_bootstrap(n, cfg["block"], rng)
        g = pc(data, variables, cfg["alpha"], cfg["max_depth"], rows, cfg["colliders"])["cpdag"]
        for a, b in g.directed:
            tally["|".join(sorted((a, b)))][f"{a}>{b}"] += 1
            tally["|".join(sorted((a, b)))]["adjacent"] += 1
        for e in g.undirected:
            tally["|".join(sorted(e))]["undirected"] += 1
            tally["|".join(sorted(e))]["adjacent"] += 1
    runs = cfg["bootstrap"]
    return {k: {kk: round(v / runs, 3) for kk, v in c.items()} | {"runs": runs} for k, c in tally.items()}


def _edge_table(data, cpdag: PDAG, model: PDAG, dag, support, stability, claims) -> list[dict[str, Any]]:
    stated = defaultdict(list)
    for c in claims:
        stated[frozenset((c["cause"], c["effect"]))].append(
            {"cause": c["cause"], "effect": c["effect"], "sign": c["sign"], "support": c["support"], "held": c.get("held")})
    dag_parents = defaultdict(list)
    for a, b in dag:
        dag_parents[b].append(a)
    rows = []
    for e in model.edges():
        a, b = e["src"], e["dst"]
        key = "|".join(sorted((a, b)))
        if e["directed"]:
            status = "compelled" if (a, b) in cpdag.directed else (
                "pinned" if (e["why"] or {}).get("rule") == "pin" else "implied by pins")
        else:
            status = "reversible"
        # direction in the chosen member, and the effect along it
        src, dst = (a, b) if (a, b) in {tuple(x) for x in dag} or not dag else (b, a)
        eff = adjusted_effect(data, src, dst, sorted(dag_parents[src]))
        rows.append({
            "a": a, "b": b, "directed": e["directed"], "status": status, "why": e.get("why"),
            "member_direction": [src, dst], "effect": eff["effect"], "coverage": eff["coverage"],
            "weakest_test": support.get(key, {}).get("weakest"), "stability": stability.get(key, {}),
            "stated": stated.get(frozenset((a, b)), []),
        })
    return rows


def effects(data: dict[str, list[int]], model: PDAG, x: str, y: str) -> dict[str, Any]:
    return possible_effects(data, model, x, y)
