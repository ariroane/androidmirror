"""Causal discovery of the subject's own model, from the situations it reasons about.

The units are situations (see situations.py), not passages: an insult, an
illness, the world an "if" imagines, each with the level of every variable the
text states or the subject's beliefs imply, and unknown elsewhere. Unknown
stays missing (test-wise deletion), so what a passage happens to be about can
no longer pass for a dependence between two ideas.

`discover`:
1. selects variables (concepts the knowledge graph marks as quantities);
2. runs PC on the situation table for the chosen view (stated + implied by
   default), giving the data's CPDAG;
3. applies the subject's stated model as background knowledge: causal claims
   and "do x so that y" as pins, and tests its stated independences ("it is
   not x but y", "whether x or not") against the data;
4. counts the class before and after, and resamples situations for stability;
5. re-runs on the other views (stated only, everything) so they can be
   compared;
6. if there are vignettes, re-runs on corpus plus vignettes and records exactly
   what the vignettes changed, so the app can draw it in its own colour.

If the situations carry a topic (what their passage is about, see topics.py),
the topic joins every run as a background variable that comes first: PC can
condition on it, so two concepts that only go together because their topic
brings both stay unlinked, and no arrow can point into it. It is not a concept
of the subject's, so it is kept out of the edge list; `topic` reports which
concepts depend on the topic and which links it explains away.

Nothing here calls Claude, so it can be re-run cheaply with other settings.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .. import situations as sit
from .citest import g2_test
from .ida import adjusted_effect, possible_effects
from .pc import apply_pins, pc, tiers_to_pins
from .pdag import PDAG
from .selection import DEFAULTS as SELECTION_DEFAULTS
from .selection import select

DEFAULTS = {"alpha": 0.05, "max_depth": 3, "bootstrap": 20, "block": 5, "seed": 7, "claims_as_pins": True,
            "pin_bases": ["explicit", "temporal"], "tiers": [], "colliders": "majority", "view": sit.DEFAULT_VIEW,
            "compare_views": True, "vignettes": True, "topic": True, "selection": SELECTION_DEFAULTS}
DO = "do:"
TOPIC = sit.TOPIC


def _pc(data: dict[str, list], variables: list[str], cfg: dict[str, Any], rows: list[int] | None = None) -> dict[str, Any]:
    """PC over the variables, with the topic as a first-tier background variable when the data has one.

    The topic's edges are oriented out of it as background knowledge (it is what a passage is about,
    so nothing in a situation causes it), and the result is the CPDAG given that knowledge."""
    if TOPIC not in data:
        return pc(data, variables, cfg["alpha"], cfg["max_depth"], rows, cfg["colliders"])
    found = pc(data, [*variables, TOPIC], cfg["alpha"], cfg["max_depth"], rows, cfg["colliders"])
    full: PDAG = found["cpdag"]
    keep = [v for v in variables if v != TOPIC]
    g = PDAG(keep)  # the topic did its work in the tests; it is not a concept of the subject's
    g.ambiguous = {a for a in full.ambiguous if TOPIC not in a[0] and a[1] != TOPIC}
    # keep the colliders among the subject's concepts and redo Meek's rules without the topic: it is a rough
    # stand-in for what a passage is about, so it may test the subject's links but should not orient them
    for e in full.undirected | {frozenset(d) for d in full.directed}:
        if TOPIC not in e:
            g.undirected.add(e)
    for d in full.directed:
        w = full.why.get(d) or {}
        if TOPIC not in d and w.get("rule") == "v-structure" and TOPIC not in w.get("collider", []):
            g.orient(*d, w)
    g.meek()
    found["cpdag"] = g
    found["topic_links"] = sorted(v for v in keep if full.adjacent(v, TOPIC))
    return found


def claim_pins(claims: list[dict[str, Any]], selected: set[str], bases: list[str],
               stated_model: list[dict[str, Any]] | None = None) -> tuple[list[dict], list[dict]]:
    """The subject's stated, grounded causal claims (and "x so that y") as pins; two-way claims as feedback."""
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for c in claims:
        if c["cause"] in selected and c["effect"] in selected and c.get("held"):
            grounded = sum(n for b, n in c["bases"].items() if b in bases)
            if grounded:
                by_pair[(c["cause"], c["effect"])] = {**c, "grounded": grounded}
    for m in stated_model or []:  # the situations pass: causes, direct, so_that
        if m["kind"] in ("causes", "direct", "so_that") and m["x"] in selected and m["y"] in selected:
            rec = by_pair.setdefault((m["x"], m["y"]), {"cause": m["x"], "effect": m["y"], "sign": 1, "support": 0,
                                                        "grounded": 0, "evidence": []})
            rec["grounded"] += 1
            rec["support"] = rec.get("support", 0) + 1
            rec["evidence"] = rec["evidence"] + [{"passage": m["passage"], "locator": m["locator"], "quote": m["quote"],
                                                  "kind": m["kind"]}]
    pins, feedback = [], []
    for (a, b), c in by_pair.items():
        back = by_pair.get((b, a))
        if back:
            if a < b:
                feedback.append({"pair": [a, b], "support": [c["grounded"], back["grounded"]]})
            if back["grounded"] >= c["grounded"]:
                continue
        pins.append({"src": a, "dst": b, "source": "stated claim", "support": c["grounded"], "sign": c.get("sign", 1),
                     "text": f"stated in {c['support']} passage(s), {c['grounded']} with explicit wording",
                     "evidence": c["evidence"][:3]})
    return pins, feedback


def independence_checks(data: dict[str, list], stated_model: list[dict[str, Any]], selected: set[str],
                        alpha: float) -> list[dict[str, Any]]:
    """The subject's stated independences, tested against the situations."""
    groups: dict[tuple, list] = defaultdict(list)
    for m in stated_model:
        if m["kind"] == "not_x_but_y" and {m["x"], m["y"], m["z"]} <= selected and len({m["x"], m["y"], m["z"]}) == 3:
            groups[(m["x"], m["z"], (m["y"],))].append(m)
        elif m["kind"] == "whether_or_not" and {m["x"], m["y"]} <= selected and m["x"] != m["y"]:
            groups[(m["x"], m["y"], ())].append(m)
    out = []
    for (x, y, given), ms in groups.items():
        t = g2_test(data, x, y, given)
        status = "untestable: too few situations" if not t.enough_data else (
            "agrees" if t.p > alpha else "the data disagrees")
        out.append({"x": x, "y": y, "given": list(given), "kind": ms[0]["kind"], "status": status, "test": t.to_dict(),
                    "evidence": [{"passage": m["passage"], "locator": m["locator"], "quote": m["quote"]} for m in ms[:3]]})
    return out


def block_bootstrap(n: int, block: int, rng: random.Random) -> list[int]:
    starts = list(range(0, max(1, n - block + 1)))
    rows: list[int] = []
    while len(rows) < n:
        s = rng.choice(starts)
        rows.extend(range(s, min(n, s + block)))
    return rows[:n]


def _run(data: dict[str, list], variables: list[str], cfg: dict[str, Any], pins: list[dict[str, Any]],
         bootstrap: bool) -> dict[str, Any]:
    found = _pc(data, variables, cfg)
    cpdag: PDAG = found["cpdag"]
    model, pin_report = apply_pins(cpdag, pins, found["removed"])
    stability = _bootstrap(data, variables, cfg) if bootstrap and cfg["bootstrap"] else {}
    return {"found": found, "cpdag": cpdag, "model": model, "pins": pin_report, "stability": stability}


def discover(situations: list[dict[str, Any]], candidates: list[str], concepts: dict[str, dict[str, Any]],
             claims: list[dict[str, Any]], stated_model: list[dict[str, Any]] | None = None,
             settings: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(settings or {})}
    cfg["selection"] = {**SELECTION_DEFAULTS, **(cfg.get("selection") or {})}
    stated_model = stated_model or []
    claim_counts: Counter = Counter()
    for c in claims:
        claim_counts[c["cause"]] += c["support"]
        claim_counts[c["effect"]] += c["support"]
    matrix, row_ids = sit.table(situations, candidates, cfg["view"], topic=cfg["topic"])
    topic_col = matrix.pop(TOPIC, None)
    report = select(matrix, concepts, claim_counts, cfg["selection"], cfg.get("variables"))
    variables = [r["id"] for r in report if r["status"] == "selected"]
    data = {v: matrix[v] for v in variables}
    if topic_col is not None:
        data[TOPIC] = topic_col
    result: dict[str, Any] = {"settings": cfg, "view": cfg["view"], "selection": report, "variables": variables,
                              "n_rows": len(row_ids), "row_ids": row_ids}
    if len(variables) < 2:
        result["empty"] = True
        return result

    pins, feedback = claim_pins(claims, set(variables), cfg["pin_bases"], stated_model) if cfg["claims_as_pins"] else ([], [])
    base_pins = list(pins)
    if cfg["tiers"]:
        found0 = _pc(data, variables, cfg)
        base_pins += tiers_to_pins(cfg["tiers"], found0["cpdag"])
    base_pins += [{**p, "source": "you", "support": 10**6} for p in cfg.get("pins", [])]
    run = _run(data, variables, cfg, base_pins, bootstrap=True)
    cpdag, model = run["cpdag"], run["model"]
    dag = model.extension() or []
    result.update({
        "cpdag": cpdag.to_dict(), "model": model.to_dict(), "removed": run["found"]["removed"],
        "untestable": sum(1 for r in run["found"]["removed"] if not r.get("enough_data", True)),
        "conflicts": run["found"]["conflicts"], "ambiguous": run["found"]["ambiguous"], "n_tests": run["found"]["n_tests"],
        "depth_reached": run["found"]["depth_reached"],
        "class_size": {"data": cpdag.class_size(), "with_pins": model.class_size()},
        "pins": run["pins"], "feedback_claims": feedback, "stability": run["stability"], "dag": [list(e) for e in dag],
        "edges": _edge_table(data, cpdag, model, dag, run["found"]["edge_support"], run["stability"], claims),
        "independences": independence_checks(data, stated_model, set(variables), cfg["alpha"]),
        "coverage": {v: sum(1 for x in data[v] if x is not None) for v in variables},
    })
    if TOPIC in data:
        result["topic"] = _topic_report(data, variables, cfg, cpdag, run["found"])
    if cfg["compare_views"]:
        result["views"] = _compare_views(situations, variables, cfg, base_pins, model)
    if cfg["vignettes"] and any(s["origin"] == "vignette" for s in situations):
        result["vignettes"] = _with_vignettes(situations, variables, cfg, base_pins, model, claims, result["edges"])
    return result


def _edges_of(g: PDAG) -> dict[frozenset, tuple[str, str] | None]:
    """The edges among the subject's concepts (the topic's edges are background, not part of the model)."""
    out: dict[frozenset, tuple[str, str] | None] = {e: None for e in g.undirected if TOPIC not in e}
    for a, b in g.directed:
        if TOPIC not in (a, b):
            out[frozenset((a, b))] = (a, b)
    return out


def _topic_report(data, variables, cfg, cpdag: PDAG, found: dict[str, Any]) -> dict[str, Any]:
    """Which concepts depend on the topic, and which links holding the topic fixed explains away."""
    without = pc({v: data[v] for v in variables}, variables, cfg["alpha"], cfg["max_depth"], colliders=cfg["colliders"])
    mine, theirs = set(_edges_of(cpdag)), set(_edges_of(without["cpdag"]))
    removed = {frozenset(r["pair"]): r for r in found["removed"]}
    support = found["edge_support"]
    links = [{"variable": v, "test": (support.get("|".join(sorted((v, TOPIC)))) or {}).get("weakest")}
             for v in found.get("topic_links", [])]
    explained = []
    for e in sorted(theirs - mine, key=sorted):
        a, b = sorted(e)
        r = removed.get(e) or {}
        explained.append({"a": a, "b": b, "test": {k: r[k] for k in ("x", "y", "given", "g2", "df", "p", "n", "enough_data") if k in r},
                          "via_topic": TOPIC in r.get("given", [])})
    values = Counter(x for x in data[TOPIC] if x is not None)
    return {"topics": len(values), "links": links, "explained": explained, "added": [sorted(e) for e in sorted(mine - theirs, key=sorted)],
            "conflicts": found.get("topic_conflicts", [])}


def _compare_views(situations, variables, cfg, pins, main_model: PDAG) -> dict[str, Any]:
    """The same variables under each evidence view, so "stated only" and "everything" can be set beside the default."""
    main = _edges_of(main_model)
    out = {}
    for view in sit.VIEWS:
        data, rows = sit.table(situations, variables, view, topic=cfg["topic"])
        found = _pc(data, variables, cfg)
        model, _ = apply_pins(found["cpdag"], pins, found["removed"])
        mine = _edges_of(model)
        out[view] = {"cells": sum(1 for v in variables for x in data[v] if x is not None), "rows": len(rows),
                     "edges": [[*sorted(e), list(d) if d else None] for e, d in sorted(mine.items(), key=lambda kv: sorted(kv[0]))],
                     "class_size": model.class_size(), "shared_with_default": len(set(mine) & set(main)),
                     "only_here": len(set(mine) - set(main)), "missing_here": len(set(main) - set(mine))}
    return out


def _with_vignettes(situations, variables, cfg, pins, corpus_model: PDAG, claims, corpus_edges) -> dict[str, Any]:
    """Discovery on corpus plus vignettes, and exactly what the vignettes changed.

    Vignettes are interventions on the subject's believed world: each sets one
    target variable. Following Joint Causal Inference (Mooij, Magliacane and
    Claassen 2020), each target gets a context variable do:X (1 in the
    vignettes that set X, 0 elsewhere) that comes before every system
    variable. And where the vignettes that set X show Y responding to it, X is
    an ancestor of Y, which orients an X - Y edge (an interventional pin).
    """
    rows = [s for s in situations if s["origin"] in ("corpus", "vignette")]
    targets = sorted({s["target"] for s in rows if s["origin"] == "vignette" and s.get("target") in variables})
    data, row_ids = sit.table(rows, variables, cfg["view"], origins=("corpus", "vignette"), topic=cfg["topic"])
    for t in targets:
        data[DO + t] = [1 if s.get("target") == t else 0 for s in rows]
    contexts = [DO + t for t in targets]
    allv = variables + contexts
    found = _pc(data, allv, cfg)
    vpins = list(pins) + tiers_to_pins([contexts, variables], found["cpdag"]) if contexts else list(pins)
    responses = []
    for t in targets:
        vrows = [i for i, s in enumerate(rows) if s.get("target") == t]
        sub = {v: [data[v][i] for i in vrows] for v in variables}
        for y in variables:
            if y == t:
                continue
            test = g2_test(sub, t, y, ())
            if test.enough_data and test.p <= cfg["alpha"]:
                responses.append({"target": t, "responds": y, "test": test.to_dict()})
                if found["cpdag"].adjacent(t, y):
                    vpins.append({"src": t, "dst": y, "source": "vignettes", "support": 1000,
                                  "text": f"setting {t} in vignettes moves {y} (p = {test.p:.4f})"})
    model, pin_report = apply_pins(found["cpdag"], vpins, found["removed"])
    keep = variables
    proj = PDAG(keep)  # the context variables have done their work; show only the subject's variables
    for a, b in model.directed:
        if a in keep and b in keep:
            proj.directed.add((a, b))
            if (a, b) in model.why:
                proj.why[(a, b)] = model.why[(a, b)]
    proj.undirected = {e for e in model.undirected if e <= set(keep)}
    dag = proj.extension() or []
    sysdata = {v: data[v] for v in keep + ([TOPIC] if TOPIC in data else [])}
    edges = _edge_table(sysdata, found["cpdag"], proj, dag, found["edge_support"], {}, claims)
    corpus_eff = {frozenset((e["a"], e["b"])): e for e in corpus_edges}
    return {"targets": targets, "n_vignettes": sum(1 for s in rows if s["origin"] == "vignette"), "rows": len(row_ids),
            "model": proj.to_dict(), "class_size": proj.class_size(), "pins": pin_report, "responses": responses,
            "edges": edges, "diff": diff(corpus_model, proj, corpus_eff, {frozenset((e["a"], e["b"])): e for e in edges})}


def diff(corpus: PDAG, combined: PDAG, corpus_eff: dict, combined_eff: dict) -> list[dict[str, Any]]:
    """Every way the corpus-plus-vignettes graph differs from the corpus graph, edge by edge."""
    a, b = _edges_of(corpus), _edges_of(combined)
    out = []
    for e in sorted(set(a) | set(b), key=sorted):
        x, y = sorted(e)
        ca, cb_ = a.get(e, "absent"), b.get(e, "absent")
        if e not in b:
            change = "removed by vignettes"
        elif e not in a:
            change = "added by vignettes"
        elif ca is None and cb_ is not None:
            change = "oriented by vignettes"
        elif ca is not None and cb_ is None:
            change = "unoriented by vignettes"
        elif ca != cb_:
            change = "reversed by vignettes"
        else:
            change = "same"
        rec = {"a": x, "b": y, "change": change, "corpus": list(ca) if isinstance(ca, tuple) else ca,
               "with_vignettes": list(cb_) if isinstance(cb_, tuple) else cb_,
               "effect_corpus": (corpus_eff.get(e) or {}).get("effect"),
               "effect_with_vignettes": (combined_eff.get(e) or {}).get("effect")}
        out.append(rec)
    return out


def _bootstrap(data: dict[str, list], variables: list[str], cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    rng = random.Random(cfg["seed"])
    n = len(data[variables[0]])
    tally: dict[str, Counter] = defaultdict(Counter)
    for _ in range(cfg["bootstrap"]):
        rows = block_bootstrap(n, cfg["block"], rng)
        g = _pc(data, variables, cfg, rows)["cpdag"]
        for a, b in g.directed:
            if TOPIC in (a, b):
                continue
            tally["|".join(sorted((a, b)))][f"{a}>{b}"] += 1
            tally["|".join(sorted((a, b)))]["adjacent"] += 1
        for e in [e for e in g.undirected if TOPIC not in e]:
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
    dagset = {tuple(x) for x in dag}
    rows = []
    for e in model.edges():
        a, b = e["src"], e["dst"]
        if TOPIC in (a, b):
            continue
        key = "|".join(sorted((a, b)))
        if e["directed"]:
            if (a, b) in cpdag.directed:
                status = "compelled"
            else:
                rule = (e["why"] or {}).get("rule")
                source = ((e["why"] or {}).get("pin") or {}).get("source")
                status = ("oriented by vignettes" if source == "vignettes" else "pinned") if rule == "pin" else (
                    "implied by pins")
        else:
            status = "reversible"
        src, dst = (a, b) if (a, b) in dagset or not dag else (b, a)
        eff = adjusted_effect(data, src, dst, sorted(dag_parents[src]) + ([TOPIC] if TOPIC in data else []))
        both = sum(1 for x, y in zip(data[a], data[b]) if x is not None and y is not None)
        rows.append({
            "a": a, "b": b, "directed": e["directed"], "status": status, "why": e.get("why"),
            "member_direction": [src, dst], "effect": eff["effect"], "coverage": eff["coverage"], "both_known": both,
            "weakest_test": support.get(key, {}).get("weakest"), "stability": stability.get(key, {}),
            "stated": stated.get(frozenset((a, b)), []),
        })
    return rows


def effects(data: dict[str, list], model: PDAG, x: str, y: str) -> dict[str, Any]:
    """Possible effects of x on y; within topics when the data has them (the topic comes first, so adjusting
    for it is always allowed)."""
    return possible_effects(data, model, x, y, always=[TOPIC] if TOPIC in data else [])
