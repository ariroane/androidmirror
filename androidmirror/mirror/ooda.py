"""The OODA model: the Mirror's concepts placed in four tiers (Observe, Orient, Decide, Act),
linked by the subject's own causal judgments and stated facts.

It is a third view beside the knowledge graph and the causal model, built from both and
from nothing else: no Claude calls, and nothing in either graph is changed.

Where each concept goes is scored from evidence in the Mirror, and every score keeps its
reasons so the placement can be checked:

- the stage of the passages it is mentioned in, when the corpus is grouped by OODA stage
  (the OODA interview is): the strongest signal;
- its kind: things in the world (agents, events, objects, places) are observed; beliefs,
  dispositions, emotions, faculties and abstract ideas orient; practices are acts;
- the knowledge graph: what the subject practices or embodies is an act; what the subject
  values, rejects, or calls good or bad is what decisions weigh; whom the subject learned
  from orients;
- the causal model: what the subject urges more or less of is decided on; a cause that
  nothing else in the model causes is an input, so observed;
- a placement you made yourself, which overrides the rest and survives a rebuild
  (ooda_overrides.json in the Mirror's folder).

Links between concepts are the causal model's arrows, the subject's claims the causal model
holds aside because they would close a cycle (a loop is what OODA is about), and, secondarily,
stated facts,
classified by the tiers they join: forward (the next tier), within a tier, skipping ahead
(Boyd's "implicit guidance and control", e.g. Orient straight to Act), feedback (from Act
back to an earlier tier, closing the loop) and backward.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schema import EVALUATIONS, SELF

TIERS = ("observe", "orient", "decide", "act")
LABEL = {"observe": "Observe", "orient": "Orient", "decide": "Decide", "act": "Act"}
ABOUT = {
    "observe": "what is taken in: things, people and events in the world, and the inputs nothing else causes",
    "orient": "how it is read: beliefs, dispositions, feelings, faculties and the traditions they come from",
    "decide": "what is weighed and chosen: what is valued or rejected, called good or bad, wanted more or less of",
    "act": "what is done: practices and what is carried out",
}
KIND_SCORE = {
    "agent": {"observe": 1.0}, "event": {"observe": 1.0}, "object": {"observe": 1.0}, "place": {"observe": 1.0},
    "state": {"observe": 0.5, "orient": 0.5},
    "belief": {"orient": 1.0}, "disposition": {"orient": 1.0}, "emotion": {"orient": 1.0},
    "faculty": {"orient": 1.0}, "concept": {"orient": 0.5}, "class": {"orient": 0.5},
    "practice": {"act": 1.5},
}
STAGE_WEIGHT = 3.0  # share of mentions in a stage's passages, times this
OVERRIDES = "ooda_overrides.json"


def _overrides(root: Path | None) -> dict[str, Any]:
    p = root / OVERRIDES if root else None
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else {}


def place(root: Path, concept: str, tier: str | None, by: str = "") -> dict[str, Any]:
    """Put a concept in a tier yourself (tier None: back to the automatic placement)."""
    if tier is not None and tier not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}")
    o = _overrides(root)
    if tier is None:
        o.pop(concept, None)
    else:
        o[concept] = {"tier": tier, "by": by or "you", "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (root / OVERRIDES).write_text(json.dumps(o, indent=1, ensure_ascii=False), encoding="utf-8")
    return o


def _stage(group: str) -> str | None:
    g = (group or "").strip().lower()
    return g if g in TIERS else None


def score(m: Any) -> dict[str, dict[str, Any]]:
    """For each concept: its score per tier and the reasons behind each point."""
    pts: dict[str, Counter] = defaultdict(Counter)
    why: dict[str, list[dict[str, Any]]] = defaultdict(list)

    said: set[tuple[str, str, str]] = set()

    def add(c: str, tier: str, w: float, reason: str) -> None:
        if (c, tier, reason) in said:  # a fact stated in many passages counts once
            return
        said.add((c, tier, reason))
        pts[c][tier] += w
        why[c].append({"tier": tier, "points": round(w, 2), "reason": reason})

    concepts = [c for c, n in m.nodes.items() if c != SELF and c not in EVALUATIONS]
    held = [f for f in m.facts if f["held"] and not f["neg"]]
    has_fact = {f["s"] for f in held} | {f["o"] for f in held}
    variables = set(m.variables)
    keep = [c for c in concepts if m.nodes[c].get("support", 0) or c in has_fact or c in variables]

    # 1. the stage of the passages a concept is mentioned in
    group = {pid: _stage(p.get("group", "")) for pid, p in m.passages.items()}
    for c in keep:
        stages = Counter(group.get(x["passage"]) for x in m.nodes[c].get("mentions", []))
        stages.pop(None, None)
        total = sum(stages.values())
        for t, k in stages.items():
            add(c, t, STAGE_WEIGHT * k / total, f"{k} of its {total} mentions are answers in the {LABEL[t]} stage")

    # 2. its kind
    for c in keep:
        kind = m.nodes[c].get("kind", "concept")
        for t, w in KIND_SCORE.get(kind, {}).items():
            add(c, t, w, f"it is a{'n' if kind[0] in 'aeiou' else ''} {kind}")

    # 3. the knowledge graph
    for f in held:
        s, r, o = f["s"], f["r"], f["o"]
        if s == SELF and r in ("practices", "exemplifies"):
            add(o, "act", 1.0, f"{m.label(SELF)} {'practices' if r == 'practices' else 'embodies'} it")
        elif s == SELF and r in ("values", "rejects") and o not in EVALUATIONS:
            add(o, "decide", 1.0, f"{m.label(SELF)} {r} it")
        elif r == "is_a" and o in ("good", "bad") and s != SELF:
            add(s, "decide", 1.0, f"{m.label(SELF)} calls it {o}")
        elif r == "learned_from" and o != SELF:
            add(o, "orient", 1.0, "someone learned from them")

    # 4. the causal model
    for v in m.values:
        if v["concept"] in m.nodes:
            word = "more" if v["direction"] > 0 else "less" if v["direction"] < 0 else "more and less"
            add(v["concept"], "decide", 1.0, f"{m.label(SELF)} urges {word} of it")
    edges = m.causal.get("edges", [])
    into = Counter(e["dst"] for e in edges)
    out = Counter(e["src"] for e in edges)
    for c in out:
        if not into[c] and c in m.nodes:
            add(c, "observe", 0.75, "in the causal model it causes other things and nothing causes it")
    return {c: {"points": pts[c], "reasons": why[c]} for c in pts if c in m.nodes and c != SELF and c not in EVALUATIONS}


def model(m: Any) -> dict[str, Any]:
    """The four tiers, the links between them, and what they show."""
    overrides = _overrides(getattr(m, "root", None))
    scored = score(m)
    tier_of: dict[str, str] = {}
    placed: dict[str, dict[str, Any]] = {}
    for c, s in scored.items():
        ranked = sorted(TIERS, key=lambda t: (-s["points"][t], t != "orient"))  # a tie goes to Orient
        top, second = ranked[0], ranked[1]
        margin = round(s["points"][top] - s["points"][second], 2)
        o = overrides.get(c)
        tier = o["tier"] if o else top
        tier_of[c] = tier
        placed[c] = {"id": c, "label": m.label(c), "kind": m.nodes[c].get("kind"), "tier": tier,
                     "auto": top, "by": o["by"] if o else None, "margin": margin, "weak": not o and margin < 0.5,
                     "points": {t: round(s["points"][t], 2) for t in TIERS if s["points"][t]},
                     "reasons": s["reasons"], "support": m.nodes[c].get("support", 0),
                     "causal": c in m.variables}
    for c, o in overrides.items():  # placed by hand though the evidence gives no tier
        if c not in placed and c in m.nodes:
            tier_of[c] = o["tier"]
            placed[c] = {"id": c, "label": m.label(c), "kind": m.nodes[c].get("kind"), "tier": o["tier"],
                         "auto": None, "by": o["by"], "margin": 0, "weak": False, "points": {}, "reasons": [],
                         "support": m.nodes[c].get("support", 0), "causal": c in m.variables}

    idx = {t: i for i, t in enumerate(TIERS)}

    def kind_of(a: str, b: str) -> str:
        d = idx[tier_of[b]] - idx[tier_of[a]]
        if d == 0:
            return "within"
        if d == 1:
            return "forward"
        if d > 1:
            return "skip"
        return "feedback" if tier_of[a] == "act" else "back"

    links = []
    for e in m.causal.get("edges", []):
        if e["src"] in tier_of and e["dst"] in tier_of:
            links.append({"src": e["src"], "dst": e["dst"], "via": "causal", "sign": e["sign"], "weight": e["weight"],
                          "kind": kind_of(e["src"], e["dst"]), "from": tier_of[e["src"]], "to": tier_of[e["dst"]],
                          "text": f"{m.label(e['src'])} {'raises' if e['sign'] > 0 else 'lowers'} {m.label(e['dst'])}"})
    # a claim the subject makes that would close a cycle is held aside by the (acyclic) causal model,
    # but a loop is what OODA is about, so it is kept here as a loop link
    for c in m.causal.get("constraints", []):
        if c["type"] == "reaches" and c["status"] == "rejected" and c.get("reason") == "cycle" and c["held"] \
                and c["x"] in tier_of and c["y"] in tier_of:
            links.append({"src": c["x"], "dst": c["y"], "via": "loop", "sign": c["sign"], "weight": c["weight"],
                          "kind": kind_of(c["x"], c["y"]), "from": tier_of[c["x"]], "to": tier_of[c["y"]],
                          "text": f"{m.label(c['x'])} {'raises' if c['sign'] > 0 else 'lowers'} {m.label(c['y'])}"
                                  " (closes a cycle, so the causal model holds it aside)"})
    seen = {(l["src"], l["dst"]) for l in links}
    for f in m.facts:
        s, o = f["s"], f["o"]
        if f["held"] and not f["neg"] and s in tier_of and o in tier_of and s != o and (s, o) not in seen \
                and tier_of[s] != tier_of[o]:
            seen.add((s, o))
            links.append({"src": s, "dst": o, "via": "fact", "sign": 0, "weight": f["support"], "rel": f["r"],
                          "kind": kind_of(s, o), "from": tier_of[s], "to": tier_of[o],
                          "text": f"{m.label(s)} {f['r'].replace('_', ' ')} {m.label(o)}"})

    flow = {a: {b: 0 for b in TIERS} for a in TIERS}
    for l in links:
        if l["via"] != "fact":
            flow[l["from"]][l["to"]] += l["weight"]
    tiers = [{"id": t, "label": LABEL[t], "about": ABOUT[t],
              "concepts": sorted((p for p in placed.values() if p["tier"] == t), key=lambda p: (-p["support"], p["label"]))}
             for t in TIERS]
    staged = any(_stage(p.get("group", "")) for p in m.passages.values())
    return {"tiers": tiers, "links": links, "flow": flow, "findings": findings(tiers, links, flow),
            "staged": staged, "placed": len(placed), "overrides": len(overrides)}


def findings(tiers: list[dict[str, Any]], links: list[dict[str, Any]], flow: dict[str, dict[str, int]]) -> list[str]:
    """Plain sentences about the shape of the loop."""
    out = []
    count = {t["id"]: len(t["concepts"]) for t in tiers}
    empty = [LABEL[t] for t in TIERS if not count[t]]
    if empty:
        out.append(f"Nothing the Mirror holds belongs in {' or '.join(empty)} yet.")
    causal = [l for l in links if l["via"] != "fact"]
    if not causal:
        out.append("The causal model links none of these ideas, so the loop has no stated causes yet.")
        return out
    k = Counter(l["kind"] for l in causal)
    into = Counter(l["to"] for l in causal if l["from"] != l["to"])
    for t in ("orient", "decide", "act"):
        if count[t] and not into[t]:
            out.append(f"No cause-and-effect link reaches {LABEL[t]} from another stage.")
    if k["feedback"]:
        n = k["feedback"]
        out.append(f"The loop closes: {n} link{'s run' if n != 1 else ' runs'} from Act back to an earlier stage.")
    elif count["act"]:
        out.append("The loop does not close: no link runs from Act back to what is observed or how it is read.")
    if flow["orient"]["act"]:
        out.append(f"Orient leads straight to Act in {flow['orient']['act']} passage{'s' if flow['orient']['act'] != 1 else ''}, without a stated decision (Boyd's implicit guidance and control).")
    if flow["observe"]["act"]:
        out.append(f"Observe leads straight to Act in {flow['observe']['act']} passage{'s' if flow['observe']['act'] != 1 else ''}: a reflex, with no orientation in between.")
    between = sorted(((flow[a][b], a, b) for a in TIERS for b in TIERS if a != b), reverse=True)
    (top, a_, b_), (second, _, _) = between[0], between[1]
    heaviest = (a_, b_)
    if top and top > second:  # only when one pair stands out
        out.append(f"Most of the stated causation runs from {LABEL[heaviest[0]]} to {LABEL[heaviest[1]]}.")
    return out


def graph(m: Any) -> dict[str, Any]:
    """The canvas: concepts with their tier, links with their kind."""
    md = model(m)
    nodes = [{"id": p["id"], "label": p["label"], "tier": t["id"], "support": p["support"], "weak": p["weak"],
              "by": p["by"]} for t in md["tiers"] for p in t["concepts"]]
    edges = [{"id": f"{l['src']}|{l['dst']}", "source": l["src"], "target": l["dst"], "sign": l["sign"],
              "weight": l["weight"], "kind": l["kind"], "via": l["via"]} for l in md["links"]]
    return {"layer": "ooda", "nodes": nodes, "edges": edges}
