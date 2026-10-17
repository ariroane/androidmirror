"""Derange: where the subject's worldview rests on the least, relative to what matters most in it.

Every arrow and every fact in a Mirror goes back to passages. Some of what the
subject holds is said in forty passages and some in one; some of it carries
nothing else and some of it carries a whole account. Derange looks for the
second kind of weakness: little support under a lot of weight.

1. The most important subgraphs.
   - Causal: the subject's ends, the concepts the values layer says they urge
     more or less of (importance: the passages that urge it). The subgraph of an
     end is everything that acts on it in the subject's model, with how (the
     signs of the paths). With no values, the concepts the most passages
     explain stand in.
   - Conceptual: the concepts the corpus is most about (importance: the
     passages that mention them; the subject `self` is a hub by construction
     and is left out). The subgraph of a concept is what the subject holds
     about it, stated or inferred: its classes, its evaluation, what it
     requires, opposes, and so on.
2. The weakest basis of each.
   - Causal: for each thing that acts on an end, the minimum cut between them
     with each arrow's capacity the passages that make it: the fewest passages
     the subject's account of "this bears on that" rests on at its thinnest.
     An arrow the user settled by answering a question (pinned) is never cut.
   - Conceptual: the stated facts each held fact about the concept rests on
     (the premises of its proof): a premise said in one passage that a dozen
     inferences pass through.
3. Knock it out and look. A candidate basis is removed (the judgments or the
   stated fact behind it, whole) and the model is fitted again, or the
   knowledge base re-derived, exactly as a build would. What changes is what
   the basis was holding up: concepts that no longer bear on an end, effects
   whose sign changes, claims that stop following, inferred facts that stop
   following. If the subject's other words carry it anyway, nothing falls and
   it is not reported.

A finding's score is how much of the important subgraphs falls (each
subgraph's share that falls, times its importance) per passage of support.

Derange only reads. It never changes the graphs, the judgments or the facts:
each finding is a place to look, with the quotes behind it and the question it
raises (`probe`), for the person to answer or for a later reading to settle.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from .causal import fit, graph_of
from .kg import KnowledgeBase, fact_key
from .schema import EVALUATIONS, SELF

DEFAULTS = {"ends": 8, "concepts": 12, "facts_per_concept": 80, "knockouts": 30, "top": 20}
HOLDING = {"arrow", "follows", "holds"}


def run(mirror: Any, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Both layers' weakest bases, strongest findings first. `mirror` is a loaded store.Mirror."""
    cfg = {**DEFAULTS, **(settings or {})}
    causal = causal_bases(mirror, cfg)
    conceptual = conceptual_bases(mirror, cfg)
    return {"settings": cfg, "causal": causal, "conceptual": conceptual,
            "stats": {"ends": len(causal["subgraphs"]), "concepts": len(conceptual["subgraphs"]),
                      "causal_findings": len(causal["findings"]), "conceptual_findings": len(conceptual["findings"]),
                      "knocked_out": causal["tried"] + conceptual["tried"]}}


# ---------- causal ----------


def judgment_constraints(j: dict[str, Any]) -> set[str]:
    """The ids of the constraints a judgment counts toward (the same mapping as judgments.constraints)."""
    k = j["kind"]
    if k == "ruling":  # the user's ruling on a constraint (Question engine): bears on that constraint only
        return {j["target"]}
    if k == "screens_off":
        return {f"reaches|{j['y']}|{j['z']}", f"only_through|{j['x']}|{j['y']}|{j['z']}"}
    if k == "independent" or (k == "intervention" and j["sign"] == 0):
        return {f"no_path|{j['x']}|{j['y']}"}
    return {f"reaches|{j['x']}|{j['y']}"}


def account(g: Any, end: str) -> dict[str, frozenset[int]]:
    """Everything that acts on `end` in a fitted graph, with the signs of its paths there."""
    rev: dict[str, set[str]] = defaultdict(set)
    for a, outs in g.succ.items():
        for b in outs:
            rev[b].add(a)
    seen, todo = set(), [end]
    while todo:
        for a in rev[todo.pop()]:
            if a not in seen and a != end:
                seen.add(a)
                todo.append(a)
    return {a: frozenset(s for _, s in g.signed_paths(a, end)) for a in sorted(seen)}


def ends_of(mirror: Any, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The subject's ends in the causal model: what they urge more or less of, else the most explained effects."""
    nodes = set(mirror.causal.get("nodes", []))
    rows = [{"id": v["concept"], "importance": v["more"] + v["less"], "direction": v["direction"], "why": "urged"}
            for v in mirror.values if v["concept"] in nodes]
    if not rows:
        into: dict[str, int] = defaultdict(int)
        for e in mirror.causal.get("edges", []):
            into[e["dst"]] += e["weight"]
        rows = [{"id": n, "importance": w, "direction": 0, "why": "explained"} for n, w in into.items()]
    rows.sort(key=lambda r: (-r["importance"], r["id"]))
    return rows[:cfg["ends"]]


def causal_bases(mirror: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    model = mirror.causal
    edges = model.get("edges", [])
    ends = ends_of(mirror, cfg)
    g0 = graph_of(model)
    before = {e["id"]: account(g0, e["id"]) for e in ends}
    ends = [e for e in ends if before[e["id"]]]
    if not edges or not ends:
        return {"subgraphs": [], "findings": [], "tried": 0}

    # an arrow the user settled by answering a question is not a weakness of the corpus: it is never cut
    rows = {c["id"]: c for c in model.get("constraints", [])}
    net = nx.DiGraph()
    for e in edges:
        if rows.get(e["constraint"], {}).get("pinned"):
            net.add_edge(e["src"], e["dst"])  # no capacity: unbounded
        else:
            net.add_edge(e["src"], e["dst"], capacity=e["weight"])
    by_pair = {(e["src"], e["dst"]): e for e in edges}

    # candidate bases: the minimum cut under each thing that acts on each end
    candidates: dict[frozenset, float] = {}
    for end in ends:
        v = end["id"]
        for a in before[v]:
            try:
                _, (side, _) = nx.minimum_cut(net, a, v)
            except nx.NetworkXUnbounded:
                continue  # every way from a to v runs through what the user settled
            cut = frozenset((s, t) for s in side for t in net.successors(s) if t not in side)
            if cut and cut not in candidates:
                candidates[cut] = _surgery_score(net, cut, ends, before)
    ranked = sorted(candidates, key=lambda c: (-candidates[c], sorted(c)))[:cfg["knockouts"]]

    settings = model.get("settings") or {}
    statuses = {c["id"]: c["status"] for c in model.get("constraints", [])}
    findings = []
    for cut in ranked:
        removed = {by_pair[p]["constraint"] for p in cut}
        gone = [j for j in mirror.judgments if j["held"] and judgment_constraints(j) & removed]
        passages = sorted({j["passage"] for j in gone})
        if not passages:
            continue
        rest = [j for j in mirror.judgments if not (j["held"] and judgment_constraints(j) & removed)]
        after = fit(rest, settings)
        g1 = after["_graph"]
        hits, damage = [], 0.0
        for end in ends:
            v, acc = end["id"], before[end["id"]]
            now = account(g1, v) if v in g1.succ else {}
            lost = [a for a in acc if a not in now]
            changed = [a for a in acc if a in now and now[a] != acc[a]]
            if lost or changed:
                share = (len(lost) + len(changed)) / len(acc)
                damage += end["importance"] * share
                hits.append({"end": v, "label": mirror.label(v), "importance": end["importance"], "of": len(acc),
                             "lost": lost, "lost_labels": [mirror.label(a) for a in lost],
                             "changed": [{"id": a, "label": mirror.label(a), "before": sorted(acc[a]),
                                          "after": sorted(now[a])} for a in changed], "share": round(share, 3)})
        if not damage:
            continue  # the subject's other words carry it
        new_status = {c["id"]: c["status"] for c in after["constraints"]}
        falls = [mirror.say(rows[cid]) for cid, st in statuses.items()
                 if st in HOLDING and cid not in removed and rows[cid]["held"] and new_status.get(cid) not in HOLDING]
        old_arrows = set(by_pair)
        drawn = [f"{mirror.label(e['src'])} → {mirror.label(e['dst'])}" for e in after["edges"]
                 if (e["src"], e["dst"]) not in old_arrows]
        against = [mirror.say(c) for c in model.get("constraints", [])
                   if c["status"] == "rejected" and set(c.get("conflicts") or []) & removed]
        basis = []
        for p in sorted(cut):
            e = by_pair[p]
            c = rows.get(e["constraint"], {})
            basis.append({"src": e["src"], "dst": e["dst"], "sign": e["sign"], "weight": e["weight"],
                          "constraint": e["constraint"], "text": mirror.say(c) if c else e["constraint"],
                          "open": bool(e.get("open")), "sign_conflict": bool(c.get("sign_conflict")),
                          "evidence": [x for x in c.get("evidence", []) if x["held"]][:4]})
        strength = len(passages)
        findings.append({
            "layer": "causal", "id": "cut:" + ";".join(f"{a}>{b}" for a, b in sorted(cut)), "basis": basis,
            "strength": strength, "passages": passages, "ends": hits, "claims_fall": falls, "now_drawn": drawn,
            "against": against, "damage": round(damage, 3), "score": round(damage / strength, 3),
            "probe": _causal_probe(mirror, basis, hits, strength),
        })
    findings.sort(key=lambda f: (-f["score"], f["strength"], f["id"]))
    subgraphs = []
    for end in ends:
        hit = [f for f in findings if any(h["end"] == end["id"] for h in f["ends"])]
        weakest = min(hit, key=lambda f: (f["strength"], -f["score"])) if hit else None
        subgraphs.append({"id": end["id"], "label": mirror.label(end["id"]), "importance": end["importance"],
                          "direction": end["direction"], "why": end["why"], "size": len(before[end["id"]]),
                          "weakest": {"finding": weakest["id"], "strength": weakest["strength"]} if weakest else None})
    return {"subgraphs": subgraphs, "findings": findings[:cfg["top"]], "tried": len(ranked)}


def _surgery_score(net: nx.DiGraph, cut: frozenset, ends: list[dict[str, Any]],
                   before: dict[str, dict[str, frozenset[int]]]) -> float:
    """A quick estimate before refitting: the important share cut off, per passage of the cut."""
    h = net.copy()
    h.remove_edges_from(cut)
    capacity = sum(net.edges[p].get("capacity", 0) for p in cut)
    damage = 0.0
    for end in ends:
        v, acc = end["id"], before[end["id"]]
        still = nx.ancestors(h, v) if v in h else set()
        damage += end["importance"] * sum(1 for a in acc if a not in still) / len(acc)
    return damage / max(capacity, 1)


def _causal_probe(mirror: Any, basis: list[dict[str, Any]], hits: list[dict[str, Any]], strength: int) -> str:
    who = (mirror.meta.get("corpus") or {}).get("subject") or "the subject"
    claims = " and ".join(f"{mirror.label(b['src'])} {'raises' if b['sign'] > 0 else 'lowers'} {mirror.label(b['dst'])}"
                          for b in basis)
    top = max(hits, key=lambda h: h["importance"] * h["share"])
    n = len(top["lost"]) + len(top["changed"])
    return (f"Is it so that {claims}? {who[:1].upper() + who[1:]} says so in {_n(strength, 'passage')}, yet without it {n} of the "
            f"{_n(top['of'], 'thing')} that act on {top['label']} would no longer act on it the same way.")


def _n(k: int, word: str) -> str:
    return f"{k} {word}{'' if k == 1 else 's'}"


# ---------- conceptual ----------


def conceptual_bases(mirror: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    kb = mirror.kb
    by_subject: dict[str, list[tuple]] = defaultdict(list)
    for t in kb.closure:
        by_subject[t[0]].append(t)
    concepts = sorted((n for n in by_subject if n != SELF and n not in EVALUATIONS),
                      key=lambda n: (-mirror.nodes.get(n, {}).get("support", 0), n))[:cfg["concepts"]]
    if not concepts:
        return {"subgraphs": [], "findings": [], "tried": 0}
    imp = {c: max(1, mirror.nodes.get(c, {}).get("support", 0)) for c in concepts}
    sub = {c: sorted(by_subject[c])[:cfg["facts_per_concept"]] for c in concepts}
    stated = {fact_key(f["s"], f["r"], f["o"]): f for f in mirror.facts if not f["neg"]}
    memo: dict[tuple, frozenset] = {}

    # which stated facts each important fact rests on, by its proof
    load: dict[str, float] = defaultdict(float)
    for c in concepts:
        for t in sub[c]:
            for k in _premises(kb, t, memo):
                load[k] += imp[c] / len(sub[c])
    support = {k: max(1, stated[k]["support"]) for k in load if k in stated}
    ranked = sorted(support, key=lambda k: (-load[k] / support[k], k))[:cfg["knockouts"]]

    findings = []
    for k in ranked:
        f = stated[k]
        after = KnowledgeBase(mirror.nodes, [x for x in mirror.facts if x is not f])
        hits, damage = [], 0.0
        for c in concepts:
            lost = [t for t in sub[c] if t not in after.closure]
            if lost:
                share = len(lost) / len(sub[c])
                damage += imp[c] * share
                hits.append({"concept": c, "label": mirror.label(c), "importance": imp[c], "of": len(sub[c]),
                             "share": round(share, 3), "lost": [_fact_text(mirror, t) for t in lost[:12]],
                             "n_lost": len(lost)})
        if not damage:
            continue
        denial = next((x for x in mirror.facts if x["neg"] and x["held"]
                       and (x["s"], x["r"], x["o"]) == (f["s"], f["r"], f["o"])), None)
        against = [v["type"] for v in kb.violations if k in (v.get("facts") or []) or v.get("fact") == k]
        strength = f["support"]
        findings.append({
            "layer": "conceptual", "id": "fact:" + k, "fact": k, "text": _fact_text(mirror, (f["s"], f["r"], f["o"])),
            "s": f["s"], "r": f["r"], "o": f["o"], "strength": strength,
            "passages": sorted({e["passage"] for e in f["evidence"]}),
            "evidence": [e for e in f["evidence"] if e.get("attribution") == SELF][:4] or f["evidence"][:4],
            "concepts": hits, "denied": bool(denial), "against": sorted(set(against)),
            "damage": round(damage, 3), "score": round(damage / max(strength, 1), 3),
            "probe": _fact_probe(mirror, f, hits, strength),
        })
    findings.sort(key=lambda x: (-x["score"], x["strength"], x["id"]))
    subgraphs = []
    for c in concepts:
        hit = [x for x in findings if any(h["concept"] == c for h in x["concepts"])]
        weakest = min(hit, key=lambda x: (x["strength"], -x["score"])) if hit else None
        subgraphs.append({"id": c, "label": mirror.label(c), "importance": imp[c], "size": len(sub[c]),
                          "weakest": {"finding": weakest["id"], "strength": weakest["strength"]} if weakest else None})
    return {"subgraphs": subgraphs, "findings": findings[:cfg["top"]], "tried": len(ranked)}


def _premises(kb: KnowledgeBase, t: tuple, memo: dict[tuple, frozenset]) -> frozenset:
    """The stated facts a fact's recorded proof bottoms out in."""
    if t in memo:
        return memo[t]
    memo[t] = frozenset()  # a guard; proofs are acyclic, but cheaply so
    proof = kb.closure.get(t)
    if proof is None:
        out = frozenset()
    elif proof["rule"] == "stated":
        out = frozenset([fact_key(*t)])
    else:
        out = frozenset().union(*(_premises(kb, p, memo) for p in proof["from"]))
    memo[t] = out
    return out


def _fact_text(mirror: Any, t: tuple) -> str:
    return f"{mirror.label(t[0])} {t[1].replace('_', ' ')} {mirror.label(t[2])}"


def _fact_probe(mirror: Any, f: dict[str, Any], hits: list[dict[str, Any]], strength: int) -> str:
    who = (mirror.meta.get("corpus") or {}).get("subject") or "the subject"
    top = max(hits, key=lambda h: h["importance"] * h["share"])
    return (f"Is it so that {_fact_text(mirror, (f['s'], f['r'], f['o']))}? {who[:1].upper() + who[1:]} says so in {_n(strength, 'passage')}, "
            f"yet {top['n_lost']} of the {_n(top['of'], 'thing')} held about {top['label']} rest on it.")


# ---------- saved reports ----------


def report(root: Path, mirror: Any, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run Derange on the Mirror at `root` and save it beside it as derange.json, tagged with the Mirror it read."""
    out = {**run(mirror, settings), "mirror_mtime": (root / "mirror.json").stat().st_mtime}
    (root / "derange.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def saved(root: Path) -> dict[str, Any] | None:
    """The saved report, if it was made from the Mirror as it is now."""
    path = root / "derange.json"
    if not path.exists() or not (root / "mirror.json").exists():
        return None
    out = json.loads(path.read_text(encoding="utf-8"))
    return out if out.get("mirror_mtime") == (root / "mirror.json").stat().st_mtime else None


def print_report(out: dict[str, Any], top: int = 10) -> None:
    for layer, title in (("causal", "Causal: what the subject's ends rest on"),
                         ("conceptual", "Conceptual: what the main concepts rest on")):
        part = out[layer]
        print(f"\n{title}")
        if not part["findings"]:
            print("  nothing found: no single basis holds up much of it, or there is too little to look at")
        for i, f in enumerate(part["findings"][:top], 1):
            locs = ", ".join(e["locator"] for e in (f.get("evidence") or [x for b in f.get("basis", []) for x in b["evidence"]])[:4])
            print(f"{i:>3}. {f['probe']}\n     score {f['score']}; passages: {locs}")
