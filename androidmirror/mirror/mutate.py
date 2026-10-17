"""Mutate: combine concepts or split one into senses, then re-evaluate their edges in both graphs.

A mutation changes what the Mirror counts as one concept:

- merge: two or more concepts are one ("wrath" and "anger"), or one is the
  other's opposite pole ("tranquillity" is "disturbance" at -1). Every fact,
  causal judgment, value and mention of a member moves to the concept it is
  merged into, with its sign turned around for an opposite.
- split: one concept is used in two or more senses (the "pleasure" the subject
  calls good and the "pleasure" they call bad). Every passage that uses it is
  assigned to one sense, and everything read from that passage goes with it.

Re-evaluation does not patch edges one at a time. The Mirror is re-assembled
from the per-passage evidence under the new identities, then the knowledge base
is re-run to its fixpoint and the causal model is re-fitted to the subject's
judgments, held-out check included. So a merged edge carries the evidence of
both originals and is re-judged with their combined weight, a judgment about two
members of a merge ("anger breeds wrath") collapses into a statement about one
concept and is reported, and arrows elsewhere in the model can change too.

Nothing here changes a Mirror by itself. `Engine.propose*` only writes
proposals with a preview of what would change; `Engine.approve(id, approver)`
is the single path that writes mirror.json, and only for a proposal whose base
is still the Mirror on disk (a stale one is recomputed and shown again).
Approved mutations are replayed when the Mirror is rebuilt, and each one can be
undone.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from .. import llm
from . import answers
from . import codebook as cb
from . import judgments as jd
from .causal import fit, holdout, public
from .kg import KnowledgeBase, fact_key
from .schema import EVALUATIONS, HELD_MODALITIES, SELF

RESERVED = {SELF, *EVALUATIONS}
SHOW = 30  # list items kept in a preview
CLAUDE_BATCH = 50  # passages per sense-reading call
CLAUDE_CHARS = 700


class MutationError(ValueError):
    pass


# ---------- the mutation itself ----------


def _pole(n: str) -> str:
    return f"{n}~"


def check(data: dict[str, Any], m: dict[str, Any]) -> None:
    nodes = data["nodes"]

    def usable(n: str) -> None:
        if n not in nodes:
            raise MutationError(f"no concept {n!r} in this Mirror")
        if n in RESERVED or nodes[n].get("pole_of"):
            raise MutationError(f"{n!r} cannot be mutated (the subject, an evaluation, or an opposite pole)")

    if m["type"] == "merge":
        usable(m["into"])
        if not m["members"]:
            raise MutationError("a merge needs at least two concepts")
        for x in m["members"]:
            usable(x["id"])
            if x["id"] == m["into"] or x["polarity"] not in (1, -1):
                raise MutationError(f"bad member {x}")
        if len({x["id"] for x in m["members"]}) != len(m["members"]):
            raise MutationError("a concept is listed twice")
    elif m["type"] == "split":
        usable(m["node"])
        if len(m["senses"]) < 2:
            raise MutationError("a split needs at least two senses")
        ids = [s["id"] for s in m["senses"]]
        if len(set(ids)) != len(ids) or any(i in nodes and i != m["node"] for i in ids) or m["default"] not in ids:
            raise MutationError("sense ids must be new, distinct, and include the default")
    else:
        raise MutationError(f"unknown mutation {m['type']!r}")


def _mapper(m: dict[str, Any]) -> Callable[[str, str], tuple[str, int]]:
    """(concept, passage) -> (the concept it becomes, polarity) under the mutation."""
    if m["type"] == "merge":
        table = {x["id"]: (m["into"], x["polarity"]) for x in m["members"]}
        return lambda n, pid: table.get(n, (n, 1))
    sense = {pid: s["id"] for s in m["senses"] for pid in s["passages"]}
    node, default = m["node"], m["default"]
    return lambda n, pid: (sense.get(pid, default), 1) if n == node else (n, 1)


def removed(m: dict[str, Any]) -> set[str]:
    return {x["id"] for x in m["members"]} if m["type"] == "merge" else {m["node"]}


def created(m: dict[str, Any]) -> set[str]:
    return {m["into"]} if m["type"] == "merge" else {s["id"] for s in m["senses"]}


def _cids(j: dict[str, Any]) -> list[str]:
    """The constraint ids a judgment bears on (judgments.constraints)."""
    if j["kind"] == "screens_off":
        return [f"reaches|{j['y']}|{j['z']}", f"only_through|{j['x']}|{j['y']}|{j['z']}"]
    if j["kind"] == "independent" or (j["kind"] == "intervention" and j["sign"] == 0):
        return [f"no_path|{j['x']}|{j['y']}"]
    return [f"reaches|{j['x']}|{j['y']}"]


def _is_codebook_pole_fact(f: dict[str, Any], nodes: dict[str, Any]) -> bool:
    return f["r"] == "opposes" and bool(nodes.get(f["s"], {}).get("pole_of")) and \
        all(e["passage"] == "codebook" for e in f["evidence"])


def transform(data: dict[str, Any], m: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The Mirror with the mutation made (causal model not yet re-fitted), and where everything went."""
    check(data, m)
    old = data["nodes"]
    where = _mapper(m)
    gone = removed(m)
    order = {pid: i for i, pid in enumerate(data.get("passage_ids", []))}
    need_pole: dict[str, str] = {}  # new pole -> label to give it if it is new

    def end(n: str, pid: str) -> str:
        base = old.get(n, {}).get("pole_of")
        t, pol = where(base, pid) if base else where(n, pid)
        if base:
            pol = -pol
        if pol == 1:
            return t
        need_pole.setdefault(_pole(t), old.get(n, {}).get("label", n))
        return _pole(t)

    lineage: dict[str, Any] = {"facts": defaultdict(set), "collapsed_facts": [], "judgments": defaultdict(set),
                               "collapsed_judgments": [], "lossy_values": False}

    # facts: regrouped from their evidence, passage by passage
    groups: dict[str, dict[str, Any]] = {}
    fed: dict[str, set[str]] = defaultdict(set)
    for f in data["facts"]:
        if _is_codebook_pole_fact(f, old):
            continue
        okey = fact_key(f["s"], f["r"], f["o"], f.get("neg", False))
        for e in f["evidence"]:
            s, o = end(f["s"], e["passage"]), end(f["o"], e["passage"])
            key = fact_key(s, f["r"], o, f.get("neg", False))
            g = groups.setdefault(key, {"s": s, "r": f["r"], "o": o, "neg": f.get("neg", False), "evidence": [],
                                        "held": False, "qualifiers": []})
            g["evidence"].append(e)
            g["held"] = g["held"] or (e.get("attribution") == SELF and e.get("modality") in HELD_MODALITIES)
            lineage["facts"][key].add(okey)
            fed[okey].add(key)
    by_old = {fact_key(f["s"], f["r"], f["o"], f.get("neg", False)): f for f in data["facts"]}
    for okey, keys in fed.items():
        f = by_old[okey]
        for key in keys:
            g = groups[key]
            g["qualifiers"] += [q for q in f.get("qualifiers", []) if q not in g["qualifiers"]]
            if f.get("wordings"):
                g.setdefault("wordings", [])
                g["wordings"] += [w for w in f["wordings"] if w not in g["wordings"]]
    facts = []
    for key, g in groups.items():
        g["support"] = len({e["passage"] for e in g["evidence"]})
        g["modalities"] = sorted({e["modality"] for e in g["evidence"] if e.get("modality")})
        if g["s"] == g["o"] and g["r"] != "opposes":  # "anger is_a wrath" says nothing once they are one
            lineage["collapsed_facts"].append({**{k: g[k] for k in ("s", "r", "o", "neg", "held", "support")},
                                               "from": sorted(lineage["facts"].pop(key))})
            continue
        facts.append(g)

    # causal judgments
    judgments = []
    for j in data["judgments"]:
        pid = j["passage"]
        x, px = where(j["x"], pid)
        y, py = where(j["y"], pid)
        if j["kind"] == "screens_off":
            z, pz = where(j["z"], pid)
            nj = {**j, "x": x, "y": y, "z": z, "sign": j["sign"] * py * pz}
            same = len({x, y, z}) < 3
        else:
            nj = {**j, "x": x, "y": y, "sign": j["sign"] * px * py}
            same = x == y
        if same:
            lineage["collapsed_judgments"].append({**nj, "was": {k: j[k] for k in ("x", "y", "z", "sign")}})
            continue
        for new, was in zip(_cids(nj), _cids(j)):
            lineage["judgments"][new].add(was)
        judgments.append(nj)

    # values
    records = data.get("value_records")
    if records is None:  # a Mirror saved before raw values were kept: rebuild them from the shown evidence
        lineage["lossy_values"] = True
        records = [e for v in data.get("values", []) for e in v["evidence"]]
    vrec = []
    for v in records:
        t, pol = where(v["concept"], v["passage"])
        vrec.append({**v, "concept": t, "direction": v["direction"] * pol})

    # nodes
    nodes = {k: copy.deepcopy(v) for k, v in old.items() if k not in gone and v.get("pole_of") not in gone}
    if m["type"] == "merge":
        into = nodes[m["into"]]
        if m.get("label"):
            if into["label"] != m["label"] and into["label"] not in into["aliases"]:
                into["aliases"].append(into["label"])
            into["label"] = m["label"]
        seen = {e["passage"] for e in into.get("mentions", [])}
        into["merged_from"] = into.get("merged_from", [])
        for x in m["members"]:
            b = old[x["id"]]
            tag = "" if x["polarity"] == 1 else " (opposite)"
            for a in [b["label"], *b.get("aliases", [])]:
                a = a + tag if tag and not a.endswith(tag) else a
                if a != into["label"] and a not in into["aliases"]:
                    into["aliases"].append(a)
            for e in b.get("mentions", []):
                if e["passage"] not in seen:
                    seen.add(e["passage"])
                    into["mentions"].append({**e, "level": e["level"] * x["polarity"]})
            into["variable"] = into.get("variable", False) or b.get("variable", False)
            into["merged_from"].append({"id": x["id"], "polarity": x["polarity"],
                                        **{k: b.get(k) for k in ("label", "kind", "gloss", "aliases", "support")}})
    else:
        a = old[m["node"]]
        for s in m["senses"]:
            mentions = [e for e in a.get("mentions", []) if where(m["node"], e["passage"])[0] == s["id"]]
            words = Counter(e["label"] for e in mentions if e.get("label"))
            for j in data["judgments"]:
                for k in ("x", "y", "z"):
                    if j.get(k) == m["node"] and where(m["node"], j["passage"])[0] == s["id"] and j["labels"].get(k):
                        words[j["labels"][k].strip().lower()] += 1
            aliases = [w for w, _ in words.most_common() if w != s["label"]]
            if a["label"] != s["label"] and a["label"] not in aliases:
                aliases.insert(0, a["label"])
            nodes[s["id"]] = {"label": s["label"], "kind": a["kind"], "gloss": s.get("gloss") or a.get("gloss", ""),
                              "variable": a.get("variable", False), "aliases": aliases, "mentions": mentions,
                              "split_from": {k: a.get(k) for k in ("label", "gloss", "aliases", "support")} | {"id": m["node"]}}
    for nid in created(m):
        nodes[nid]["mentions"].sort(key=lambda e: order.get(e["passage"], 0))
        nodes[nid]["support"] = len({e["passage"] for e in nodes[nid]["mentions"]})
    for pole, label in need_pole.items():
        if pole not in nodes:
            base = pole[:-1]
            nodes[pole] = {"label": label, "kind": nodes[base]["kind"], "variable": False,
                           "gloss": f"the opposite pole of {nodes[base]['label']}", "aliases": [], "pole_of": base,
                           "mentions": [], "support": 0}
    for pole, n in nodes.items():  # an opposite pole is the contrary of its concept, by construction
        if n.get("pole_of"):
            facts.append({"s": pole, "r": "opposes", "o": n["pole_of"], "neg": False, "held": True, "qualifiers": [],
                          "evidence": [{"passage": "codebook", "locator": "codebook", "quote": "", "modality": "asserts",
                                        "attribution": SELF}], "support": 1, "modalities": ["asserts"]})

    out = {**data, "nodes": nodes, "facts": facts, "judgments": judgments, "values": jd.values(vrec),
           "value_records": vrec, "meta": {**data["meta"], "mutations": [*data["meta"].get("mutations", []), _mid(m)]}}
    lineage["facts"] = {k: sorted(v) for k, v in lineage["facts"].items()}
    lineage["judgments"] = {k: sorted(v) for k, v in lineage["judgments"].items()}
    return out, lineage


def refit(data: dict[str, Any], check: bool = True, book: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-fit the causal model and (check=True) its held-out check, with the Mirror's own settings (no Claude calls).

    As in build.finalize, the fit includes the user's approved answers (`book`); the held-out check is on the corpus."""
    cfg = (data.get("causal") or {}).get("settings", {})
    model = fit(answers.overlay(data, book or {})["judgments"], cfg)
    data["causal"] = {**public(model), "holdout": holdout(data["judgments"], cfg) if check else None}
    return data


# ---------- re-evaluation: what the mutation changes ----------


def _agreement(h: dict[str, Any] | None) -> dict[str, float | int] | None:
    """Of the held-out claims, the share the rest of the corpus restates or implies, and the share it contradicts.

    A merge of two words for one thing turns "new" claims into restated ones; a merge of two different things,
    or a bad split, shows up as contradictions."""
    if h is None:
        return None
    p = h.get("positive") or {}
    n = h.get("negative") or {}
    pos, neg = sum(p.values()), sum(n.values())
    bad = p.get("contradicted", 0) + p.get("sign differs", 0) + n.get("contradicted", 0)
    return {"supported": round((p.get("restated", 0) + p.get("follows", 0)) / pos, 3) if pos else 0.0,
            "contradicted": round(bad / (pos + neg), 3) if pos + neg else 0.0, "claims": pos + neg}


def evaluate(data: dict[str, Any], m: dict[str, Any], check: bool = True,
             book: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The mutated, re-fitted Mirror and a preview of everything it changes in both graphs.

    check=False skips the held-out check (five more fits), which is what takes time on a large Mirror.
    `book` is the Question engine's answers (answers.json), fitted with the corpus as build does."""
    after, lineage = transform(data, m)
    refit(after, check, book)
    prev = compare(data, after, m, lineage)
    gone = removed(m)
    stale = [p for p in answers.approved(book or {}) if any(
        {ch.get("x"), ch.get("y"), ch.get("z"), ch.get("s"), ch.get("o"), ch.get("concept"),
         *answers.parts(ch.get("target", "r||"))} & gone for ch in p["changes"])]
    if stale:
        prev["warnings"].append(f"{len(stale)} approved answer(s) name a concept this removes and will no longer "
                                f"attach to it, e.g. “{stale[0]['question']}”; revoke and answer again after")
    return after, prev


def compare(before: dict[str, Any], after: dict[str, Any], m: dict[str, Any], lineage: dict[str, Any]) -> dict[str, Any]:
    nb, na = before["nodes"], after["nodes"]
    lab_b = lambda n: nb.get(n, {}).get("label", n)  # noqa: E731
    lab_a = lambda n: na.get(n, {}).get("label", n)  # noqa: E731
    tb = removed(m) | ({m["into"]} if m["type"] == "merge" else set())
    tb |= {p for p, n in nb.items() if n.get("pole_of") in tb}
    ta = created(m) | {p for p, n in na.items() if n.get("pole_of") in created(m)}
    fb = {fact_key(f["s"], f["r"], f["o"], f.get("neg", False)): f for f in before["facts"]}
    fa = {fact_key(f["s"], f["r"], f["o"], f.get("neg", False)): f for f in after["facts"]}

    def fview(f: dict[str, Any], lab: Callable[[str], str]) -> dict[str, Any]:
        return {"s": f["s"], "r": f["r"], "o": f["o"], "neg": f.get("neg", False), "held": f["held"],
                "support": f["support"], "text": f"{lab(f['s'])} {'not ' if f.get('neg') else ''}{f['r'].replace('_', ' ')} {lab(f['o'])}"}

    combined, carried, divided = [], 0, []
    for key, src in lineage["facts"].items():
        f = fa.get(key)
        if f is None or not ({f["s"], f["o"]} & ta) or src == ["codebook"]:
            continue
        if len(src) > 1:
            combined.append({**fview(f, lab_a), "from": [fview(fb[k], lab_b) for k in src if k in fb]})
        else:
            carried += 1
    if m["type"] == "split":
        into: dict[str, list[str]] = defaultdict(list)
        for key, src in lineage["facts"].items():
            for k in src:
                into[k].append(key)
        for k, keys in into.items():
            if len(keys) > 1 and k in fb:
                divided.append({**fview(fb[k], lab_b), "into": [fview(fa[x], lab_a) for x in keys if x in fa]})
    collapsed = [{**c, "text": f"{lab_a(c['s'])} {c['r'].replace('_', ' ')} itself",
                  "from": [fview(fb[k], lab_b) for k in c["from"] if k in fb]} for c in lineage["collapsed_facts"]]

    kb_b = KnowledgeBase(nb, before["facts"])
    kb_a = KnowledgeBase(na, after["facts"])
    about = lambda vs, ids: [v for v in vs if any(f'"{i}"' in json.dumps(v) for i in ids)]  # noqa: E731
    inferred = lambda kb, ids: sum(1 for (s, r, o), p in kb.closure.items()  # noqa: E731
                                   if p["rule"] != "stated" and (s in ids or o in ids))

    cb_, ca = before.get("causal") or {}, after["causal"]
    eb = {(e["src"], e["dst"]): e for e in cb_.get("edges", [])}
    ea = {(e["src"], e["dst"]): e for e in ca.get("edges", [])}
    sb = {c["id"]: c for c in cb_.get("constraints", [])}
    sa = {c["id"]: c for c in ca.get("constraints", [])}

    def eview(e: dict[str, Any], lab: Callable[[str], str]) -> dict[str, Any]:
        return {"src": e["src"], "dst": e["dst"], "sign": e["sign"], "weight": e["weight"], "open": bool(e.get("open")),
                "text": f"{lab(e['src'])} {'raises' if e['sign'] > 0 else 'lowers'} {lab(e['dst'])}"}

    def say(c: dict[str, Any], lab: Callable[[str], str]) -> str:
        if c["type"] == "reaches":
            return f"{lab(c['x'])} {'raises' if c['sign'] > 0 else 'lowers'} {lab(c['y'])}"
        if c["type"] == "no_path":
            return f"{lab(c['x'])} makes no difference to {lab(c['y'])}"
        return f"{lab(c['x'])} acts on {lab(c['z'])} only through {lab(c['y'])}"

    def cview(c: dict[str, Any], lab: Callable[[str], str]) -> dict[str, Any]:
        return {"id": c["id"], "type": c["type"], "weight": c["weight"], "status": c["status"],
                "reason": c.get("reason"), "text": say(c, lab)}

    touches = lambda c, ids: bool({c["x"], c["y"], c["z"]} & ids)  # noqa: E731
    constraints = []
    for c in sorted((c for c in sa.values() if touches(c, ta) and c["held"]), key=lambda c: -c["weight"]):
        src = [sb[k] for k in lineage["judgments"].get(c["id"], []) if k in sb]
        constraints.append({**cview(c, lab_a), "from": [cview(x, lab_b) for x in src]})
    mapped = set()
    if m["type"] == "merge":
        table = {x["id"]: m["into"] for x in m["members"]}
        mapped = {(table.get(a, a), table.get(b, b)) for a, b in eb}
    ripple_ids = [k for k in set(sb) & set(sa) if not touches(sa[k], ta) and sb[k]["status"] != sa[k]["status"]]
    ripple_edges = {"added": [eview(ea[k], lab_a) for k in ea if k not in eb and not set(k) & ta],
                    "removed": [eview(eb[k], lab_b) for k in eb if k not in ea and not set(k) & tb]}
    tens = lambda c, ids: sorted(t["id"] for t in c.get("tensions", []) if {t["x"], t["y"]} & ids)  # noqa: E731

    prev = {
        "kg": {
            "facts_before": sum(1 for f in before["facts"] if {f["s"], f["o"]} & tb and not _is_codebook_pole_fact(f, nb)),
            "facts_after": sum(1 for f in after["facts"] if {f["s"], f["o"]} & ta and not _is_codebook_pole_fact(f, na)),
            "carried": carried, "combined": combined[:SHOW], "n_combined": len(combined),
            "divided": divided[:SHOW], "n_divided": len(divided),
            "collapsed": collapsed[:SHOW], "n_collapsed": len(collapsed),
            "inferred_before": inferred(kb_b, tb), "inferred_after": inferred(kb_a, ta),
            "violations_before": about(kb_b.violations, tb)[:SHOW], "violations_after": about(kb_a.violations, ta)[:SHOW],
            "stats_before": kb_b.stats(), "stats_after": kb_a.stats(),
        },
        "causal": {
            "edges_before": [eview(e, lab_b) for k, e in eb.items() if set(k) & tb],
            "edges_after": [eview(e, lab_a) for k, e in ea.items() if set(k) & ta],
            "edges_new": [eview(e, lab_a) for k, e in ea.items() if set(k) & ta and m["type"] == "merge" and k not in mapped],
            "edges_lost": [eview(e, lab_b) for k, e in eb.items() if set(k) & tb and m["type"] == "merge"
                           and (table.get(k[0], k[0]), table.get(k[1], k[1])) not in ea],
            "constraints": constraints[:SHOW], "n_constraints": len(constraints),
            "combined": sum(1 for c in constraints if len(c["from"]) > 1),
            "collapsed": [{"kind": j["kind"], "locator": j["locator"], "passage": j["passage"], "quote": j["quote"],
                           "held": j["held"], "text": say({"type": "reaches" if j["kind"] != "independent" else "no_path",
                                                           "x": j["was"]["x"], "y": j["was"]["y"], "z": j["was"]["z"],
                                                           "sign": j["was"]["sign"]}, lab_b)}
                          for j in lineage["collapsed_judgments"]][:SHOW],
            "n_collapsed": len(lineage["collapsed_judgments"]),
            "tensions_before": tens(cb_, tb), "tensions_after": tens(ca, ta),
            "ripple": {"status_changed": [{"text": say(sa[k], lab_a), "before": sb[k]["status"], "after": sa[k]["status"]}
                                          for k in sorted(ripple_ids)][:SHOW], **ripple_edges},
            "stats_before": cb_.get("stats"), "stats_after": ca.get("stats"),
            "holdout_before": _agreement(cb_.get("holdout")), "holdout_after": _agreement(ca.get("holdout")),
        },
        "values": {"before": [v for v in before.get("values", []) if v["concept"] in tb],
                   "after": [v for v in after.get("values", []) if v["concept"] in ta], "lossy": lineage["lossy_values"]},
    }
    prev["warnings"] = _warnings(before, m, prev, kb_b)
    prev["summary"] = _summary(before, m, prev)
    return prev


def _warnings(data: dict[str, Any], m: dict[str, Any], prev: dict[str, Any], kb: KnowledgeBase) -> list[str]:
    lab = lambda n: data["nodes"].get(n, {}).get("label", n)  # noqa: E731
    out = []
    if m["type"] == "merge":
        ids = [m["into"], *(x["id"] for x in m["members"])]
        pol = {m["into"]: 1, **{x["id"]: x["polarity"] for x in m["members"]}}
        kinds = {data["nodes"][i]["kind"] for i in ids}
        if len(kinds) > 1:
            kind = lambda i: ("an " if data["nodes"][i]["kind"][0] in "aeiou" else "a ") + data["nodes"][i]["kind"]  # noqa: E731
            out.append(f"different kinds: {', '.join(f'{lab(i)} is {kind(i)}' for i in ids)}")
        for a, b in combinations(ids, 2):
            same = pol[a] == pol[b]
            if same and (a, "opposes", b) in kb.closure:
                out.append(f"the subject holds that {lab(a)} is the contrary of {lab(b)}; merged, it would oppose itself")
            if not same and (a, "same_as", b) in kb.closure:
                out.append(f"the subject calls {lab(a)} another name for {lab(b)}, not its opposite")
            for s, o in ((a, b), (b, a)):
                f = kb.facts.get(fact_key(s, "same_as", o, True))
                if f and f.get("held"):
                    out.append(f"the subject denies that {lab(s)} is the same as {lab(o)}")
                if (s, "is_a", o) in kb.closure:
                    out.append(f"{lab(s)} is a kind of {lab(o)}: merged, that distinction is lost")
                if (s, "part_of", o) in kb.closure:
                    out.append(f"{lab(s)} is part of {lab(o)}: merged, that distinction is lost")
        held = [c for c in prev["causal"]["collapsed"] if c["held"]]
        if prev["causal"]["n_collapsed"]:
            out.append(f"{prev['causal']['n_collapsed']} causal judgment(s) between them say nothing once merged"
                       + (f", e.g. “{held[0]['text']}” ({held[0]['locator']})" if held else ""))
    else:
        default = next(s for s in m["senses"] if s["id"] == m["default"])
        n = m.get("basis", {}).get("default", 0)
        if n:
            out.append(f"{n} passage(s) gave no sign of their sense and went to the default, {default['label']}")
        for s in m["senses"]:
            if not s["passages"] and s["id"] != m["default"]:
                out.append(f"no passage is assigned to {s['label']}")
    if prev["values"]["lossy"]:
        out.append("this Mirror predates raw value records, so values were rebuilt from the evidence shown (rebuild to fix)")
    before, after = len(prev["kg"]["violations_before"]), len(prev["kg"]["violations_after"])
    if after > before:
        out.append(f"inconsistencies involving these concepts go from {before} to {after}")
    return out


def _summary(data: dict[str, Any], m: dict[str, Any], prev: dict[str, Any]) -> str:
    lab = lambda n: data["nodes"].get(n, {}).get("label", n)  # noqa: E731
    k, c = prev["kg"], prev["causal"]
    if m["type"] == "merge":
        names = " and ".join(lab(x["id"]) + (" (as its opposite)" if x["polarity"] == -1 else "") for x in m["members"])
        head = f"Merge {names} into {m.get('label') or lab(m['into'])}"
        kg = f"{k['facts_before']} facts become {k['facts_after']} ({k['n_combined']} combined, {k['n_collapsed']} collapsed)"
    else:
        head = f"Split {lab(m['node'])} into " + ", ".join(f"{s['label']} ({len(s['passages'])})" for s in m["senses"])
        kg = f"{k['facts_before']} facts become {k['facts_after']} ({k['n_divided']} divided between senses)"
    causal = f"arrows here {len(c['edges_before'])} → {len(c['edges_after'])}"
    ripple = len(c["ripple"]["added"]) + len(c["ripple"]["removed"]) + len(c["ripple"]["status_changed"])
    if ripple:
        causal += f", {ripple} change(s) elsewhere in the model"
    hb, ha = c["holdout_before"], c["holdout_after"]
    if not hb or not ha:
        return f"{head}. Knowledge graph: {kg}. Causal model: {causal}. Held-out check not run yet."
    return (f"{head}. Knowledge graph: {kg}. Causal model: {causal}. Held-out claims the rest restates or implies: "
            f"{hb['supported']:.0%} → {ha['supported']:.0%}, contradicted: {hb['contradicted']:.0%} → {ha['contradicted']:.0%}.")


# ---------- building a mutation ----------


def merge(data: dict[str, Any], ids: list[str], into: str | None = None, opposite: list[str] | tuple = (),
          label: str | None = None) -> dict[str, Any]:
    """A merge of `ids` (into the one read from the most passages unless `into` is given); `opposite` are merged at -1."""
    ids = list(dict.fromkeys(ids))
    if len(ids) < 2:
        raise MutationError("a merge needs at least two concepts")
    for i in ids:
        if i not in data["nodes"]:
            raise MutationError(f"no concept {i!r} in this Mirror")
    into = into or max((i for i in ids if i not in opposite), key=lambda i: (len(passages_of(data, i)), -ids.index(i)),
                       default=ids[0])
    if into not in ids:
        ids.insert(0, into)
    m = {"type": "merge", "into": into, "members": [{"id": i, "polarity": -1 if i in opposite else 1}
                                                     for i in ids if i != into]}
    if label:
        m["label"] = label
    check(data, m)
    return m


def passages_of(data: dict[str, Any], node: str) -> list[str]:
    """Every passage anything about the concept (or its opposite pole) was read from, in corpus order."""
    ids = {node} | {p for p, n in data["nodes"].items() if n.get("pole_of") == node}
    out = {e["passage"] for e in data["nodes"][node].get("mentions", [])}
    out |= {e["passage"] for f in data["facts"] if {f["s"], f["o"]} & ids for e in f["evidence"]}
    out |= {j["passage"] for j in data["judgments"] if {j["x"], j["y"], j.get("z", "")} & ids}
    out |= {v["passage"] for v in data.get("value_records") or [] if v["concept"] == node}
    out.discard("codebook")
    order = {pid: i for i, pid in enumerate(data.get("passage_ids", []))}
    return sorted(out, key=lambda p: (order.get(p, 1 << 30), p))


def _words(data: dict[str, Any], node: str) -> dict[str, set[str]]:
    """Passage -> the corpus words the concept was read from there."""
    out: dict[str, set[str]] = defaultdict(set)
    for e in data["nodes"][node].get("mentions", []):
        if e.get("label"):
            out[e["passage"]].add(cb.norm(e["label"]))
    for j in data["judgments"]:
        for k in ("x", "y", "z"):
            if j.get(k) == node and j.get("labels", {}).get(k):
                out[j["passage"]].add(cb.norm(j["labels"][k]))
    return out


def split(data: dict[str, Any], node: str, senses: list[dict[str, Any]], default: int = 0) -> dict[str, Any]:
    """A split of `node`. Each sense: label, gloss, and optionally `passages` (evidence) and `aliases` (its words).

    Passages are assigned by evidence first, then by the words used for the concept in them, then to the default."""
    if node not in data["nodes"]:
        raise MutationError(f"no concept {node!r} in this Mirror")
    taken = set(data["nodes"]) - {node}
    out, basis = [], Counter()
    for s in senses:
        sid = base = cb.slug(s["label"])
        n = 2
        while sid in taken or sid in {o["id"] for o in out}:
            sid, n = f"{base}_{n}", n + 1
        out.append({"id": sid, "label": s["label"].strip(), "gloss": s.get("gloss", ""), "passages": []})
    words = _words(data, node)
    assigned: dict[str, int] = {}
    for i, s in enumerate(senses):  # evidence: a passage claimed by two senses is ambiguous
        for pid in s.get("passages", []):
            assigned[pid] = i if assigned.get(pid, i) == i else -1
    by_word = {cb.norm(a): i for i, s in enumerate(senses) for a in s.get("aliases", [])}
    if not by_word:  # no words given: a word the evidence for one sense uses, and the others never do, marks that sense
        used = [set().union(*(words.get(p, set()) for p in s.get("passages", []))) for s in senses]
        by_word = {w: i for i, ws in enumerate(used) for w in ws - set().union(*(u for k, u in enumerate(used) if k != i))}
    for pid in passages_of(data, node):
        i = assigned.get(pid, -1)
        how = "evidence" if i >= 0 else ""
        if i < 0:
            hits = {by_word[w] for w in words.get(pid, ()) if w in by_word}
            if len(hits) == 1:
                i, how = hits.pop(), "wording"
        if i < 0:
            i, how = default, "default"
        basis[how] += 1
        out[i]["passages"].append(pid)
    m = {"type": "split", "node": node, "senses": out, "default": out[default]["id"], "basis": dict(basis)}
    check(data, m)
    return m


# ---------- finding candidates (no Claude calls) ----------


def _held_reach(data: dict[str, Any]) -> dict[str, dict[tuple, int]]:
    """Per concept, its held causal claims as (direction, other end, sign) -> weight."""
    sig: dict[str, dict[tuple, int]] = defaultdict(dict)
    for c in (data.get("causal") or {}).get("constraints", []):
        if c["type"] == "reaches" and c["held"]:
            sig[c["x"]][("out", c["y"], c["sign"])] = c["weight"]
            sig[c["y"]][("in", c["x"], c["sign"])] = c["weight"]
    return sig


def candidates(data: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """Merges and splits the Mirror's own structure points to, best first: [{mutation, source, reason, score}]."""
    nodes = data["nodes"]
    ok = lambda n: n in nodes and n not in RESERVED and not nodes[n].get("pole_of")  # noqa: E731
    lab = lambda n: nodes.get(n, {}).get("label", n)  # noqa: E731
    kb = KnowledgeBase(nodes, data["facts"])
    out: list[dict[str, Any]] = []
    seen: dict[tuple, dict[str, Any]] = {}

    def add(mut: dict[str, Any], source: str, reason: str, score: float) -> None:
        key = ("split", mut["node"], frozenset(frozenset(s["passages"]) for s in mut["senses"])) \
            if mut["type"] == "split" else ("merge", frozenset([mut["into"], *(x["id"] for x in mut["members"])]))
        if key in seen:  # two signs pointing at the same mutation: one proposal, both reasons
            hit = seen[key]
            hit["reason"] += f"; also, {reason}"
            hit["score"] = round(max(hit["score"], score) + 0.5, 2)
            return
        seen[key] = {"mutation": mut, "source": source, "reason": reason, "score": round(score, 2)}
        out.append(seen[key])

    # 1. the subject says two concepts are one ("same_as"), or each is a kind of the other
    for f in data["facts"]:
        if f["r"] == "same_as" and f["held"] and not f.get("neg") and ok(f["s"]) and ok(f["o"]) and f["s"] != f["o"]:
            add(merge(data, [f["s"], f["o"]]), "same_as",
                f"the subject calls {lab(f['s'])} another name for {lab(f['o'])} ({f['support']} passage(s))",
                3 + f["support"])
    for v in kb.violations:
        if v["type"] == "taxonomy_cycle" and all(ok(n) for n in v["nodes"]):
            add(merge(data, v["nodes"]), "taxonomy_cycle",
                f"each of {lab(v['nodes'][0])} and {lab(v['nodes'][1])} is said to be a kind of the other", 3)

    # 2. causal twins: the subject says the same things cause them and they cause the same things
    sig = _held_reach(data)
    vars_ = [n for n in sig if ok(n) and len(sig[n]) >= 2]
    direct = {(c["x"], c["y"]) for c in (data.get("causal") or {}).get("constraints", []) if c["held"]}
    for a, b in combinations(sorted(vars_), 2):
        if (a, b) in direct or (b, a) in direct:
            continue
        sa = set(sig[a])
        for pol in (1, -1):
            sb = {(d, o, s * pol) for d, o, s in sig[b]}
            shared = sa & sb
            if len(shared) < 2 or len(shared) / len(sa | sb) < 0.5:
                continue
            if pol == 1 and (a, "opposes", b) in kb.closure:
                continue
            w = sum(min(sig[a][k], sig[b][(k[0], k[1], k[2] * pol)]) for k in shared)
            words = ", ".join(f"{lab(o)} {'raises' if s > 0 else 'lowers'} it" if d == "in" else
                              f"it {'raises' if s > 0 else 'lowers'} {lab(o)}" for d, o, s in sorted(shared)[:3])
            opp = [b] if pol == -1 else []
            add(merge(data, [a, b], opposite=opp) if pol == 1 else merge(data, [a, b], into=a, opposite=opp),
                "causal_twins",
                f"{lab(a)} and {lab(b)} play {'the same causal role' if pol == 1 else 'mirror-image causal roles'} "
                f"({len(shared)} shared judgments: {words})" + ("; perhaps one is the other's opposite" if pol == -1 else ""),
                1 + len(shared) / len(sa | sb) + w / 10)

    # 3. one word, two senses: called both good and bad, or said both to raise and to lower the same thing
    for v in kb.violations:
        n = v.get("node")
        if v["type"] != "disjoint_evaluations" or not ok(n):
            continue
        sides = [{"label": f"{lab(n)} ({e})", "gloss": f"{lab(n)} in the passages that call it {e}",
                  "passages": sorted(kb.stated_support((n, "is_a", e)))} for e in v["evaluations"]]
        add(split(data, n, sides, default=max(range(len(sides)), key=lambda i: len(sides[i]["passages"]))),
            "disjoint_evaluations", f"{lab(n)} is held to be both {' and '.join(v['evaluations'])}: perhaps two senses",
            2 + min(len(s["passages"]) for s in sides) / 5)
    for c in (data.get("causal") or {}).get("tensions", []):
        cons = next((k for k in data["causal"]["constraints"] if k["id"] == f"reaches|{c['x']}|{c['y']}"), None)
        if not cons:
            continue
        held = [e for e in cons["evidence"] if e["held"] and e["role"] != "y acts on z"]
        side = {s: sorted({e["passage"] for e in held if e["sign"] == s}) for s in (1, -1)}
        if not side[1] or not side[-1]:
            continue
        words = {n: _words(data, n) for n in (c["x"], c["y"]) if ok(n)}
        # split the end whose wording differs between the two sides; else the cause
        differs = [n for n, w in words.items() if set().union(*(w.get(p, set()) for p in side[1]))
                   != set().union(*(w.get(p, set()) for p in side[-1]))]
        n = (differs or [k for k in (c["x"], c["y"]) if k in words] or [None])[0]
        if n is None:
            continue
        other = c["y"] if n == c["x"] else c["x"]
        verb = {1: "raises", -1: "lowers"}
        sides = [{"label": f"{lab(n)} ({'that ' + verb[s] + ' ' + lab(other) if n == c['x'] else verb[s] + ' by ' + lab(other)})",
                  "gloss": f"{lab(n)} where the subject says {lab(c['x'])} {verb[s]} {lab(c['y'])}", "passages": side[s]}
                 for s in (1, -1)]
        add(split(data, n, sides, default=0 if len(side[1]) >= len(side[-1]) else 1), "sign_conflict",
            f"the subject says {lab(c['x'])} raises {lab(c['y'])} in {len(side[1])} passage(s) and lowers it in "
            f"{len(side[-1])}: perhaps {lab(n)} is used in two senses", 1.5 + min(len(side[1]), len(side[-1])) / 5)
    out.sort(key=lambda c: -c["score"])
    return out[:limit]


# ---------- reading senses with Claude ----------

_STR = {"type": "string"}
SENSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "senses": {"type": "array", "items": {"type": "object", "properties": {"label": _STR, "gloss": _STR},
                                              "required": ["label", "gloss"], "additionalProperties": False}},
        "assign": {"type": "array", "items": {"type": "object", "properties": {"passage": _STR, "sense": {"type": "integer"}},
                                              "required": ["passage", "sense"], "additionalProperties": False}},
    },
    "required": ["senses", "assign"], "additionalProperties": False,
}

SENSE_SYSTEM = """You are checking one concept of the Android Mirror, a model of one subjective system's world (here: {subject}) built only from its own words. The Mirror treats every passage below as using the concept "{label}" ({gloss}). Decide whether the subject uses it in one sense throughout, or in two or more distinct senses (different things that share a word, such as pleasure as a passion to be resisted and pleasure as the joy of acting well).

- senses: if there is one sense, return exactly one. Otherwise 2 to 4 senses, each a short label in the corpus's own wording (lower case, no article) and a one-line gloss. Only distinguish senses the passages themselves distinguish; do not import distinctions from elsewhere.
- assign: for every passage id given, the 0-based index of the sense it uses; -1 if the passage does not show which.{fixed}"""


def claude_senses(subject: str, label: str, gloss: str, passages: list[dict[str, str]], hint: str = "",
                  fixed: list[dict[str, str]] | None = None) -> dict[str, Any]:
    note = ""
    if fixed:
        note = "\n\nThe senses are already fixed; return them unchanged and only assign passages:\n" + \
               "\n".join(f"{i}. {s['label']}: {s['gloss']}" for i, s in enumerate(fixed))
    system = SENSE_SYSTEM.format(subject=subject, label=label, gloss=gloss or "no gloss", fixed=note)
    user = (f"Hint from the Mirror: {hint}\n\n" if hint else "") + "\n\n".join(
        f"[{p['id']}] ({p['locator']})\n{p['text'][:CLAUDE_CHARS]}" for p in passages)
    return llm._json_call(system, user, SENSE_SCHEMA, effort="medium")


def read_senses(data: dict[str, Any], passages: dict[str, dict[str, Any]], node: str, hint: str = "",
                reader: Callable[..., dict[str, Any]] = claude_senses) -> list[dict[str, Any]]:
    """Senses of a concept as Claude reads them in the passages that use it, with each passage assigned."""
    n = data["nodes"][node]
    subject = data["meta"].get("corpus", {}).get("subject", "the subject")
    pids = [p for p in passages_of(data, node) if p in passages]
    if not pids:
        raise MutationError(f"no passages to read for {n['label']}")
    senses: list[dict[str, Any]] | None = None
    assign: dict[str, int] = {}
    for i in range(0, len(pids), CLAUDE_BATCH):
        batch = [passages[p] for p in pids[i:i + CLAUDE_BATCH]]
        r = reader(subject, n["label"], n.get("gloss", ""), batch, hint, senses)
        senses = senses or r["senses"]
        if len(senses) < 2:
            raise MutationError(f"Claude reads {n['label']} in one sense throughout; nothing to split")
        assign.update({a["passage"]: a["sense"] for a in r["assign"] if 0 <= a["sense"] < len(senses)})
    return [{**s, "passages": [p for p in pids if assign.get(p) == i]} for i, s in enumerate(senses)]


# ---------- proposals, approval, undo ----------


def _mid(m: dict[str, Any]) -> str:
    core = {k: v for k, v in m.items() if k != "basis"}
    if core["type"] == "split":
        core = {**core, "senses": [{"id": s["id"], "label": s["label"], "passages": sorted(s["passages"])}
                                   for s in core["senses"]]}
    return "m-" + hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


class Engine:
    """The mutate engine for one built Mirror (data/mirror/<corpus>/). Proposes; writes only on approval."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dir = root / "mutations"

    # storage
    def _mirror(self) -> tuple[dict[str, Any], str]:
        path = self.root / "mirror.json"
        if not path.exists():
            raise FileNotFoundError(f"no Mirror built at {path}")
        raw = path.read_bytes()
        data = json.loads(raw)
        if "judgments" not in data:
            raise MutationError("this Mirror was built by Prototype 2; run build again first")
        return data, _sha(raw)

    def _passages(self) -> dict[str, dict[str, Any]]:
        path = self.root / "passages.json"
        return {p["id"]: p for p in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}

    def _read(self, name: str, default: Any) -> Any:
        path = self.dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def _write(self, name: str, obj: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{name}.tmp"
        tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / name)

    def proposals(self) -> dict[str, dict[str, Any]]:
        return self._read("proposals.json", {})

    def applied(self) -> list[dict[str, Any]]:
        return self._read("applied.json", [])

    def get(self, pid: str) -> dict[str, Any]:
        p = self.proposals().get(pid)
        if p is None:
            raise KeyError(pid)
        return p

    def list(self) -> list[dict[str, Any]]:
        order = {"pending": 0, "applied": 1, "undone": 2, "rejected": 3, "invalid": 4}
        return sorted(self.proposals().values(), key=lambda p: (order.get(p["status"], 9), -p.get("score", 0), p["created"]))

    # proposing (read-only on the Mirror)
    def _record(self, data: dict[str, Any], base: str, mut: dict[str, Any], source: str, reason: str,
                score: float, check: bool = True) -> dict[str, Any]:
        _, preview = evaluate(data, mut, check, answers.load(self.root))
        return {"id": _mid(mut), "mutation": mut, "source": source, "reason": reason, "score": score, "base": base,
                "preview": preview, "status": "pending", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def _store(self, recs: list[dict[str, Any]], drop_stale_detected: bool = False) -> list[dict[str, Any]]:
        allp = self.proposals()
        if drop_stale_detected:
            fresh = {r["id"] for r in recs}
            allp = {k: v for k, v in allp.items() if v["status"] != "pending" or v["source"] == "user" or k in fresh}
        out = []
        for r in recs:
            had = allp.get(r["id"])
            if had and had["status"] in ("rejected", "applied"):  # the user's decision stands
                out.append(had)
                continue
            if had and r["source"] == "user" and had["source"] != "user":  # asked for, and found: keep why it was found
                r = {**r, "source": had["source"], "reason": had["reason"], "score": had["score"]}
            allp[r["id"]] = r
            out.append(r)
        self._write("proposals.json", allp)
        return out

    def propose(self, limit: int = 8, check: bool = False) -> list[dict[str, Any]]:
        """Run the detectors and preview each candidate. Nothing in the Mirror changes.

        The held-out check is left for `check(id)` unless check=True, so that finding candidates stays quick."""
        data, base = self._mirror()
        recs = [self._record(data, base, c["mutation"], c["source"], c["reason"], c["score"], check)
                for c in candidates(data, limit)]
        return self._store(recs, drop_stale_detected=True)

    def check(self, pid: str) -> dict[str, Any]:
        """Add the held-out check to a proposal's preview (against the Mirror as it is now)."""
        allp = self.proposals()
        p = allp.get(pid)
        if p is None:
            raise KeyError(pid)
        if p["status"] != "pending":
            return p
        data, base = self._mirror()
        fresh = self._record(data, base, p["mutation"], p["source"], p["reason"], p["score"])
        allp[pid] = {**fresh, "created": p["created"]}
        self._write("proposals.json", allp)
        return allp[pid]

    def propose_merge(self, ids: list[str], into: str | None = None, opposite: list[str] | tuple = (),
                      label: str | None = None) -> dict[str, Any]:
        data, base = self._mirror()
        mut = merge(data, ids, into, opposite, label)
        return self._store([self._record(data, base, mut, "user", "asked for", 0)])[0]

    def propose_split(self, node: str, senses: list[dict[str, Any]] | None = None, claude: bool = False,
                      reader: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
        """Split `node` into the given senses, or (claude=True) into the senses Claude reads in its passages."""
        data, base = self._mirror()
        if node not in data["nodes"]:
            raise MutationError(f"no concept {node!r} in this Mirror")
        reason = "asked for"
        if claude or reader:
            hint = next((c["reason"] for c in candidates(data, 50)
                         if c["mutation"]["type"] == "split" and c["mutation"]["node"] == node), "")
            senses = read_senses(data, self._passages(), node, hint, reader or claude_senses)
            reason = "senses read by Claude in the passages that use it"
        elif not senses:
            hit = next((c for c in candidates(data, 50) if c["mutation"]["type"] == "split" and c["mutation"]["node"] == node), None)
            if hit is None:
                raise MutationError("give the senses (label and the words or passages for each), or ask Claude to read them")
            return self._store([self._record(data, base, hit["mutation"], hit["source"], hit["reason"], hit["score"])])[0]
        mut = split(data, node, senses)
        return self._store([self._record(data, base, mut, "user", reason, 0)])[0]

    # deciding (the only writes to mirror.json)
    def approve(self, pid: str, approver: str) -> dict[str, Any]:
        """Apply a pending proposal on a user's approval. A stale one is recomputed and handed back, not applied."""
        if not approver or not approver.strip():
            raise MutationError("approving needs the name of whoever approves")
        allp = self.proposals()
        p = allp.get(pid)
        if p is None:
            raise KeyError(pid)
        if p["status"] != "pending":
            raise MutationError(f"proposal {pid} is {p['status']}, not pending")
        data, base = self._mirror()
        if base != p["base"]:
            try:
                fresh = self._record(data, base, p["mutation"], p["source"], p["reason"], p["score"])
            except MutationError as e:
                p.update(status="invalid", note=f"the Mirror changed and this no longer applies: {e}")
                allp[pid] = p
                self._write("proposals.json", allp)
                return {"status": "invalid", "proposal": p}
            allp[pid] = {**fresh, "created": p["created"], "note": "the Mirror changed since this was proposed; recomputed"}
            self._write("proposals.json", allp)
            return {"status": "stale", "proposal": allp[pid]}
        after, _ = evaluate(data, p["mutation"], book=answers.load(self.root))
        backup = self.dir / "backups" / f"{pid}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes((self.root / "mirror.json").read_bytes())
        raw = json.dumps(after, indent=1, ensure_ascii=False).encode("utf-8")
        tmp = self.root / "mirror.json.tmp"
        tmp.write_bytes(raw)
        tmp.replace(self.root / "mirror.json")
        at = time.strftime("%Y-%m-%dT%H:%M:%S")
        log = self.applied()
        log.append({"id": pid, "mutation": p["mutation"], "approver": approver.strip(), "at": at, "before": base,
                    "after": _sha(raw), "backup": str(backup.relative_to(self.root))})
        self._write("applied.json", log)
        p.update(status="applied", approver=approver.strip(), decided=at)
        allp[pid] = p
        self._write("proposals.json", allp)
        return {"status": "applied", "proposal": p}

    def reject(self, pid: str, approver: str = "user") -> dict[str, Any]:
        allp = self.proposals()
        if pid not in allp:
            raise KeyError(pid)
        allp[pid].update(status="rejected", approver=approver, decided=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._write("proposals.json", allp)
        return allp[pid]

    def undo(self, approver: str) -> dict[str, Any]:
        """Undo the last applied mutation: restore the Mirror from before it, or, if it has been rebuilt since,
        take it out of the replay list so the next build leaves it out."""
        log = self.applied()
        if not log:
            raise MutationError("nothing to undo")
        last = log.pop()
        _, now = self._mirror()
        if now == last["after"]:
            (self.root / "mirror.json").write_bytes((self.root / last["backup"]).read_bytes())
            how = "restored"
        else:
            how = "rebuild"
        self._write("applied.json", log)
        allp = self.proposals()
        if last["id"] in allp:
            allp[last["id"]].update(status="undone", undone_by=approver, decided=time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._write("proposals.json", allp)
        return {"id": last["id"], "how": how}


def replay(mirror: dict[str, Any], root: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Re-make the approved mutations on a freshly assembled Mirror (called by build, before the causal fit)."""
    for rec in Engine(root).applied():
        m = rec["mutation"]
        if m["type"] == "split":  # passages read since the split go to the default sense
            known = {p for s in m["senses"] for p in s["passages"]}
            extra = [p for p in passages_of(mirror, m["node"]) if p not in known] if m["node"] in mirror["nodes"] else []
            if extra:
                log(f"mutation {rec['id']}: {len(extra)} new passage(s) go to the default sense")
        try:
            mirror, _ = transform(mirror, m)
        except MutationError as e:
            log(f"mutation {rec['id']} no longer applies, skipped: {e}")
    return mirror


def describe(p: dict[str, Any]) -> list[str]:
    """A proposal as lines of text, for the command line."""
    v = p["preview"]
    out = [f"{p['id']}  [{p['status']}]  {v['summary']}", f"  why: {p['reason']}"]
    out += [f"  ! {w}" for w in v["warnings"]]
    k, c = v["kg"], v["causal"]
    for f in k["combined"][:5]:
        out.append(f"  combined: {f['text']} ({f['support']} passages) <- " + "; ".join(x["text"] for x in f["from"]))
    for f in k["divided"][:5]:
        out.append(f"  divided: {f['text']} -> " + "; ".join(x["text"] for x in f["into"]))
    for f in k["collapsed"][:5]:
        out.append("  collapsed: " + "; ".join(x["text"] for x in f["from"]))
    for e in c["edges_after"][:8]:
        out.append(f"  arrow: {e['text']} (weight {e['weight']}{', open' if e['open'] else ''})")
    for e in c["edges_lost"][:5]:
        out.append(f"  arrow lost: {e['text']}")
    for x in c["constraints"][:8]:
        src = "; ".join(f"{s['text']} [{s['status']}]" for s in x["from"])
        out.append(f"  judgment: {x['text']} w{x['weight']} [{x['status']}]" + (f" <- {src}" if src else ""))
    r = c["ripple"]
    for x in r["status_changed"][:5]:
        out.append(f"  elsewhere: {x['text']}: {x['before']} -> {x['after']}")
    return out
