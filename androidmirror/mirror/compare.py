"""How two Mirrors differ: in knowledge, in causes, and in their OODA loops.

Transmutate shows this beside its proposal, so you can see what combining would blend
before you approve it. It reads both Mirrors as they are (your approved answers and OODA
placements included) and changes nothing.

Concepts are compared through the alignment (see transmutate.align): a pair is the same idea
in both Mirrors, or contraries (polarity -1), in which case every sign and every good/bad
through it is flipped before comparing. Unpaired concepts can only differ by being absent.

- Knowledge: size; the relations each subject reasons with (share of held facts); facts both
  hold, only one holds, or one holds and the other denies; shared concepts they evaluate
  differently (one calls good what the other calls bad).
- Causes: size and the held-out check; for every pair of shared concepts that either model
  connects, what each says changing one does to the other (raises, lowers, mixed, denies, or
  nothing); shared concepts that drive in one model and are driven in the other.
- OODA: how many concepts sit in each stage and how the stated causes flow between stages;
  whether each loop closes; shared concepts placed in different stages.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from . import ooda
from .kg import fact_key
from .schema import EVALUATIONS, SELF
from .store import Mirror

FLIP = {"good": "bad", "bad": "good"}
ROLE = {"driver": "only a cause", "driven": "only an effect", "both": "a cause and an effect"}
LIMIT = 40  # rows kept per list


class _Map:
    """B's concept ids in A's terms, through the alignment."""

    def __init__(self, pairs: list[dict[str, Any]]) -> None:
        self.to_a = {p["b"]: (p["a"], p["polarity"]) for p in pairs}
        self.pairs = [(p["a"], p["b"], p["polarity"]) for p in pairs]

    def node(self, nid: str) -> str | None:
        """A B fact endpoint as an A node (poles included), or None if it has no partner."""
        if nid in (SELF, *EVALUATIONS):
            return nid
        pole = nid.endswith("~")
        hit = self.to_a.get(nid[:-1] if pole else nid)
        if not hit:
            return None
        a, pol = hit
        return a + "~" if pole != (pol == -1) else a


def _held(m: Mirror) -> dict[str, dict[str, Any]]:
    return {fact_key(f["s"], f["r"], f["o"], f["neg"]): f for f in m.facts if f["held"]}


def _share(c: Counter) -> dict[str, float]:
    n = sum(c.values()) or 1
    return {k: round(v / n, 3) for k, v in c.most_common()}


def knowledge(a: Mirror, b: Mirror, mp: _Map, names: tuple[str, str] = ("A", "B")) -> dict[str, Any]:
    na, nb = names
    fa, fb = _held(a), _held(b)
    rel_a = Counter(f["r"] for f in fa.values() if not f["neg"])
    rel_b = Counter(f["r"] for f in fb.values() if not f["neg"])
    sa, sb = _share(rel_a), _share(rel_b)
    relations = sorted(({"relation": r, "a": sa.get(r, 0.0), "b": sb.get(r, 0.0)} for r in set(sa) | set(sb)),
                       key=lambda x: -abs(x["a"] - x["b"]))
    # facts about shared concepts, in A's terms
    shared_nodes = {SELF, *EVALUATIONS} | {a_ for a_, _, _ in mp.pairs} | {a_ + "~" for a_, _, _ in mp.pairs}
    in_a = {k: f for k, f in fa.items() if f["s"] in shared_nodes and f["o"] in shared_nodes}
    in_b = {}
    for f in fb.values():
        s, o = mp.node(f["s"]), mp.node(f["o"])
        if s and o:
            in_b[fact_key(s, f["r"], o, f["neg"])] = {**f, "s": s, "o": o}
    both = sorted(set(in_a) & set(in_b))
    only_a = sorted(set(in_a) - set(in_b))
    only_b = sorted(set(in_b) - set(in_a))
    denied = []
    for f in in_a.values():
        flip = fact_key(f["s"], f["r"], f["o"], not f["neg"])
        if flip in in_b:
            denied.append({"fact": fact_key(f["s"], f["r"], f["o"]), "a": "denies" if f["neg"] else "holds",
                           "b": "holds" if f["neg"] else "denies",
                           "text": f"{a.label(f['s'])} {f['r'].replace('_', ' ')} {a.label(f['o'])}"})
    evals = []
    for x, y, pol in mp.pairs:
        ea, eb = a.evaluation(x), b.evaluation(y)
        if pol == -1:
            eb = FLIP.get(eb, eb)
        if ea and eb and ea != eb:
            evals.append({"a": x, "b": y, "label": a.label(x), "a_says": ea, "b_says": eb,
                          "text": f"{a.label(x)}: {ea} for {na}, {eb} for {nb}"
                                  + (f" (as the opposite of {nb}'s {b.label(y)})" if pol == -1 else "")})
    def text(k: str) -> str:
        s_, r, o, *neg = k.split("|")
        verb = ("is" if o in EVALUATIONS else "is a kind of") if r == "is_a" else r.replace("_", " ")
        return f"{a.label(s_)} {verb} {a.label(o)}" + (" (denied)" if neg else "")
    union = len(set(in_a) | set(in_b))
    return {
        "a": {"concepts": len(a.nodes), **a.kb.stats()}, "b": {"concepts": len(b.nodes), **b.kb.stats()},
        "relations": relations,
        "shared_facts": {"both": len(both), "only_a": len(only_a), "only_b": len(only_b),
                         "overlap": round(len(both) / union, 3) if union else 0.0},
        "both": [text(k) for k in both[:LIMIT]],
        "only_a": [text(k) for k in sorted(only_a, key=lambda k: -in_a[k]["support"])[:LIMIT]],
        "only_b": [text(k) for k in sorted(only_b, key=lambda k: -in_b[k]["support"])[:LIMIT]],
        "contradicted": denied[:LIMIT], "evaluations": evals[:LIMIT],
    }


def _reach(m: Mirror, start: str) -> set[str]:
    seen, todo = {start}, deque([start])
    while todo:
        n = todo.popleft()
        for k in m.graph_.succ.get(n, {}):
            if k not in seen:
                seen.add(k)
                todo.append(k)
    return seen - {start}


def _says(m: Mirror, x: str, y: str, flip: int = 1) -> str:
    """What m's causal model says changing x does to y (signs flipped by `flip`)."""
    signs = {s * flip for _, s in m.graph_.signed_paths(x, y)} if y in _reach(m, x) else set()
    if signs == {1}:
        return "raises"
    if signs == {-1}:
        return "lowers"
    if signs:
        return "mixed"
    c = m.constraint.get(f"no_path|{x}|{y}")
    return "no effect" if c and c["status"] == "holds" else "silent"


def _role(m: Mirror, n: str) -> str:
    edges = m.causal.get("edges", [])
    i, o = sum(e["dst"] == n for e in edges), sum(e["src"] == n for e in edges)
    return "driver" if o and not i else "driven" if i and not o else "both" if i else "absent"


def causal(a: Mirror, b: Mirror, mp: _Map, names: tuple[str, str] = ("A", "B")) -> dict[str, Any]:
    na, nb = names
    def size(m: Mirror) -> dict[str, Any]:
        st, h = m.causal.get("stats") or {}, (m.causal.get("holdout") or {}).get("positive") or {}
        n = sum(h.values())
        return {"arrows": st.get("arrows", 0), "held_judgments": st.get("held_constraints", 0),
                "in_tension": st.get("rejected", 0),
                "held_out_predicted": round((h.get("restated", 0) + h.get("follows", 0)) / n, 3) if n else None}

    reach_a = {x: _reach(a, x) for x, _, _ in mp.pairs}
    reach_b = {y: _reach(b, y) for _, y, _ in mp.pairs}
    rows = []
    for x, bx, px in mp.pairs:
        for y, by, py in mp.pairs:
            if x == y or (y not in reach_a[x] and by not in reach_b[bx]):
                continue
            sa, sb = _says(a, x, y), _says(b, bx, by, px * py)
            kind = "agree" if sa == sb else ("opposite" if {sa, sb} == {"raises", "lowers"} else
                                             "only a" if sb == "silent" else "only b" if sa == "silent" else
                                             "denied" if "no effect" in (sa, sb) else "differ")
            rows.append({"x": x, "y": y, "x_label": a.label(x), "y_label": a.label(y), "a_says": sa, "b_says": sb,
                         "kind": kind,
                         "text": f"{a.label(x)} → {a.label(y)}: {na} {sa}, {nb} {sb}"})
    order = {"opposite": 0, "denied": 1, "differ": 2, "only a": 3, "only b": 4, "agree": 5}
    rows.sort(key=lambda r: (order[r["kind"]], r["text"]))
    roles = []
    for x, y, _ in mp.pairs:
        ra, rb = _role(a, x), _role(b, y)
        if ra != rb and "absent" not in (ra, rb):
            roles.append({"concept": x, "label": a.label(x), "a": ra, "b": rb,
                          "text": f"{a.label(x)}: {ROLE[ra]} in {na}, {ROLE[rb]} in {nb}"})
    return {"a": size(a), "b": size(b), "counts": dict(Counter(r["kind"] for r in rows)),
            "effects": rows[:LIMIT * 2], "roles": roles[:LIMIT]}


def loops(a: Mirror, b: Mirror, mp: _Map, names: tuple[str, str] = ("A", "B")) -> dict[str, Any]:
    na, nb = names
    ma, mb = ooda.model(a), ooda.model(b)

    def shape(md: dict[str, Any]) -> dict[str, Any]:
        total = sum(sum(r.values()) for r in md["flow"].values()) or 1
        kinds = Counter(l["kind"] for l in md["links"] if l["via"] != "fact")
        return {"stages": {t["id"]: len(t["concepts"]) for t in md["tiers"]},
                "flow": {f"{x}>{y}": round(v / total, 3) for x, r in md["flow"].items() for y, v in r.items() if v},
                "closes": bool(kinds["feedback"]), "links": dict(kinds), "findings": md["findings"]}

    tier = lambda md: {p["id"]: t["id"] for t in md["tiers"] for p in t["concepts"]}  # noqa: E731
    ta, tb = tier(ma), tier(mb)
    moved = [{"concept": x, "label": a.label(x), "a": ta[x], "b": tb[y],
              "text": f"{a.label(x)}: {ooda.LABEL[ta[x]]} in {na}, {ooda.LABEL[tb[y]]} in {nb}"}
             for x, y, _ in mp.pairs if x in ta and y in tb and ta[x] != tb[y]]
    same = sum(1 for x, y, _ in mp.pairs if x in ta and y in tb and ta[x] == tb[y])
    sa, sb = shape(ma), shape(mb)
    flows = sorted(set(sa["flow"]) | set(sb["flow"]), key=lambda k: -abs(sa["flow"].get(k, 0) - sb["flow"].get(k, 0)))
    return {"a": sa, "b": sb, "same_stage": same, "moved": moved[:LIMIT],
            "flow": [{"link": k, "a": sa["flow"].get(k, 0.0), "b": sb["flow"].get(k, 0.0)} for k in flows]}


def headline(names: tuple[str, str], k: dict[str, Any], c: dict[str, Any], o: dict[str, Any], n_pairs: int) -> list[str]:
    """The differences that matter most, in sentences."""
    a, b = names
    out = [(f"{n_pairs} ideas are shared. Of the facts about them, {k['shared_facts']['both']} are held by both, "
            f"{k['shared_facts']['only_a']} only by {a} and {k['shared_facts']['only_b']} only by {b}.")]
    if k["contradicted"]:
        out.append(f"{len(k['contradicted'])} fact(s) one Mirror holds, the other denies.")
    if k["evaluations"]:
        out.append(f"They judge {len(k['evaluations'])} shared idea(s) differently, e.g. {k['evaluations'][0]['text']}.")
    if k["relations"] and abs(k["relations"][0]["a"] - k["relations"][0]["b"]) >= 0.1:
        r = k["relations"][0]
        word = "is a kind of" if r["relation"] == "is_a" else r["relation"].replace("_", " ")
        out.append(f"{a} reasons with '{word}' in {r['a']:.0%} of its facts, {b} in {r['b']:.0%}.")
    cc = c["counts"]
    if cc:
        out.append(f"Between shared concepts the causal models agree on {cc.get('agree', 0)} effect(s), point opposite ways "
                   f"on {cc.get('opposite', 0)}, and one is silent on {cc.get('only a', 0) + cc.get('only b', 0)}.")
    for x in c["roles"][:2]:
        out.append(f"{x['label']} is {ROLE[x['a']]} in {a}'s causal model and {ROLE[x['b']]} in {b}'s.")
    if o["a"]["closes"] != o["b"]["closes"]:
        closes, open_ = (a, b) if o["a"]["closes"] else (b, a)
        out.append(f"{closes}'s OODA loop closes (Act feeds back); {open_}'s does not.")
    if o["flow"] and abs(o["flow"][0]["a"] - o["flow"][0]["b"]) >= 0.1:
        f = o["flow"][0]
        x, y = (ooda.LABEL[s] for s in f["link"].split(">"))
        out.append(f"{x} → {y} carries {f['a']:.0%} of {a}'s stated causation and {f['b']:.0%} of {b}'s.")
    if o["moved"]:
        out.append(f"{len(o['moved'])} shared idea(s) sit in a different OODA stage, e.g. {o['moved'][0]['text']}.")
    return out


def compare(a: Mirror, b: Mirror, pairs: list[dict[str, Any]], names: tuple[str, str] = ("A", "B")) -> dict[str, Any]:
    mp = _Map(pairs)
    k, c, o = knowledge(a, b, mp, names), causal(a, b, mp, names), loops(a, b, mp, names)
    return {"names": list(names), "headline": headline(names, k, c, o, len(pairs)), "knowledge": k, "causal": c, "ooda": o}
