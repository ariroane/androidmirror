"""Merging two Mirrors through their anchors in WordNet (the pivot), for Transmutate.

Each Mirror's approved anchors (anchors.py) say which WordNet sense each concept
is. Read side by side, they give Transmutate four things the shared-name
alignment cannot:

- pairs: both concepts are exactly the same sense (polarity 1), or exactly
  antonymous senses (polarity -1). These merge as today's pairs do.
- blocked: a pair that the same name (or Claude) proposes but the anchors put in
  senses that are neither the same, nor antonyms, nor one a kind of the other,
  nor near each other in WordNet: a homonym. It is not merged; `--pair A=B`
  overrides. A same-name pair where one Mirror says none of the word's senses
  fits and the other names one is blocked too (only for name matches).
- candidates: both concepts anchor to the same sense but at least one only as a
  closeMatch, or a same-sense exact pair that lost out to another partner. Shown
  with both glosses, never merged unless you pair them.
- scaffold: one concept's sense is a kind of the other's (hypernyms, through
  exactMatch, broadMatch and narrowMatch). Both are kept, joined by an `is_a`
  link whose source is the pivot. A scaffold link is never a fact the subject
  holds: it is not in `facts`, so it enters no inference, fitness or trimming.

The rules read only what each anchor carries (the sense's hypernyms with their
distances, and its antonyms), so WordNet itself is not loaded here.
"""

from __future__ import annotations

from typing import Any

REL = {"exactMatch": "eq", "closeMatch": "close", "broadMatch": "below", "narrowMatch": "above",
       "relatedMatch": "related", "noMatch": "none"}
GENERIC_DEPTH = 3  # an upper sense this close to WordNet's root ("state", "act") is too general to connect anything
MAX_STEPS = 5  # hypernym steps a scaffold link may span
NEAR = 3  # two senses whose nearest shared hypernym is this few steps away (summed) are near, not disjoint


def _up(anchor: dict[str, Any]) -> dict[str, int]:
    """The anchor's sense and everything above it, with distances."""
    return {anchor["object"]: 0, **anchor.get("up", {})}


def relation(x: dict[str, Any] | None, y: dict[str, Any] | None) -> tuple[str, str]:
    """How two anchors' senses relate: (verdict, reason). Verdicts:
    unknown (an anchor is missing or both say no sense fits), same, opposite, kind_of (one is above the other),
    near (nearest shared hypernym within NEAR steps), none_fits (only one says no sense fits), disjoint."""
    if not x or not y:
        return "unknown", "not both anchored"
    xn, yn = x["predicate"] == "noMatch", y["predicate"] == "noMatch"
    if xn and yn:
        return "unknown", "neither has a WordNet sense"
    if xn or yn:
        return "none_fits", f"{(x if xn else y)['label']!r} is none of the word's WordNet senses; " \
                            f"{(y if xn else x)['label']!r} is {(y if xn else x)['object']} {(y if xn else x).get('object_label', '')}"
    sx, sy = x["object"], y["object"]
    pol = x.get("polarity", 1) * y.get("polarity", 1)
    if sx == sy:
        return ("same" if pol == 1 else "opposite"), f"both {sx} {x.get('object_label', '')}"
    if sy in x.get("antonyms", []) or sx in y.get("antonyms", []):
        return ("opposite" if pol == 1 else "same"), f"{x.get('object_label', sx)} and {y.get('object_label', sy)} are antonyms"
    ux, uy = _up(x), _up(y)
    if sx in uy or sy in ux:
        lo, hi = (y, x) if sx in uy else (x, y)
        return "kind_of", f"{lo.get('object_label', '')} is a kind of {hi.get('object_label', '')} in WordNet"
    common = [(ux[s] + uy[s], s) for s in ux.keys() & uy.keys()]
    if common and min(common)[0] <= NEAR:
        return "near", f"near senses ({x.get('object_label', sx)} / {y.get('object_label', sy)})"
    return "disjoint", f"different senses: {sx} {x.get('object_label', '')} ({x.get('lexfile', '')}) vs " \
                       f"{sy} {y.get('object_label', '')} ({y.get('lexfile', '')})"


def anchor_pairs(a_anch: dict[str, dict[str, Any]], b_anch: dict[str, dict[str, Any]],
                 a_nodes: dict[str, Any], b_nodes: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact same-sense and antonym pairs (best first, not yet one-to-one), and closeMatch candidates."""
    by_sense: dict[str, list[str]] = {}
    for bid, p in b_anch.items():
        if p["predicate"] in ("exactMatch", "closeMatch") and bid in b_nodes:
            by_sense.setdefault(p["object"], []).append(bid)
    pairs, cands = [], []
    for aid, pa in a_anch.items():
        if pa["predicate"] not in ("exactMatch", "closeMatch") or aid not in a_nodes:
            continue
        for sense, flip in [(pa["object"], 1)] + [(s, -1) for s in pa.get("antonyms", [])]:
            for bid in by_sense.get(sense, []):
                pb = b_anch[bid]
                pol = flip * pa.get("polarity", 1) * pb.get("polarity", 1)
                row = {"a": aid, "b": bid, "polarity": pol, "sense": sense,
                       "why": (f"both are {sense} {pa.get('object_label', '')}" if flip == 1 else
                               f"{pa.get('object_label', '')} and {pb.get('object_label', '')} are WordNet antonyms"),
                       "_rank": (pa["confidence"] + pb["confidence"],
                                 a_nodes[aid].get("support", 0) + b_nodes[bid].get("support", 0))}
                if pa["predicate"] == pb["predicate"] == "exactMatch":
                    pairs.append({**row, "how": "anchor" if flip == 1 else "anchor (antonym)"})
                else:
                    cands.append({**row, "how": "closeMatch",
                                  "a_gloss": a_nodes[aid].get("gloss", ""), "b_gloss": b_nodes[bid].get("gloss", "")})
    key = lambda p: p["_rank"]  # noqa: E731
    return sorted(pairs, key=key, reverse=True), sorted(cands, key=key, reverse=True)


def screen(pairs: list[dict[str, Any]], a_anch: dict[str, dict[str, Any]], b_anch: dict[str, dict[str, Any]]
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split name and Claude pairs into those the anchors allow and those they block (homonyms)."""
    kept, blocked = [], []
    for p in pairs:
        verdict, why = relation(a_anch.get(p["a"]), b_anch.get(p["b"]))
        names = p["how"] in ("same label", "alias")
        if verdict in ("disjoint", "kind_of") or (verdict == "none_fits" and names):
            blocked.append({**p, "verdict": verdict, "reason": why})
            continue
        if verdict in ("same", "opposite"):
            pol = 1 if verdict == "same" else -1
            if pol != p["polarity"]:
                blocked.append({**p, "verdict": "polarity", "reason": f"the anchors say {verdict}, the pair says "
                                f"{'same' if p['polarity'] == 1 else 'opposite'}"})
                continue
        kept.append({**p, "anchors": verdict} if verdict != "unknown" else p)
    return kept, blocked


def scaffold(a_id: str, b_id: str, a_anch: dict[str, dict[str, Any]], b_anch: dict[str, dict[str, Any]],
             merged: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    """Cross-Mirror is_a links from the pivot: the lower concept's sense lies under the upper's, nearest upper only.

    `merged` maps (side, concept) to its combined id, so a link inside one merged node is skipped."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for lo_side, lo_anch, hi_side, hi_anch in (("a", a_anch, "b", b_anch), ("b", b_anch, "a", a_anch)):
        uppers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for cid, p in hi_anch.items():
            if REL.get(p["predicate"]) in ("eq", "above") and p.get("depth", 99) > GENERIC_DEPTH \
                    and p.get("polarity", 1) == 1:
                uppers.setdefault(p["object"], []).append((cid, p))
        for cid, p in lo_anch.items():
            rel = REL.get(p["predicate"])
            if rel not in ("eq", "below") or p.get("polarity", 1) != 1:
                continue
            best: list[tuple[int, str, dict[str, Any], str]] = []
            for sense, d in _up(p).items():
                if d > MAX_STEPS or sense not in uppers:
                    continue
                for hid, hp in uppers[sense]:
                    strict = d > 0 or rel == "below" or REL[hp["predicate"]] == "above"
                    if strict:
                        best.append((d, hid, hp, sense))
            if not best:
                continue
            dmin = min(b[0] for b in best)
            for d, hid, hp, sense in best:
                if d != dmin:
                    continue
                lo_c, hi_c = merged[(lo_side, cid)], merged[(hi_side, hid)]
                if lo_c == hi_c or lo_c.rstrip("~") == hi_c.rstrip("~"):
                    continue
                out.setdefault((lo_c, hi_c), {
                    "s": lo_c, "r": "is_a", "o": hi_c, "source": "pivot", "steps": d, "via": sense,
                    "lower": {"mirror": a_id if lo_side == "a" else b_id, "concept": cid, "sense": p["object"],
                              "sense_label": p.get("object_label", "")},
                    "upper": {"mirror": a_id if hi_side == "a" else b_id, "concept": hid, "sense": hp["object"],
                              "sense_label": hp.get("object_label", "")}})
    return sorted(out.values(), key=lambda r: (r["steps"], r["s"], r["o"]))
