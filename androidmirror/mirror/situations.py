"""Situations: the units of the subject's own model.

A passage is an act of writing; what the subject's model is about are the
situations the passage reasons about (an insult, an illness, the hour of
death, the world an "if" imagines). Each situation records, for every causal
variable, how it stands there (low -1, typical 0, high 1) and how we know:

- stated: the passage says so (with a quote);
- implied: it follows from what is stated plus the subject's own beliefs in
  the knowledge graph (with the proof), or the subject transformation filled
  it in citing a passage or belief (see subject.py);
- defaulted: the subject's usual level for that variable, not in play here;
- reading: the subject transformation filled it in without a citation, or it
  failed a prior-leak check; Claude's reading rather than the subject's;
- set: a vignette fixed it (see vignettes in subject.py);
- common sense: a vignette answer an ordinary adult gives too, with no
  citation; credited to common sense rather than to the subject.

Anything else is unknown and stays missing, never 0: an unmentioned variable
is unobserved, not absent. Discovery can be run on stated cells only, on
stated and implied (the default), or on everything, and the answers compared.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from . import codebook as cb
from .kg import KnowledgeBase, fact_key

TIERS = ("set", "stated", "implied", "defaulted", "reading", "common sense")
VIEWS = {
    "stated": {"set", "stated"},
    "implied": {"set", "stated", "implied"},
    "all": {"set", "stated", "implied", "defaulted", "reading"},
}
DEFAULT_VIEW = "implied"
TOPIC = "topic"  # the column holding each situation's topic (what its passage is about), when there is one
TYPICAL_MIN = 5  # stated values needed before a variable has a "usual level"
TYPICAL_SHARE = 0.5


def from_reading(passage: dict[str, Any], reading: dict[str, Any], variables: set[str]) -> list[dict[str, Any]]:
    """The situations one Claude reading of a passage found, with their stated cells."""
    out = []
    sits = reading.get("situations", [])
    for j, s in enumerate(sits):
        cells = {}
        for lv in s.get("levels", []):
            v = lv["variable"]
            if v in variables and v not in cells:
                cells[v] = {"level": int(lv["level"]), "tier": "stated", "quote": lv.get("quote", "")}
        if not cells:
            continue
        other = s.get("contrast_with", -1)
        out.append({
            "id": f"{passage['id']}:{j + 1}", "passage": passage["id"], "locator": passage["locator"],
            "order": passage["order"] * 100 + j, "origin": "corpus", "kind": s.get("kind", "actual"),
            "description": s.get("description", ""), "cells": cells,
            "contrast": f"{passage['id']}:{other + 1}" if isinstance(other, int) and 0 <= other < len(sits) and other != j else None,
        })
    return out


def model_claims(passage: dict[str, Any], reading: dict[str, Any], variables: set[str]) -> list[dict[str, Any]]:
    out = []
    for c in reading.get("model_claims", []):
        ends = [c.get(k, "") for k in ("x", "y", "z")]
        used = [e for e in ends if e]
        if len(used) < 2 or any(e not in variables for e in used):
            continue
        out.append({**c, "passage": passage["id"], "locator": passage["locator"]})
    return out


# ---------- the subject's beliefs as implications between levels ----------


def _end(node: str, variables: set[str]) -> tuple[str, int] | None:
    """A KG node as (variable, sign): the variable itself, or its opposite pole."""
    if node in variables:
        return node, 1
    if node.endswith("~") and node[:-1] in variables:
        return node[:-1], -1
    return None


def implications(kb: KnowledgeBase, variables: set[str], labels: dict[str, str]) -> list[dict[str, Any]]:
    """Rules "if v is at level a then w is at level b" from beliefs the subject holds.

    requires: x high needs y high (and so y low means x low); opposes: x high
    means y low, both ways; same_as: the two move together. Only high and low
    propagate; "typical" says nothing. Causal claims are deliberately not used,
    since the causal layer is what is being learned.
    """
    rules = []
    for (s, r, o), proof in kb.closure.items():
        if r not in ("requires", "opposes", "same_as"):
            continue
        a, b = _end(s, variables), _end(o, variables)
        if not a or not b or a[0] == b[0]:
            continue
        (va, sa), (vb, sb) = a, b
        why = {"fact": fact_key(s, r, o), "text": f"{labels.get(s, s)} {r.replace('_', ' ')} {labels.get(o, o)}",
               "rule": proof["rule"], "passages": sorted(kb.stated_support((s, r, o)))[:5]}
        if r == "requires":
            pairs = [((va, sa), (vb, sb)), ((vb, -sb), (va, -sa))]
        elif r == "opposes":
            pairs = [((va, sa), (vb, -sb)), ((vb, sb), (va, -sa))]
        else:
            pairs = [((va, sa), (vb, sb)), ((va, -sa), (vb, -sb)), ((vb, sb), (va, sa)), ((vb, -sb), (va, -sa))]
        for (pv, pl), (cv, cl) in pairs:
            rules.append({"if": [pv, pl], "then": [cv, cl], "why": why})
    return rules


def propagate(situation: dict[str, Any], rules: list[dict[str, Any]]) -> None:
    """Add implied cells to a situation, to a fixpoint. Never overrides a stated or set cell."""
    by_premise = defaultdict(list)
    for r in rules:
        by_premise[tuple(r["if"])].append(r)
    cells = situation["cells"]
    tensions = []
    changed = True
    while changed:
        changed = False
        for v, cell in list(cells.items()):
            if cell["tier"] not in ("set", "stated", "implied") or cell.get("conflicted"):
                continue
            for r in by_premise.get((v, cell["level"]), []):
                cv, cl = r["then"]
                have = cells.get(cv)
                if have is None or have["tier"] in ("defaulted", "reading"):
                    cells[cv] = {"level": cl, "tier": "implied", "source": "belief", "from": [v, cell["level"]],
                                 "why": r["why"]}
                    changed = True
                elif have["level"] != cl and have["tier"] in ("set", "stated"):
                    tensions.append({"variable": cv, "stated": have["level"], "belief_says": cl, "why": r["why"]})
                elif have["level"] != cl and have["tier"] == "implied" and not have.get("conflicted"):
                    have["conflicted"] = True  # two beliefs disagree here: leave it unknown
                    changed = True
    for v in [v for v, c in cells.items() if c.get("conflicted")]:
        del cells[v]
    if tensions:
        situation["tensions"] = tensions[:10]


def typical_levels(situations: list[dict[str, Any]], variables: list[str]) -> dict[str, dict[str, Any]]:
    """Each variable's usual level across the situations where the subject states it."""
    counts: dict[str, Counter] = {v: Counter() for v in variables}
    for s in situations:
        for v, c in s["cells"].items():
            if v in counts and c["tier"] == "stated":
                counts[v][c["level"]] += 1
    out = {}
    for v, cnt in counts.items():
        n = sum(cnt.values())
        if n >= TYPICAL_MIN:
            level, k = cnt.most_common(1)[0]
            if k / n >= TYPICAL_SHARE:
                out[v] = {"level": level, "share": round(k / n, 3), "n": n}
    return out


def add_defaults(situations: list[dict[str, Any]], typical: dict[str, dict[str, Any]]) -> None:
    for s in situations:
        for v, t in typical.items():
            if v not in s["cells"]:
                s["cells"][v] = {"level": t["level"], "tier": "defaulted"}


def table(situations: list[dict[str, Any]], variables: list[str], view: str = DEFAULT_VIEW,
          origins: tuple[str, ...] = ("corpus",), topic: bool = False) -> tuple[dict[str, list], list[str]]:
    """The situation-by-variable table for one view: known levels, None where unknown.

    With `topic`, a "topic" column too, if the situations carry at least two topics."""
    tiers = VIEWS[view]
    rows = [s for s in situations if s["origin"] in origins]
    data = {v: [] for v in variables}
    for s in rows:
        for v in variables:
            c = s["cells"].get(v)
            data[v].append(c["level"] if c and c["tier"] in tiers else None)
    if topic and len({s.get("topic") for s in rows} - {None}) >= 2:
        data[TOPIC] = [s.get("topic") for s in rows]
    return data, [s["id"] for s in rows]


def stated_support(situations: list[dict[str, Any]]) -> Counter:
    return Counter(v for s in situations if s["origin"] == "corpus" for v, c in s["cells"].items() if c["tier"] == "stated")


def resolve_label(book: dict[str, Any], label: str) -> str | None:
    hit = cb.resolve(book, label)
    return hit[0] if hit else None
