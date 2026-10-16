"""The codebook: one fixed set of concepts, and how each corpus label maps onto one.

Labels from open coding are first grouped by a conservative normal form
(case, articles, a plural s). If Claude is available, the groups are then
consolidated by meaning (synonyms merged, opposites folded in with polarity -1,
word senses kept apart). Without Claude the normal-form grouping stands.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Callable

from .schema import EVALUATIONS, SELF, VARIABLE_KINDS

BATCH = 60  # labels per consolidation call; larger batches can overflow Claude's answer
MIN_BATCH = 8


def norm(label: str) -> str:
    s = re.sub(r"[^a-z0-9' -]+", " ", label.lower()).strip()
    s = re.sub(r"^(the|a|an|one's|his|her|my|our)\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 4 and s.endswith("s") and not s.endswith(("ss", "us", "is", "'s")):
        s = s[:-1]
    return s


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "concept"


def collect_mentions(codings: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per normalized label: kind votes, glosses, and the passages it appears in."""
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"labels": Counter(), "kinds": Counter(), "glosses": [],
                                                          "passages": set()})
    for pid, c in codings.items():
        for m in c.get("concepts", []):
            n = norm(m["label"])
            if not n or n == SELF or n in EVALUATIONS:
                continue
            rec = out[n]
            rec["labels"][m["label"].strip().lower()] += 1
            rec["kinds"][m.get("kind", "concept")] += 1
            if m.get("gloss") and len(rec["glosses"]) < 3:
                rec["glosses"].append(m["gloss"])
            rec["passages"].add(pid)
        # labels used in statements or claims but not listed as concepts
        refs = [(s["subject"], s["object"]) for s in c.get("statements", [])] + \
               [(x["cause"], x["effect"]) for x in c.get("causal_claims", [])]
        for a, b in refs:
            for lab in (a, b):
                n = norm(lab)
                if n and n != SELF and n not in EVALUATIONS and n not in out:
                    out[n]["labels"][lab.strip().lower()] += 1
                    out[n]["kinds"]["concept"] += 1
                    out[n]["passages"].add(pid)
    return dict(out)


def build(codings: dict[str, dict[str, Any]], subject: str,
          consolidate: Callable[[list[str], str], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    mentions = collect_mentions(codings)
    concepts: dict[str, dict[str, Any]] = {}
    label_map: dict[str, list] = {}  # norm label -> [concept id, polarity]

    def add(cid: str, label: str, kind: str, gloss: str, variable: bool) -> str:
        if cid not in concepts:
            concepts[cid] = {"label": label, "kind": kind, "gloss": gloss, "variable": variable, "aliases": []}
        return cid

    add(SELF, subject, "agent", "the subject the Mirror reflects", False)
    for e in EVALUATIONS:
        add(e, e, "class", f"the evaluation '{e}'", False)

    if consolidate:
        lines = []
        for n, rec in sorted(mentions.items(), key=lambda kv: (kv[1]["kinds"].most_common(1)[0][0], -len(kv[1]["passages"]))):
            gloss = rec["glosses"][0] if rec["glosses"] else ""
            lines.append(f"{n} | {rec['kinds'].most_common(1)[0][0]} | {len(rec['passages'])} | {gloss}")
        groups: list[dict[str, Any]] = []
        for i in range(0, len(lines), BATCH):
            groups += _consolidate_safely(consolidate, lines[i:i + BATCH], subject)
        for c in groups:
            cid = slug(c["id"] or c["label"])
            if cid in (SELF, *EVALUATIONS):
                continue
            add(cid, c["label"], c["kind"], c["gloss"], bool(c["variable"]))
            for m in c["members"]:
                n = norm(m["label"])
                if n in mentions and n not in label_map:
                    label_map[n] = [cid, m["polarity"]]

    for n, rec in mentions.items():  # anything consolidation left out stands on its own
        if n in label_map:
            continue
        cid = slug(n)
        while cid in concepts and concepts[cid]["label"] != n:
            cid += "_"
        kind = rec["kinds"].most_common(1)[0][0]
        add(cid, n, kind, rec["glosses"][0] if rec["glosses"] else "", kind in VARIABLE_KINDS)
        label_map[n] = [cid, 1]

    for n, (cid, pol) in label_map.items():
        for lab in mentions[n]["labels"]:
            if lab != concepts[cid]["label"].lower() and lab not in concepts[cid]["aliases"]:
                concepts[cid]["aliases"].append(lab if pol == 1 else f"{lab} (opposite)")
    return {"concepts": concepts, "label_map": label_map}


def _consolidate_safely(consolidate, lines: list[str], subject: str) -> list[dict[str, Any]]:
    """Consolidate one batch; if the answer fails, halve the batch and retry.

    A batch that still fails at MIN_BATCH labels is left to the normal-form
    grouping, so one bad answer never stops a build.
    """
    from ..llm import LLMUnavailable
    try:
        return consolidate(lines, subject)
    except LLMUnavailable:
        raise
    except Exception:
        if len(lines) <= MIN_BATCH:
            return []
        mid = len(lines) // 2
        return _consolidate_safely(consolidate, lines[:mid], subject) + _consolidate_safely(consolidate, lines[mid:], subject)


def resolve(codebook: dict[str, Any], label: str) -> tuple[str, int] | None:
    """The concept id and polarity a corpus label maps to."""
    n = norm(label)
    if n == SELF or label.strip().lower() == SELF:
        return SELF, 1
    if n in EVALUATIONS:
        return n, 1
    hit = codebook["label_map"].get(n)
    return (hit[0], hit[1]) if hit else None
