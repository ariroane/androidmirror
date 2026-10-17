"""The user's answers to the Question engine, and how approved ones amend a Mirror.

Answers live beside the Mirror in answers.json, one proposal per answer:

    {"proposals": [{"id", "question_id", "kind", "question", "option", "answer", "note",
                    "changes": [...], "status": "proposed" | "approved" | "rejected" | "revoked",
                    "created", "approved_by", "approved_at", "preview"}]}

Only approved proposals change anything, and they never touch what was read
from the corpus: `overlay` lays them over the corpus-built Mirror each time it
is loaded or re-fitted, so a rebuild keeps them and revoking one undoes it.

An approved answer is a source of its own, shown as the passage
"answer:<proposal id>". Its changes are:

- judgment {kind, x, y, z, sign, scope}: a causal judgment, as if a passage made it;
- ruling {target, ruling: affirm | retract | common_cause, sign?, direct?, indirect?, via?}:
  a ruling on a constraint the corpus gives. Affirmed constraints are fitted
  before any other (the subject has settled them); retracted ones carry no
  weight; common_cause says neither end drives the other, something else
  drives both (`via`, a concept, when the user names it): the claim is never
  drawn as an arrow, only as a two-way link between its ends;
- fact {s, r, o, neg}: a descriptive fact the subject holds;
- retract_fact {key}: a fact that stays on record but no longer enters inference;
- value {concept, direction}: what the subject wants more (1) or less (-1) of.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .kg import fact_key
from .schema import SELF

PREFIX = "answer:"
REGISTER = "answers"


def path(root: Path) -> Path:
    return root / "answers.json"


def load(root: Path) -> dict[str, Any]:
    p = path(root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"proposals": []}


def save(root: Path, book: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = path(root).with_suffix(".tmp")
    tmp.write_text(json.dumps(book, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path(root))


def approved(book: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in book.get("proposals", []) if p["status"] == "approved"]


def source(p: dict[str, Any]) -> dict[str, str]:
    """The pseudo-passage an approved answer stands as."""
    when = (p.get("approved_at") or p.get("created") or "")[:10]
    text = f"Q: {p['question']}\nA: {p['answer']}" + (f"\nNote: {p['note']}" if p.get("note") else "")
    return {"passage": PREFIX + p["id"], "locator": f"Answer{' by ' + p['approved_by'] if p.get('approved_by') else ''}"
            f"{', ' + when if when else ''}", "quote": p["answer"] + (f" ({p['note']})" if p.get("note") else ""), "text": text}


def parts(cid: str) -> tuple[str, str, str]:
    """A constraint id as (x, y, z)."""
    bits = cid.split("|")[1:] + ["", "", ""]
    return bits[0], bits[1], bits[2]


def overlay(mirror: dict[str, Any], book: dict[str, Any]) -> dict[str, Any]:
    """The Mirror with the approved answers laid over it (a new dict; `mirror` is not changed)."""
    props = approved(book)
    out = dict(mirror)
    out["answer_passages"] = []
    if not props:
        return out
    judgments = list(mirror.get("judgments", []))
    facts = {fact_key(f["s"], f["r"], f["o"], f.get("neg", False)): f for f in mirror.get("facts", [])}
    values = {v["concept"]: v for v in mirror.get("values", [])}
    nodes = mirror.get("nodes", {})
    for p in props:
        src = source(p)
        out["answer_passages"].append({"id": src["passage"], "locator": src["locator"], "text": src["text"],
                                       "title": "Your answer", "order": 10**9})
        base = {"passage": src["passage"], "locator": src["locator"], "quote": src["quote"], "register": REGISTER,
                "attribution": SELF, "labels": {}}
        for ch in p["changes"]:
            t = ch["type"]
            if t == "judgment":
                judgments.append({**base, "kind": ch["kind"], "x": ch["x"], "y": ch["y"], "z": ch.get("z", ""),
                                  "sign": ch.get("sign", 0), "scope": ch.get("scope", "unsaid"), "held": True})
            elif t == "ruling":
                x, y, z = parts(ch["target"])
                judgments.append({**base, "kind": "ruling", "target": ch["target"], "ruling": ch["ruling"],
                                  "x": x, "y": y, "z": z, "sign": ch.get("sign", 0), "scope": "unsaid",
                                  "direct": bool(ch.get("direct")), "indirect": bool(ch.get("indirect")), "via": ch.get("via", ""),
                                  "held": False})
            elif t == "fact":
                if ch["s"] not in nodes or ch["o"] not in nodes:
                    continue
                k = fact_key(ch["s"], ch["r"], ch["o"], ch.get("neg", False))
                f = dict(facts.get(k) or {"s": ch["s"], "r": ch["r"], "o": ch["o"], "neg": bool(ch.get("neg")),
                                          "evidence": [], "held": False, "qualifiers": []})
                f["evidence"] = [{"passage": src["passage"], "locator": src["locator"], "quote": src["quote"],
                                  "modality": "asserts", "attribution": SELF}] + list(f["evidence"])
                f["held"], f["retracted"] = True, False
                f["support"] = len({e["passage"] for e in f["evidence"]})
                f["modalities"] = sorted({e["modality"] for e in f["evidence"]})
                facts[k] = f
            elif t == "retract_fact":
                if ch["key"] in facts:
                    facts[ch["key"]] = {**facts[ch["key"]], "held": False, "retracted": True,
                                        "retracted_by": src["passage"]}
            elif t == "value":
                v = dict(values.get(ch["concept"]) or {"concept": ch["concept"], "more": 0, "less": 0, "direction": 0,
                                                       "evidence": []})
                d = 1 if ch["direction"] > 0 else -1
                v["more" if d > 0 else "less"] = v["more" if d > 0 else "less"] + 1
                v["direction"] = d  # an explicit answer settles what the passages left split
                v["answered"] = d
                v["evidence"] = [{"concept": ch["concept"], "direction": d, "passage": src["passage"],
                                  "locator": src["locator"], "quote": src["quote"]}] + list(v["evidence"])
                values[ch["concept"]] = v
    out["judgments"] = judgments
    out["facts"] = list(facts.values())
    out["values"] = sorted(values.values(), key=lambda r: -(r["more"] + r["less"]))
    return out
