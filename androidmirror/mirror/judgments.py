"""The subject's causal judgments: the data the causal layer is built from.

A judgment is one thing the subject says about how things depend on each other,
in one passage. The text is not treated as a sample of situations: how often
two ideas come up together says nothing here. What counts is what the subject
asserts, and each assertion counts once per passage, so a maxim repeated forty
times is strong evidence that the subject holds it, but it is still one belief.

Kinds (see coding.JUDGMENT_SYSTEM for the wording Claude reads them by):

- causes: x brings about y (sign + or -), directly or overall;
- intervention: the subject's own thought experiment: add or remove x and see
  what happens to y (y moves with it, against it, or stays);
- means_end: x is a way to y;
- screens_off: not x but y does z (y acts on z; x acts on z only through y);
- independent: x makes no difference to y.

They become constraints on the subject's model:

- reaches (x, y, sign): x is a cause of y, directly or through other things;
  from causes, means_end, an intervention that moves y, and the y -> z half of
  screens_off;
- no_path (x, y): x is not a cause of y at all; from independent and an
  intervention that leaves y as it was;
- only_through (x, m, z): every way x acts on z passes through m (stored as x, y = m, z).

Only the subject's own judgments count (attribution self, register describes or
urges). Reported and entertained views are kept and shown, never fitted.

Answers the user approved through the Question engine (answers.py) arrive as
judgments of their own (register "answers") and as rulings on a constraint.
A constraint with an answer behind it is pinned: the fit takes it first. A
retracted one stays on record with no weight.

Values are kept apart: what the passage urges or praises ("be free of anger")
says what the subject wants, not how things work (review recommendation R3).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from . import codebook as cb
from .coding import HELD_REGISTERS
from .schema import EVALUATIONS, SELF

ANSWERED = "answers"  # the register of a judgment the user gave in answer to a question
POSITIVE = {"causes", "means_end", "intervention", "screens_off"}


def resolve_endpoint(book: dict[str, Any], label: str) -> tuple[str, int] | None:
    """A judgment endpoint (a codebook id, or a label) as (concept id, polarity)."""
    label = (label or "").strip()
    if not label:
        return None
    if label in book["concepts"]:
        return label, 1
    return cb.resolve(book, label)


def read(passage: dict[str, Any], reading: dict[str, Any], book: dict[str, Any], subject_label: str) -> dict[str, Any]:
    """One passage's reading as resolved judgments and values. Endpoints outside the codebook are kept as `unresolved`."""
    selfish = {SELF, subject_label.strip().lower()}
    out = {"judgments": [], "values": [], "unresolved": []}
    for j in reading.get("judgments", []):
        kind = j["kind"]
        ends = {}
        missing = []
        for k in ("x", "y", "z"):
            if k == "z" and kind != "screens_off":
                continue
            hit = resolve_endpoint(book, j.get(k, ""))
            if hit is None or hit[0] in EVALUATIONS:
                missing.append(j.get(k, ""))
            else:
                ends[k] = hit
        attribution = SELF if j.get("attribution", SELF).strip().lower() in selfish else j["attribution"].strip()
        base = {"kind": kind, "passage": passage["id"], "locator": passage["locator"], "quote": j.get("quote", ""),
                "register": j.get("register", "describes"), "attribution": attribution, "scope": j.get("scope", "unsaid"),
                "labels": {k: j.get(k, "") for k in ("x", "y", "z") if j.get(k)}}
        if missing:
            out["unresolved"].append({**base, "missing": missing})
            continue
        x, px = ends["x"]
        y, py = ends["y"]
        if kind == "screens_off":
            z, pz = ends["z"]
            if len({x, y, z}) < 3:
                continue
            sign = (j.get("sign") or 1) * py * pz
            rec = {**base, "x": x, "y": y, "z": z, "sign": 1 if sign >= 0 else -1}
        else:
            if x == y:
                continue
            raw = j.get("sign", 0) or 0
            if kind == "independent":
                sign = 0
            elif kind == "intervention" and raw == 0:
                sign = 0
            else:
                sign = (raw or 1) * px * py
            rec = {**base, "x": x, "y": y, "z": "", "sign": sign}
        rec["held"] = attribution == SELF and rec["register"] in HELD_REGISTERS
        out["judgments"].append(rec)
    for v in reading.get("values", []):
        hit = resolve_endpoint(book, v.get("concept", ""))
        if hit is None or hit[0] in EVALUATIONS or hit[0] == SELF:
            continue
        out["values"].append({"concept": hit[0], "direction": v["direction"] * hit[1], "passage": passage["id"],
                              "locator": passage["locator"], "quote": v.get("quote", "")})
    return out


def constraints(judgments: list[dict[str, Any]], exclude: set[str] | frozenset = frozenset()) -> list[dict[str, Any]]:
    """The subject's held judgments as weighted constraints, one per claim, weighted by distinct passages.

    Every judgment, held or not, is listed in the constraint it bears on; only held ones add weight."""
    groups: dict[tuple, dict[str, Any]] = {}
    rulings = {j["target"]: j for j in judgments if j["kind"] == "ruling" and j["passage"] not in exclude}

    def add(key: tuple, j: dict[str, Any], sign: int | None, role: str) -> None:
        rec = groups.setdefault(key, {"type": key[0], "x": key[1], "y": key[2], "z": key[3] if len(key) > 3 else "",
                                      "evidence": [], "signs": Counter(), "kinds": Counter(), "direct": False})
        rec["evidence"].append({**{k: j[k] for k in ("kind", "passage", "locator", "quote", "register", "attribution",
                                                     "scope", "sign", "held")}, "role": role})
        if j["held"]:
            if sign is not None:
                rec["signs"][sign] += 1
            rec["kinds"][j["kind"]] += 1
            rec["direct"] = rec["direct"] or (j["kind"] == "causes" and j["scope"] == "direct")

    for j in judgments:
        if j["passage"] in exclude:
            continue
        k = j["kind"]
        if k == "ruling":
            continue
        if k == "screens_off":
            add(("reaches", j["y"], j["z"]), j, j["sign"], "y acts on z")
            add(("only_through", j["x"], j["y"], j["z"]), j, None, "x only through y")
        elif k == "independent" or (k == "intervention" and j["sign"] == 0):
            add(("no_path", j["x"], j["y"]), j, None, "makes no difference")
        else:
            add(("reaches", j["x"], j["y"]), j, j["sign"], k)
    out = []
    for key, rec in groups.items():
        held = [e for e in rec["evidence"] if e["held"]]
        rec["weight"] = len({e["passage"] for e in held})
        rec["mentions"] = len(held)  # how often it is said, repetitions included: shown, never used as weight
        rec["id"] = "|".join([rec["type"], rec["x"], rec["y"]] + ([rec["z"]] if rec["z"] else []))
        if rec["type"] == "reaches":
            plus, minus = rec["signs"][1], rec["signs"][-1]
            rec["sign"] = 1 if plus >= minus else -1
            rec["sign_conflict"] = bool(plus and minus)
        else:
            rec["sign"] = 0
            rec["sign_conflict"] = False
        rec["kinds"] = dict(rec["kinds"])
        rec["signs"] = {str(k): v for k, v in rec["signs"].items()}
        rec["held"] = rec["weight"] > 0
        answers = [e for e in held if e["register"] == ANSWERED]
        rec["pinned"] = bool(answers)
        if answers and rec["type"] == "reaches":
            rec["sign"] = answers[-1]["sign"] or rec["sign"]
        r = rulings.get(rec["id"])
        if r:
            rec["ruling"] = {"ruling": r["ruling"], "passage": r["passage"], "locator": r["locator"], "quote": r["quote"]}
            if r["ruling"] == "retract":
                rec.update(weight=0, held=False, pinned=False, retracted=True)
            else:
                rec["pinned"] = True
                rec["sign"] = r["sign"] or rec["sign"]
                rec["direct"] = rec["direct"] or r["direct"]
                rec["indirect"] = r["indirect"] and not r["direct"]
        out.append(rec)
    return sorted(out, key=lambda r: (-r["weight"], r["id"]))


def values(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The values layer: per concept, how many passages urge more of it and how many urge less."""
    by: dict[str, dict[str, Any]] = defaultdict(lambda: {"more": set(), "less": set(), "evidence": []})
    for v in records:
        rec = by[v["concept"]]
        (rec["more"] if v["direction"] > 0 else rec["less"]).add(v["passage"])
        if len(rec["evidence"]) < 8:
            rec["evidence"].append(v)
    out = []
    for cid, rec in by.items():
        more, less = len(rec["more"]), len(rec["less"])
        out.append({"concept": cid, "more": more, "less": less, "direction": 1 if more > less else -1 if less > more else 0,
                    "evidence": rec["evidence"]})
    return sorted(out, key=lambda r: -(r["more"] + r["less"]))
