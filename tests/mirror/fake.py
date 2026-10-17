"""An offline stand-in for Claude, and a synthetic corpus with a known causal structure.

Passages carry their own coding as tags, e.g. "[v anger -1] [s justice is_a virtue] [c anger distress 1 explicit]",
so the build pipeline can run end to end without a key. Each passage is one situation; a variable the passage
does not tag is unknown there (mentioned at random, independently of its level).
"""

from __future__ import annotations

import random
import re
from pathlib import Path

TAG = re.compile(r"\[(v|s|c|n|m) ([^\]]+)\]")
# the planted model: alpha -> gamma <- beta, gamma -> delta -> epsilon (negative), zeta on its own
PARENTS = {"gamma": [("alpha", 1), ("beta", 1)], "delta": [("gamma", 1)], "epsilon": [("delta", -1)]}
ORDER = ["alpha", "beta", "zeta", "gamma", "delta", "epsilon"]
TOPIC_WORDS = ["grave dust corpse mortal ashes", "senate emperor council office law", "stars cosmos river flux sea"]


def _level(score: float) -> int:
    return 1 if score > 0.6 else -1 if score < -0.6 else 0


class FakeCoder:
    uses_llm = False

    def __init__(self) -> None:
        self.calls = {"code": 0, "consolidate": 0, "situations": 0, "answer": 0}

    def code(self, text, locator, subject):
        self.calls["code"] += 1
        out = {"concepts": [], "statements": [], "causal_claims": []}
        for kind, body in TAG.findall(text):
            parts = body.split()
            if kind == "v":
                label, level = parts[0].replace("_", " "), int(parts[1])
                out["concepts"].append({"label": label, "kind": parts[2] if len(parts) > 2 else "state",
                                        "gloss": f"gloss of {label}", "level": 1 if level >= 0 else -1, "quote": text[:30]})
            elif kind == "s" or kind == "n":
                s, r, o = parts[:3]
                out["statements"].append({"subject": s.replace("_", " "), "relation": r, "object": o.replace("_", " "),
                                          "negated": kind == "n", "modality": parts[3] if len(parts) > 3 else "asserts",
                                          "attribution": parts[4] if len(parts) > 4 else "self", "qualifier": "",
                                          "wording": "", "quote": text[:30]})
            elif kind == "c":
                out["causal_claims"].append({"cause": parts[0].replace("_", " "), "effect": parts[1].replace("_", " "),
                                             "sign": int(parts[2]), "direction_basis": parts[3], "scope": "direct",
                                             "strength_word": "", "modality": "asserts", "attribution": "self",
                                             "quote": text[:30]})
        return out

    def consolidate(self, lines, subject):
        self.calls["consolidate"] += 1
        return []

    def situations(self, text, locator, subject, variables):
        self.calls["situations"] += 1
        ids = {v["id"] for v in variables}
        levels, claims = [], []
        for kind, body in TAG.findall(text):
            parts = body.split()
            if kind == "v" and parts[0] in ids:
                levels.append({"variable": parts[0], "level": int(parts[1]), "quote": text[:20]})
            elif kind == "m":
                x, y, z = (parts[1:] + ["", "", ""])[:3]
                claims.append({"kind": parts[0], "x": x, "y": y, "z": "" if z == "-" else z, "quote": text[:20]})
        return {"situations": [{"description": f"situation in {locator}", "kind": "actual", "contrast_with": -1,
                                "levels": levels}], "model_claims": claims}

    def answer(self, req):
        """A perfect stand-in for the subject (the planted model); the generic adult knows nothing."""
        self.calls["answer"] += 1
        if req["mode"] != "subject":
            return []
        known = {k["variable"]: k["level"] for k in req["situation"]["known"]}
        target = req["situation"].get("target")
        out = []
        state = dict(known)
        for v in ORDER:
            if v in state or v == target:
                continue
            ps = PARENTS.get(v)
            if ps and all(p in state for p, _ in ps):
                state[v] = _level(sum(1.3 * s * state[p] for p, s in ps))
        for a in req["ask"]:
            if a["id"] in state and a["id"] not in known:
                out.append({"variable": a["id"], "known": True, "level": state[a["id"]], "basis": "belief",
                            "cite": f"{a['id']} follows its causes", "reason": "planted"})
        return out


def sample_dag(n: int, seed: int = 1) -> list[dict[str, int]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        st: dict[str, int] = {}
        for v in ORDER:
            ps = PARENTS.get(v, [])
            st[v] = _level(sum(1.3 * s * st[p] for p, s in ps) + rng.gauss(0, 0.8))
        rows.append(st)
    return rows


def write_corpus(folder: Path, n: int = 800, seed: int = 1, mention: float = 0.75) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + 1)
    trng = random.Random(seed + 99)  # its own stream, so the topic words leave the rest of the corpus as it was
    for i, row in enumerate(sample_dag(n, seed)):
        tags = [f"[v {k} {v} state]" for k, v in row.items() if rng.random() < mention]
        if i % 50 == 0:
            tags.append("[c gamma delta 1 explicit]")
        if i % 70 == 0:
            tags.append("[c alpha zeta 1 explicit]")  # a claim the data does not support
        if i == 5:
            tags.append("[m whether_or_not zeta alpha -]")  # "whether zeta or not, alpha": true
            tags.append("[m not_x_but_y alpha gamma delta]")  # "not alpha but gamma moves delta": true
        if i == 3:
            tags += ["[s justice is_a virtue]", "[s virtue is_a good]", "[s self values good]",
                     "[s cruelty is_a vice]", "[s vice opposes virtue]", "[n justice is_a good]",
                     "[s pleasure is_a good asserts epicurus]", "[s reason requires attention]",
                     "[s virtue requires reason]"]
        words = " ".join(trng.sample(TOPIC_WORDS[i % 3].split(), 3))  # what the passage is about, unrelated to the levels
        (folder / f"{i:05d}.txt").write_text(f"# Doc {i}\ngroup: {i // 40}\npassage {i} {words} " + " ".join(tags), encoding="utf-8")
