"""An offline stand-in for Claude, and a synthetic corpus whose subject holds a known causal model.

Passages carry their own coding as tags, e.g. "[v anger -1] [s justice is_a virtue] [j causes gamma delta - 1 direct]",
so the build pipeline can run end to end without a key.

The planted model: alpha -> gamma <- beta, gamma -> delta -> epsilon (negative); zeta touches nothing.
The subject says so in these ways, some many times over (repetition must not add weight):
- gamma -> delta, directly, in 30 passages;
- alpha -> gamma as a thought experiment ("take away alpha and gamma goes"), in 8;
- beta -> gamma in 5; delta lowers epsilon in 6;
- "not alpha but gamma moves delta" (screens_off) in 3; alpha -> delta overall in 4 (open: through gamma);
- "zeta makes no difference to alpha" in 4;
- one slip, epsilon -> gamma, in 1 passage (a cycle, so it is rejected and reported);
- reported or entertained views that must not count: Epicurus's beta -> zeta, a hypothesis zeta -> delta.
"""

from __future__ import annotations

import random
from pathlib import Path
import re

TAG = re.compile(r"\[(v|s|n|j|w) ([^\]]+)\]")
PLANTED = [("alpha", "gamma", 1), ("beta", "gamma", 1), ("gamma", "delta", 1), ("delta", "epsilon", -1)]
VARIABLES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


class FakeCoder:
    uses_llm = False

    def __init__(self) -> None:
        self.calls = {"code": 0, "consolidate": 0, "judgments": 0}

    def code(self, text, locator, subject):
        self.calls["code"] += 1
        out = {"concepts": [], "statements": []}
        for kind, body in TAG.findall(text):
            parts = body.split()
            if kind == "v":
                label, level = parts[0].replace("_", " "), int(parts[1])
                out["concepts"].append({"label": label, "kind": parts[2] if len(parts) > 2 else "state",
                                        "gloss": f"gloss of {label}", "level": 1 if level >= 0 else -1, "quote": text[:30]})
            elif kind in ("s", "n"):
                s, r, o = parts[:3]
                out["statements"].append({"subject": s.replace("_", " "), "relation": r, "object": o.replace("_", " "),
                                          "negated": kind == "n", "modality": parts[3] if len(parts) > 3 else "asserts",
                                          "attribution": parts[4] if len(parts) > 4 else "self", "qualifier": "",
                                          "wording": "", "quote": text[:30]})
        return out

    def consolidate(self, lines, subject):
        self.calls["consolidate"] += 1
        return []

    def judgments(self, text, locator, subject, concepts):
        """[j kind x y z sign scope register attribution]; [w concept direction]"""
        self.calls["judgments"] += 1
        out = {"judgments": [], "values": []}
        for kind, body in TAG.findall(text):
            parts = body.split()
            if kind == "j":
                k, x, y, z, sign = parts[:5]
                scope, register, attribution = (parts[5:] + ["unsaid", "describes", "self"][len(parts[5:]):])[:3]
                out["judgments"].append({"kind": k, "x": x, "y": y, "z": "" if z == "-" else z, "sign": int(sign),
                                         "scope": scope, "register": register, "attribution": attribution,
                                         "quote": text[:40]})
            elif kind == "w":
                out["values"].append({"concept": parts[0], "direction": int(parts[1]), "quote": text[:30]})
        return out


def judgment_tags(i: int) -> list[str]:
    tags = []
    if i < 30:
        tags.append("[j causes gamma delta - 1 direct]")
        if i % 3 == 0:
            tags.append("[j causes gamma delta - 1 direct]")  # said twice in one passage: still one passage
    if 30 <= i < 38:
        tags.append("[j intervention alpha gamma - 1]")
    if 38 <= i < 43:
        tags.append("[j causes beta gamma - 1 direct]")
    if 43 <= i < 49:
        tags.append("[j causes delta epsilon - -1 direct]")
    if 49 <= i < 52:
        tags.append("[j screens_off alpha gamma delta 1]")
    if 52 <= i < 56:
        tags.append("[j causes alpha delta - 1 overall]")
    if 56 <= i < 60:
        tags.append("[j independent zeta alpha - 0]")
    if i == 60:
        tags.append("[j causes epsilon gamma - 1]")  # the slip
    if i == 61:
        tags += ["[j causes beta zeta - 1 unsaid describes epicurus]", "[j causes zeta delta - 1 unsaid entertains]",
                 "[j causes alpha unicorns - 1]"]  # unicorns are not in the codebook
    if i in (62, 63):
        tags.append("[w zeta 1]")
    if i == 64:
        tags.append("[w epsilon -1]")
    return tags


def write_corpus(folder: Path, n: int = 80, seed: int = 1) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        tags = [f"[v {v} {rng.choice([-1, 1])} state]" for v in VARIABLES if rng.random() < 0.5 or i < 6]
        tags += judgment_tags(i)
        if i == 3:
            tags += ["[s justice is_a virtue]", "[s virtue is_a good]", "[s self values good]",
                     "[s cruelty is_a vice]", "[s vice opposes virtue]", "[n justice is_a good]",
                     "[s pleasure is_a good asserts epicurus]", "[s reason requires attention]",
                     "[s virtue requires reason]"]
        (folder / f"{i:05d}.txt").write_text(f"# Doc {i}\ngroup: {i // 40}\npassage {i} " + " ".join(tags), encoding="utf-8")
