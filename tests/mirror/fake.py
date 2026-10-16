"""An offline stand-in for Claude, and a synthetic corpus with a known causal structure.

Passages carry their own coding as tags, e.g. "[v anger -1] [s justice is_a virtue] [c anger distress 1 explicit]",
so the build pipeline can run end to end without a key.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

TAG = re.compile(r"\[(v|s|c|n) ([^\]]+)\]")


class FakeCoder:
    uses_llm = False

    def __init__(self) -> None:
        self.calls = {"code": 0, "consolidate": 0, "measure": 0}

    def code(self, text, locator, subject):
        self.calls["code"] += 1
        out = {"concepts": [], "statements": [], "causal_claims": []}
        for kind, body in TAG.findall(text):
            parts = body.split()
            if kind == "v":
                label, level = parts[0].replace("_", " "), int(parts[1])
                out["concepts"].append({"label": label, "kind": parts[2] if len(parts) > 2 else "state",
                                        "gloss": f"gloss of {label}", "level": level, "quote": text[:30]})
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

    def measure(self, text, locator, subject, variables):
        self.calls["measure"] += 1
        ids = {v["id"] for v in variables}
        out = []
        for kind, body in TAG.findall(text):
            parts = body.split()
            if kind == "v" and parts[0] in ids:
                out.append({"variable": parts[0], "level": int(parts[1]), "quote": text[:20]})
        return out


def sample_dag(n: int, seed: int = 1) -> list[dict[str, int]]:
    """A -> C <- B, C -> D -> E, plus F independent. Values in {-1, 0, 1} (0 = not engaged)."""
    rng = random.Random(seed)

    def draw(p_on: float, p_pos: float) -> int:
        if rng.random() > p_on:
            return 0
        return 1 if rng.random() < p_pos else -1

    rows = []
    for _ in range(n):
        a = draw(0.4, 0.5)
        b = draw(0.4, 0.5)
        push = a + b
        c = draw(0.25 + 0.3 * abs(push), 0.5 + 0.2 * push)
        d = draw(0.2 + 0.5 * abs(c), 0.5 + 0.4 * c)
        e = draw(0.2 + 0.5 * abs(d), 0.5 - 0.4 * d)
        f = draw(0.4, 0.5)
        rows.append({"alpha": a, "beta": b, "gamma": c, "delta": d, "epsilon": e, "zeta": f})
    return rows


def write_corpus(folder: Path, n: int = 800, seed: int = 1) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(sample_dag(n, seed)):
        tags = [f"[v {k} {v} state]" for k, v in row.items() if v]
        if i % 50 == 0:
            tags.append("[c gamma delta 1 explicit]")
        if i % 70 == 0:
            tags.append("[c alpha zeta 1 explicit]")  # a claim the data does not support
        if i == 3:
            tags += ["[s justice is_a virtue]", "[s virtue is_a good]", "[s self values good]",
                     "[s cruelty is_a vice]", "[s vice opposes virtue]", "[n justice is_a good]",
                     "[s pleasure is_a good asserts epicurus]", "[s reason requires attention]",
                     "[s virtue requires reason]"]
        (folder / f"{i:05d}.txt").write_text(f"# Doc {i}\ngroup: {i // 40}\npassage {i} " + " ".join(tags), encoding="utf-8")
