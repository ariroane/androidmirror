"""The planted-model benchmark: can a method recover a mental model we wrote down ourselves?

A known model of a Stoic-style subject (12 variables, mediations, colliders,
and a constitutive pair that should stay out of the DAG) generates situations.
Then the corpus process distorts them the way real writing does:

- selection: troubling situations (high distress or anger) are more likely to
  be written about at all;
- salience: each written situation is about one of a few topics, and the
  topic decides which 2 to 5 variables the text mentions.

Two null corpora have no causation at all: in one, topic drives only which
variables are mentioned; in the other it also shifts their levels (a mood that
colours a whole topic), which is a real hidden common cause.

Methods compared (offline, no Claude):
- passage table: Prototype 2's first design, every unmentioned variable coded
  0 ("not engaged");
- situations, stated: the current design, unmentioned variables unknown and
  each test using only the rows that state its variables;
- + topic: the same, holding the topic fixed in the tests (a first-tier
  background variable), either grouped from what each passage mentions, as
  the build does without Claude, or the exact topic (a ceiling);
- + stated claims: the same (with the grouped topic), with the subject stating half of its model's
  arrows outright ("from this comes...") and one claim that is false, applied
  as background knowledge;
- + vignettes: the same, plus vignettes from a perfect stand-in for the subject
  (the planted model itself), so this is an upper bound on what vignettes add.

With `--claude`, Claude writes a passage for each situation in the subject's
voice and the real pipeline (coding, codebook, situations, discovery) runs on
that corpus, which also measures how faithfully the situations pass reads the
levels Claude was told to write.
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable

from . import topics as tp
from .causal.pc import pc

VARIABLES = {  # in a causal order of the planted model
    "misfortune": "outward misfortune or loss",
    "wrongdoing": "other people's wrongdoing toward one",
    "reason": "the use of one's reason, the ruling faculty",
    "mortality": "keeping one's death in mind",
    "desire_for_fame": "desire for fame and praise",
    "judgment_of_harm": "the judgment that one has been harmed",
    "anger": "anger",
    "wrath": "wrath (the same passion as anger under another name)",
    "distress": "distress or disturbance of the soul",
    "tranquillity": "tranquillity of mind",
    "cooperation": "working together with others",
    "gratitude": "gratitude",
}
# (cause, effect, sign): the planted model
EDGES = [
    ("misfortune", "judgment_of_harm", 1), ("wrongdoing", "judgment_of_harm", 1), ("reason", "judgment_of_harm", -1),
    ("judgment_of_harm", "anger", 1), ("judgment_of_harm", "distress", 1), ("anger", "cooperation", -1),
    ("distress", "tranquillity", -1), ("mortality", "desire_for_fame", -1), ("mortality", "tranquillity", 1),
    ("desire_for_fame", "distress", 1),
]
CONSTITUTIVE = [("anger", "wrath")]  # one idea under two names: should be dropped, not drawn
TOPICS = {  # which variables a topic makes salient (weights)
    "others": {"wrongdoing": 5, "anger": 4, "wrath": 2, "cooperation": 4, "judgment_of_harm": 3, "gratitude": 2},
    "fortune": {"misfortune": 5, "judgment_of_harm": 3, "distress": 4, "reason": 3, "tranquillity": 2},
    "death": {"mortality": 5, "desire_for_fame": 4, "tranquillity": 4, "distress": 2},
    "self": {"reason": 5, "tranquillity": 3, "gratitude": 3, "judgment_of_harm": 2, "distress": 2},
}
ORDER = list(VARIABLES)


def _level(score: float, rng: random.Random, noise: float = 0.8) -> int:
    z = score + rng.gauss(0, noise)
    return 1 if z > 0.6 else -1 if z < -0.6 else 0


def sample_states(n: int, rng: random.Random, edges=EDGES, topic_shift: dict[str, float] | None = None,
                  topic: str | None = None) -> list[dict[str, int]]:
    parents: dict[str, list[tuple[str, int]]] = {v: [] for v in ORDER}
    for a, b, s in edges:
        parents[b].append((a, s))
    out = []
    for _ in range(n):
        st: dict[str, int] = {}
        for v in ORDER:  # ORDER is a topological order of EDGES
            if v == "wrath":
                st[v] = st["anger"]
                continue
            score = sum(1.3 * s * st[p] for p, s in parents[v]) + (topic_shift or {}).get(v, 0.0)
            st[v] = _level(score, rng)
        out.append(st)
    return out


def corpus(n: int, seed: int, null: str | None = None) -> list[dict[str, Any]]:
    """Written situations: (topic, full state, mentioned variables)."""
    rng = random.Random(seed)
    out = []
    topics = list(TOPICS)
    while len(out) < n:
        t = rng.choice(topics)
        if null == "levels":  # the topic colours the levels of what it touches
            shift = {v: (1.0 if i % 2 else -1.0) for i, v in enumerate(TOPICS[t])}
            st = sample_states(1, rng, edges=[], topic_shift=shift)[0]
        else:
            st = sample_states(1, rng, edges=[] if null else EDGES)[0]
        troubling = (st["distress"] > 0) + (st["anger"] > 0)
        if rng.random() > 0.35 + 0.3 * troubling:  # selection: troubling situations get written about
            continue
        weights = {v: TOPICS[t].get(v, 0.3) for v in ORDER}  # off-topic ideas come up now and then
        k = rng.randint(2, 5)
        mentioned: list[str] = []
        while len(mentioned) < k:
            left = [v for v in ORDER if v not in mentioned]
            mentioned.append(rng.choices(left, [weights[v] for v in left])[0])
        out.append({"topic": t, "state": st, "mentioned": mentioned})
    return out


def tables(written: list[dict[str, Any]]) -> dict[str, dict[str, list]]:
    passage = {v: [] for v in ORDER}
    situ = {v: [] for v in ORDER}
    for w in written:
        for v in ORDER:
            on = v in w["mentioned"]
            passage[v].append(w["state"][v] if on and w["state"][v] != 0 else 0)
            situ[v].append(w["state"][v] if on else None)
    return {"passage table": passage, "situations, stated": situ}


def score(found: dict[str, Any], truth: list[tuple[str, str, int]] | None, variables: list[str]) -> dict[str, Any]:
    g = found["cpdag"]
    adj = {frozenset(e) for e in g.undirected} | {frozenset(d) for d in g.directed}
    out = {"edges": len(adj), "compelled": len(g.directed)}
    if truth is None:
        return out
    t_adj = {frozenset((a, b)) for a, b, _ in truth if a in variables and b in variables}
    t_dir = {(a, b) for a, b, _ in truth}
    tp = len(adj & t_adj)
    out.update({"skeleton_precision": round(tp / len(adj), 3) if adj else None,
                "skeleton_recall": round(tp / len(t_adj), 3) if t_adj else None,
                "arrows_right": sum(1 for d in g.directed if d in t_dir),
                "arrows_wrong": sum(1 for d in g.directed if d not in t_dir)})
    return out


def grouped_topics(written: list[dict[str, Any]], seed: int = 7) -> list[str]:
    """Topics the way the build finds them without Claude: passages grouped by what they mention."""
    found = tp.cluster([tp.features("", w["mentioned"]) for w in written], len(TOPICS), seed)
    return [f"t{lab + 1}" for lab in found["labels"]]


def topic_purity(found: list[str], written: list[dict[str, Any]]) -> float:
    """The share of passages whose grouped topic's majority true topic is their own."""
    from collections import Counter
    right = sum(Counter(w["topic"] for f2, w in zip(found, written) if f2 == f).most_common(1)[0][1] for f in set(found))
    return round(right / len(written), 3)


def _situations(written: list[dict[str, Any]], topics: list[str] | None = None) -> list[dict[str, Any]]:
    return [{"id": f"s{i}", "passage": f"{i:05d}", "locator": f"Note {i + 1}", "order": i, "origin": "corpus",
             "kind": "actual", "description": w["topic"], "contrast": None, "topic": topics[i] if topics else None,
             "cells": {v: {"level": w["state"][v], "tier": "stated", "quote": ""} for v in w["mentioned"]}}
            for i, w in enumerate(written)]


def _vignettes(rng: random.Random, targets: list[str], per_level: int) -> list[dict[str, Any]]:
    """Answers from a perfect stand-in: the planted model with the target set (an intervention)."""
    out = []
    for t in targets:
        for level in (-1, 1):
            for _ in range(per_level):
                st = _do(rng, t, level)
                out.append({"id": f"v{len(out)}", "origin": "vignette", "target": t, "kind": "vignette", "passage": None,
                            "locator": "vignette", "order": 10**6 + len(out), "description": "", "contrast": None,
                            "cells": {v: {"level": st[v], "tier": "set" if v == t else "implied"} for v in ORDER}})
    return out


def _do(rng: random.Random, target: str, level: int) -> dict[str, int]:
    parents: dict[str, list[tuple[str, int]]] = {v: [] for v in ORDER}
    for a, b, s in EDGES:
        parents[b].append((a, s))
    st: dict[str, int] = {}
    for v in ORDER:
        if v == target:
            st[v] = level
        elif v == "wrath":
            st[v] = st["anger"]
        else:
            st[v] = _level(sum(1.3 * s * st[p] for p, s in parents[v]), rng)
    return st


def _score_model(model, truth, variables) -> dict[str, Any]:
    model = model.copy()
    model.undirected = {e for e in model.undirected if e <= set(variables)}  # the topic's edges are background
    model.directed = {d for d in model.directed if set(d) <= set(variables)}
    adj = {frozenset(e) for e in model.undirected} | {frozenset(d) for d in model.directed}
    t_adj = {frozenset((a, b)) for a, b, _ in truth if a in variables and b in variables} if truth else set()
    t_dir = {(a, b) for a, b, _ in truth} if truth else set()
    out = {"edges": len(adj), "compelled": len(model.directed)}
    if truth:
        tp = len(adj & t_adj)
        out.update({"skeleton_precision": round(tp / len(adj), 3) if adj else None,
                    "skeleton_recall": round(tp / len(t_adj), 3) if t_adj else None,
                    "arrows_right": sum(1 for d in model.directed if d in t_dir),
                    "arrows_wrong": sum(1 for d in model.directed if d not in t_dir)})
    return out


def run_offline(n: int = 600, seeds: int = 3, alpha: float = 0.05, max_depth: int = 2, targets: int = 6,
                per_level: int = 8) -> dict[str, Any]:
    """Each method on the planted corpus and on both null corpora, averaged over seeds."""
    from .causal import discover
    from .causal.pdag import PDAG

    variables = [v for v in ORDER if v != "wrath"]  # the constitutive twin is left out, as selection would
    concepts = {v: {"label": v, "variable": True, "kind": "state"} for v in ORDER}
    cfg = {"alpha": alpha, "max_depth": max_depth, "bootstrap": 0, "compare_views": False, "view": "implied", "topic": True,
           "selection": {"min_support": 5, "max_vars": 25}, "variables": variables}
    report: dict[str, Any] = {"n": n, "seeds": seeds, "variables": len(variables), "results": {},
                              "vignettes": f"{targets} variables set low and high, {per_level} vignettes each"}
    for case, null in (("planted model", None), ("null: topic picks what is mentioned", "mentions"),
                       ("null: topic also shifts levels", "levels")):
        truth = None if null else EDGES
        for seed in range(seeds):
            rng = random.Random(500 + seed)
            written = corpus(n, 100 + seed, null)
            results = {}
            for method, data in tables(written).items():
                results[method] = score(pc({v: data[v] for v in variables}, variables, alpha, max_depth), truth, variables)
            grouped = grouped_topics(written)
            for method, labels in (("+ topic (grouped)", grouped), ("+ topic (exact)", [w["topic"] for w in written])):
                r = discover(_situations(written, labels), variables, concepts, [], [], {**cfg, "vignettes": False})
                results[method] = _score_model(PDAG.from_dict(r["cpdag"]), truth, variables) if "cpdag" in r else {}
                if method == "+ topic (grouped)":
                    results[method]["topic_purity"] = topic_purity(grouped, written)
            sits = _situations(written, grouped)
            claimed = rng.sample(EDGES, len(EDGES) // 2) + [("tranquillity", "reason", 1)]  # one false claim
            stated_model = [{"kind": "causes", "x": a, "y": b, "z": "", "quote": "", "passage": "x", "locator": "x"}
                            for a, b, _ in claimed]
            r = discover(sits, variables, concepts, [], stated_model, {**cfg, "vignettes": False})
            results["+ stated claims"] = _score_model(PDAG.from_dict(r["model"]), truth, variables) if "model" in r else {}
            vtargets = rng.sample(variables, min(targets, len(variables)))
            r = discover(sits + (_vignettes(rng, vtargets, per_level) if not null else []), variables, concepts, [],
                         stated_model, cfg)
            vm = r.get("vignettes", {}).get("model") or r.get("model")
            results["+ vignettes"] = _score_model(PDAG.from_dict(vm), truth, variables) if vm else {}
            for method, s in results.items():
                acc = report["results"].setdefault(case, {}).setdefault(method, {})
                for k, v in s.items():
                    if v is not None:
                        acc.setdefault(k, []).append(v)
    for case in report["results"].values():
        for method, acc in case.items():
            case[method] = {k: round(sum(v) / len(v), 2) for k, v in acc.items()}
    return report


def print_report(report: dict[str, Any], out: Callable[[str], None] = print) -> None:
    out(f"Planted-model benchmark: {report['n']} written situations, {report['variables']} variables, "
        f"{report['seeds']} seeds (averages)")
    for case, methods in report["results"].items():
        out(f"\n{case}")
        for method, s in methods.items():
            out(f"  {method:22s} " + ", ".join(f"{k} {v}" for k, v in s.items()))
    out("\nA valid method finds the planted edges with few wrong arrows, and almost nothing in either null corpus.")


# ---------- with Claude writing the corpus ----------

WRITE_SYSTEM = """You write one short passage (60 to 140 words) of a private notebook by a Stoic writer, addressed to himself. The passage reasons about one situation. It must convey exactly the levels you are given for the listed ideas (high, typical or low), in your own words, and mention no other idea from this list: {all_ideas}. You may phrase it as a reflection, a rule, or a conditional ("if ... then ..."). Write only the passage."""


def run_with_claude(data_dir, n: int = 150, seed: int = 101, writer: Callable[[dict[str, Any]], str] | None = None,
                    coder: Any = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Claude writes the corpus from planted situations; the real pipeline reads it back; we score both steps."""
    from . import codebook as cb
    from .build import build, mirror_dir, _read
    from .corpus import CorpusSpec

    if writer is None:
        from .. import llm

        def writer(w):  # noqa: F811
            ideas = ", ".join(VARIABLES[v] for v in ORDER)
            spec = "; ".join(f"{VARIABLES[v]}: {['low', 'typical', 'high'][w['state'][v] + 1]}" for v in w["mentioned"])
            client = llm._client()
            r = client.messages.create(model=llm.DEFAULT_MODEL, max_tokens=600,
                                       system=WRITE_SYSTEM.replace("{all_ideas}", ideas),
                                       messages=[{"role": "user", "content": f"Topic: {w['topic']}. Ideas and levels: {spec}."}])
            return next(b.text for b in r.content if b.type == "text").strip()

    written = corpus(n, seed)
    folder = data_dir / "corpus" / "benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    log(f"writing {len(written)} passages (1 Claude call each), then building (about 2 more each)")
    for i, w in enumerate(written):
        path = folder / f"{i:05d}.txt"
        if not path.exists():
            path.write_text(f"# Note {i + 1}\ngroup: {w['topic']}\n{writer(w)}", encoding="utf-8")
    (folder / "truth.json").write_text(json.dumps(written), encoding="utf-8")
    spec = CorpusSpec(id="benchmark", title="Planted-model benchmark", subject="the writer", splitter="documents",
                      dir="corpus/benchmark")
    mirror = build(spec, data_dir, coder=coder, log=log)
    book = _read(mirror_dir(data_dir, "benchmark") / "codebook.json")
    to_planted = {}
    for v, gloss in VARIABLES.items():
        for label in (v.replace("_", " "), gloss):
            hit = cb.resolve(book, label)
            if hit:
                to_planted[hit[0]] = v
                break
    # reading fidelity: did the situations pass read the levels Claude was told to write?
    right = wrong = 0
    for s in mirror["situations"]:
        if s["origin"] != "corpus":
            continue
        i = int(s["passage"]) if s["passage"].isdigit() else None
        w = written[i] if i is not None and i < len(written) else None
        if not w:
            continue
        for cid, c in s["cells"].items():
            v = to_planted.get(cid)
            if v and c["tier"] == "stated" and v in w["mentioned"]:
                right += c["level"] == w["state"][v]
                wrong += c["level"] != w["state"][v]
    c = mirror["causal"]
    model_edges = [(a, b) for a, b, _ in c.get("cpdag", {}).get("directed", [])]
    mapped = [to_planted.get(v) for v in c.get("variables", [])]
    truth = [(a, b, s) for a, b, s in EDGES]
    adj = {frozenset((to_planted.get(a), to_planted.get(b))) for a, b in c.get("cpdag", {}).get("undirected", [])}
    adj |= {frozenset((to_planted.get(a), to_planted.get(b))) for a, b in model_edges}
    t_adj = {frozenset((a, b)) for a, b, _ in truth if a in mapped and b in mapped}
    result = {"passages": len(written), "mapped_variables": to_planted,
              "reading_accuracy": round(right / (right + wrong), 3) if right + wrong else None,
              "edges": len(adj), "skeleton_precision": round(len(adj & t_adj) / len(adj), 3) if adj else None,
              "skeleton_recall": round(len(adj & t_adj) / len(t_adj), 3) if t_adj else None,
              "arrows_right": sum(1 for a, b in model_edges if (to_planted.get(a), to_planted.get(b)) in {(x, y) for x, y, _ in truth}),
              "arrows_wrong": sum(1 for a, b in model_edges if (to_planted.get(a), to_planted.get(b)) not in {(x, y) for x, y, _ in truth})}
    log(json.dumps(result, indent=1))
    return result
