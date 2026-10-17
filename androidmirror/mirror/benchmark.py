"""The planted-subject benchmark: can the fit recover a causal model we wrote down ourselves?

A known model of a Stoic-style subject (12 variables, chains, a fork, signs
that a modern reader would not guess) decides what the subject believes. The
subject then writes the way people write, not the way data is sampled:

- doctrines: every arrow of the model, some longer-range claims ("x, in the
  end, brings y"), thought experiments, "not x but y does z" and denials
  ("x makes no difference to y") are the things the subject could say;
- repetition: a few doctrines are said over and over, most once or never
  (Zipf), and each passage says one to three of them;
- silence: the doctrines never drawn are simply not in the corpus;
- other voices: some passages report a view the subject does not hold;
- reading errors: with probability `noise`, a judgment is misread (direction
  reversed, sign flipped, an end swapped for another concept, or a reported
  view taken for the subject's own).

Offline, the judgments are generated directly, so this measures the fit, not
the reading. With `--claude`, Claude writes each passage in the subject's voice
from the doctrines it must express and the real pipeline reads it back, which
also measures how faithfully the judgments pass reads them.

Scores: of the arrows the fit draws, how many are true arrows, and how many at
least point from a true cause to its effect with the right sign; of all the
true cause-effect pairs, how many the model entails; and the same for the naive
reading that makes every claim an arrow.
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable

from .causal import Graph, fit

VARIABLES = {  # in a causal order of the planted model
    "misfortune": "outward misfortune or loss",
    "wrongdoing": "other people's wrongdoing toward one",
    "reason": "the use of one's reason, the ruling faculty",
    "mortality": "keeping one's death in mind",
    "desire_for_fame": "desire for fame and praise",
    "judgment_of_harm": "the judgment that one has been harmed",
    "anger": "anger",
    "distress": "distress or disturbance of the soul",
    "tranquillity": "tranquillity of mind",
    "cooperation": "working together with others",
    "gratitude": "gratitude",
    "pain": "bodily pain",
}
# (cause, effect, sign): the planted model
EDGES = [
    ("misfortune", "judgment_of_harm", 1), ("wrongdoing", "judgment_of_harm", 1), ("reason", "judgment_of_harm", -1),
    ("judgment_of_harm", "anger", 1), ("judgment_of_harm", "distress", 1), ("anger", "cooperation", -1),
    ("distress", "tranquillity", -1), ("mortality", "desire_for_fame", -1), ("mortality", "tranquillity", 1),
    ("desire_for_fame", "distress", 1), ("reason", "gratitude", 1), ("gratitude", "cooperation", 1),
]
NAMES = list(VARIABLES)


def ancestors(edges=EDGES) -> dict[tuple[str, str], set[int]]:
    """Every (cause, effect) pair of the model with the signs of its paths."""
    g = Graph()
    for a, b, s in edges:
        g.add(a, b, {"sign": s, "constraint": f"{a}>{b}"})
    out = {}
    for a in NAMES:
        for b in NAMES:
            signs = {s for _, s in g.signed_paths(a, b)}
            if signs:
                out[(a, b)] = signs
    return out


def doctrines(rng: random.Random, edges=EDGES) -> list[dict[str, Any]]:
    """Everything the subject could say, true to the planted model."""
    anc = ancestors(edges)
    adj = {(a, b) for a, b, _ in edges}
    out = [{"kind": "causes", "x": a, "y": b, "z": "", "sign": s, "scope": "direct"} for a, b, s in edges]
    for (a, b), signs in anc.items():
        if (a, b) in adj or len(signs) != 1:
            continue
        s = next(iter(signs))
        out.append({"kind": rng.choice(["causes", "intervention", "means_end" if s > 0 else "causes"]), "x": a, "y": b,
                    "z": "", "sign": s, "scope": "overall"})
    for a, m, _ in edges:  # "not a but m does z": every path from a to z runs through m
        for m2, z, s in edges:
            if m2 != m or (a, z) in adj:
                continue
            g = Graph()
            for x, y, sg in edges:
                if m not in (x, y):
                    g.add(x, y, {"sign": sg, "constraint": ""})
            if not g.reaches(a, z):
                out.append({"kind": "screens_off", "x": a, "y": m, "z": z, "sign": s, "scope": "unsaid"})
    non = [(a, b) for a in NAMES for b in NAMES if a != b and (a, b) not in anc]
    for a, b in rng.sample(non, min(12, len(non))):
        out.append({"kind": rng.choice(["independent", "intervention"]), "x": a, "y": b, "z": "", "sign": 0, "scope": "unsaid"})
    rng.shuffle(out)
    return out


def misread(j: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    j = dict(j)
    r = rng.random()
    if r < 0.3 and j["kind"] != "screens_off":
        j["x"], j["y"] = j["y"], j["x"]
    elif r < 0.55 and j["sign"]:
        j["sign"] = -j["sign"]
    elif r < 0.85:
        k = rng.choice(["x", "y"])
        j[k] = rng.choice([n for n in NAMES if n not in (j["x"], j["y"], j["z"])])
    else:
        j["held"] = not j["held"]
    return j


def corpus(n: int, seed: int, noise: float = 0.0, zipf: float = 1.1, other_voices: float = 0.1) -> list[dict[str, Any]]:
    """Judgments as a reader would return them: one record per judgment per passage."""
    rng = random.Random(seed)
    pool = doctrines(rng)
    weights = [1 / (i + 1) ** zipf for i in range(len(pool))]
    out = []
    for p in range(n):
        pid = f"{p:05d}"
        said = []
        for d in rng.choices(pool, weights, k=rng.randint(1, 3)):
            said.append({**d, "held": True})
        if rng.random() < other_voices:  # a view the subject reports and does not hold: a random claim
            a, b = rng.sample(NAMES, 2)
            said.append({"kind": "causes", "x": a, "y": b, "z": "", "sign": rng.choice([-1, 1]), "scope": "unsaid",
                         "held": False})
        for j in said:
            if noise and rng.random() < noise:
                j = misread(j, rng)
            out.append({**j, "passage": pid, "locator": f"Note {p + 1}", "quote": "", "register": "describes",
                        "attribution": "self" if j["held"] else "someone else"})
    return out


def naive(judgments: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    """Every held positive claim as an arrow, whatever else is said."""
    out = {}
    for j in judgments:
        if not j["held"]:
            continue
        if j["kind"] == "screens_off":
            out[(j["y"], j["z"])] = j["sign"]
        elif j["sign"] != 0 and j["kind"] != "independent":
            out[(j["x"], j["y"])] = j["sign"]
    return [(a, b, s) for (a, b), s in out.items()]


def score(arrows: list[tuple[str, str, int]], edges=EDGES) -> dict[str, Any]:
    anc = ancestors(edges)
    true = {(a, b): s for a, b, s in edges}
    g = Graph()
    for a, b, s in arrows:
        g.add(a, b, {"sign": s, "constraint": ""})
    entailed = sum(1 for (a, b), signs in anc.items() if signs & {s for _, s in g.signed_paths(a, b)})
    cycles = sum(1 for a, b, _ in arrows if g.reaches(b, a))
    n = len(arrows) or 1
    return {"arrows": len(arrows),
            "true_arrows": round(sum(1 for a, b, s in arrows if true.get((a, b)) == s) / n, 3),
            "cause_to_effect": round(sum(1 for a, b, s in arrows if s in anc.get((a, b), set())) / n, 3),
            "reversed": round(sum(1 for a, b, _ in arrows if (b, a) in anc) / n, 3),
            "recall_of_effects": round(entailed / len(anc), 3), "in_cycles": cycles}


def run_offline(n: int = 300, seeds: int = 5, noises: tuple[float, ...] = (0.0, 0.1, 0.2)) -> dict[str, Any]:
    report: dict[str, Any] = {"n": n, "seeds": seeds, "variables": len(VARIABLES), "edges": len(EDGES),
                              "effects": len(ancestors()), "results": {}}
    for noise in noises:
        rows: dict[str, list[dict[str, Any]]] = {"fit": [], "every claim an arrow": []}
        for seed in range(seeds):
            js = corpus(n, 100 + seed, noise)
            model = fit(js)
            rows["fit"].append({**score([(e["src"], e["dst"], e["sign"]) for e in model["edges"]]),
                                "rejected": model["stats"]["rejected"], "open": model["stats"]["open"]})
            rows["every claim an arrow"].append(score(naive(js)))
        report["results"][f"reading errors {noise:.0%}"] = {
            m: {k: round(sum(r[k] for r in rs) / len(rs), 3) for k in rs[0]} for m, rs in rows.items()}
    return report


def print_report(report: dict[str, Any], out: Callable[[str], None] = print) -> None:
    out(f"Planted-subject benchmark: {report['n']} passages, {report['variables']} variables, {report['edges']} arrows, "
        f"{report['effects']} cause-effect pairs, {report['seeds']} seeds (averages)")
    for case, methods in report["results"].items():
        out(f"\n{case}")
        for method, s in methods.items():
            out(f"  {method:22s} " + ", ".join(f"{k} {v}" for k, v in s.items()))
    out("\ntrue_arrows: share of drawn arrows that are arrows of the planted model; cause_to_effect: share that at least "
        "run from a true cause to its effect with the right sign; recall_of_effects: share of the model's cause-effect "
        "pairs the drawn graph entails.")


# ---------- with Claude writing the corpus ----------

WRITE_SYSTEM = """You write one short passage (60 to 160 words) of a private notebook by a Stoic writer, addressed to himself. The passage must make exactly the causal judgments you are given, in your own words and in the writer's voice (an exhortation, a maxim, a remembered scene, a thought experiment, a rhetorical question), and make no other causal judgment about these ideas: {all_ideas}. A judgment marked "reported" is someone else's view that the writer mentions without holding. Write only the passage."""


def describe(j: dict[str, Any]) -> str:
    x, y, z = (VARIABLES.get(j[k], j[k]) for k in ("x", "y", "z"))
    verb = "raises" if j["sign"] > 0 else "lowers"
    text = {"causes": f"{x} {verb} {y}" + (" directly" if j.get("scope") == "direct" else ""),
            "intervention": f"a thought experiment: take away {x} and {y} goes with it" if j["sign"] > 0 else
            f"a thought experiment: take away {x} and {y} grows" if j["sign"] < 0 else
            f"a thought experiment: take away {x} and {y} stays as it was",
            "means_end": f"{x} is the way to {y}", "screens_off": f"it is not {x} but {y} that brings about {z}",
            "independent": f"{x} makes no difference to {y}"}[j["kind"]]
    return text + ("" if j["held"] else " (reported: someone else's view, not the writer's)")


def run_with_claude(data_dir, n: int = 120, seed: int = 101, writer: Callable[[list[dict[str, Any]]], str] | None = None,
                    coder: Any = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Claude writes the corpus from planted judgments; the real pipeline reads it back; both steps are scored."""
    from . import codebook as cb
    from .build import _read, build, mirror_dir
    from .corpus import CorpusSpec

    if writer is None:
        from .. import llm

        def writer(said):  # noqa: F811
            ideas = ", ".join(VARIABLES.values())
            client = llm._client()
            r = client.messages.create(model=llm.DEFAULT_MODEL, max_tokens=700,
                                       system=WRITE_SYSTEM.replace("{all_ideas}", ideas),
                                       messages=[{"role": "user", "content": "Judgments:\n" + "\n".join(f"- {describe(j)}" for j in said)}])
            return next(b.text for b in r.content if b.type == "text").strip()

    planted = corpus(n, seed)
    by_passage: dict[str, list[dict[str, Any]]] = {}
    for j in planted:
        by_passage.setdefault(j["passage"], []).append(j)
    folder = data_dir / "corpus" / "benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    log(f"writing {len(by_passage)} passages (1 Claude call each), then building (about 2 more each)")
    for i, (pid, said) in enumerate(sorted(by_passage.items())):
        path = folder / f"{pid}.txt"
        if not path.exists():
            path.write_text(f"# Note {i + 1}\ngroup: {i // 40}\n{writer(said)}", encoding="utf-8")
    (folder / "truth.json").write_text(json.dumps(planted), encoding="utf-8")
    spec = CorpusSpec(id="benchmark", title="Planted-subject benchmark", subject="the writer", splitter="documents",
                      dir="corpus/benchmark")
    mirror = build(spec, data_dir, coder=coder, log=log)
    book = _read(mirror_dir(data_dir, "benchmark") / "codebook.json")
    to_planted = {}
    for v, gloss in VARIABLES.items():
        for label in (v.replace("_", " "), gloss):
            hit = cb.resolve(book, label)
            if hit and hit[0] not in to_planted:
                to_planted[hit[0]] = v
                break
    read = [{**j, **{k: to_planted.get(j[k], j[k]) for k in ("x", "y", "z") if j[k]}} for j in mirror["judgments"]]

    def key(j):
        neg = j["kind"] == "independent" or (j["kind"] == "intervention" and j["sign"] == 0)
        return (j["passage"], "no" if neg else "yes", j["x"], j["y"], j["z"], 0 if neg else j["sign"], j["held"])

    truth = {key(j) for j in planted}
    got = {key(j) for j in read}
    model = fit(read)
    arrows = [(e["src"], e["dst"], e["sign"]) for e in model["edges"] if e["src"] in VARIABLES and e["dst"] in VARIABLES]
    result = {"passages": len(by_passage), "mapped": to_planted,
              "reading_precision": round(len(truth & got) / len(got), 3) if got else None,
              "reading_recall": round(len(truth & got) / len(truth), 3) if truth else None,
              "fit": score(arrows), "fit_on_perfect_reading": score([(e["src"], e["dst"], e["sign"]) for e in fit(planted)["edges"]])}
    log(json.dumps(result, indent=1))
    return result
