"""Does merging through WordNet anchors beat merging by shared names? Two tests, each comparing the two alignments.

- `compare`: two built Mirrors (Marcus + Epictetus, say). Both alignments use the same pairs from you and,
  with --claude, the same pairs from Claude; the anchored one adds same-sense pairs, blocks homonyms and adds
  scaffold links. Nothing is written into data/mirror/, and anchors are not approved by it: `--include-proposed`
  reads anchors still waiting for approval as a what-if.
- `split_half`: one Mirror cut in two by alternate passages, with a share of the second half's concepts renamed
  to WordNet synonyms and a few of its concepts given the name of a different concept of the first half
  (planted homonyms). The right merge is known: every concept with its own twin. Both halves are anchored
  afresh by the same method (lexical, or Claude choosing among listed senses with --claude) and those anchors
  are approved automatically, inside the test folder only (data/evaluations/), never in a Mirror of yours.

Measures, for each alignment:

- concepts connected: concepts that sit in a part of the combined graph (held facts plus scaffold links;
  the subject and the evaluations left out) that has concepts from both Mirrors; and the largest part's size;
- homonyms caught: same-name pairs the anchors block (in the split-half test, planted ones blocked);
- held-out fitness: the causal model's held-out check on the combined Mirror (share of held-out claims the
  rest restates or implies) and the mean fitness of its claims (how often a reader of half the corpus keeps
  them), with the number of items fitness would trim;
- blind reading (needs Claude, --blind N): for N passages of each author where the two combined Mirrors read
  the passage differently, Claude sees the passage and the two readings in random order, unnamed, and says
  which the text supports better, or that neither.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

import networkx as nx

from .. import llm
from . import anchors as anchor_book
from . import transmutate as tm
from .causal import fit, holdout
from .schema import EVALUATIONS, HELD_MODALITIES, SELF

SHARED = {SELF, *EVALUATIONS}
AUTO = "split-half test (automatic; not a Mirror of yours)"


# ---------- measuring one combined Mirror ----------


def _side(node: dict[str, Any]) -> str:
    srcs = list((node.get("sources") or {}).keys())
    return "both" if len(srcs) > 1 else (srcs[0] if srcs else "?")


def connectivity(mirror: dict[str, Any]) -> dict[str, Any]:
    g = nx.Graph()
    for f in mirror["facts"]:
        if f["held"] and not f["neg"] and f["s"] not in SHARED and f["o"] not in SHARED:
            g.add_edge(f["s"], f["o"])
    for x in mirror.get("scaffold", []):
        g.add_edge(x["s"], x["o"])
    nodes = mirror["nodes"]
    names = mirror["meta"].get("sources", [])
    across = 0
    for comp in nx.connected_components(g):
        sides = {_side(nodes.get(n, {})) for n in comp}
        if "both" in sides or len(sides & set(names)) == 2:
            across += len(comp)
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    merged = sum(1 for n in nodes.values() if _side(n) == "both") - len(SHARED & set(nodes))
    return {"concepts_in_graph": g.number_of_nodes(), "connected_across": across, "largest_part": comps[0] if comps else 0,
            "parts": len(comps), "merged": max(0, merged), "scaffold_links": len(mirror.get("scaffold", []))}


def fitness_stats(mirror: dict[str, Any], samples: int = 20) -> dict[str, Any]:
    settings = mirror["causal"]["settings"]
    model = fit(mirror["judgments"], settings)
    fit_ = tm.fitness(mirror, model, {"samples": samples}, log=lambda *_: None)
    cut = tm.trims(mirror, model, fit_)
    h = holdout(mirror["judgments"], settings)["positive"]
    n = sum(h.values())
    claims = [v["model"] for v in fit_["claims"].values()]
    return {"held_out_predicted": round((h.get("restated", 0) + h.get("follows", 0)) / n, 3) if n else None,
            "held_out_contradicted": round(h.get("contradicted", 0) / n, 3) if n else None,
            "claim_fitness_mean": round(sum(claims) / len(claims), 3) if claims else None,
            "arrows": model["stats"]["arrows"], "in_tension": model["stats"]["rejected"],
            "trims": len(cut), "concepts": len(mirror["nodes"])}


def run(a: dict[str, Any], b: dict[str, Any], use_anchors: bool, claude_found: list[dict[str, Any]],
        add: list[dict[str, Any]] | None = None, samples: int = 20) -> dict[str, Any]:
    al = tm.align(a, b, add=add, use_anchors=use_anchors, claude_found=claude_found)
    combo = tm.combine(a, b, al["pairs"], al["pivot"]["scaffold"])
    piv = al["pivot"]
    return {"alignment": al, "combo": combo,
            "stats": {"pairs": len(al["pairs"]), "by_how": _count(p["how"] for p in al["pairs"]),
                      "blocked": len(piv["blocked"]), "candidates": len(piv["candidates"]),
                      **connectivity(combo["mirror"]), **fitness_stats(combo["mirror"], samples)}}


def _count(xs: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# ---------- blind reading ----------


def reading(mirror: dict[str, Any], pid: str, limit: int = 30) -> str:
    """What a combined Mirror makes of one passage: its concepts (with every name they go by after merging, and
    which Mirrors they come from), the facts it reads there, and the links to the rest, scaffold included."""
    nodes = mirror["nodes"]
    here = sorted({n for n, v in nodes.items() if n not in SHARED and any(m["passage"] == pid for m in v.get("mentions", []))})
    if not here:
        return ""
    lab = lambda n: nodes.get(n, {}).get("label", n)  # noqa: E731
    lines = ["Concepts:"]
    for n in here:
        v = nodes[n]
        names = [v["label"]] + [a for a in v.get("aliases", []) if a.lower() != v["label"].lower()][:6]
        lines.append(f"- {' / '.join(names)} ({v.get('kind', '')}; from {' and '.join((v.get('sources') or {}).keys())})"
                     f": {v.get('gloss', '')}")
    said, links = [], []
    for f in mirror["facts"]:
        if not f["held"] or f["s"] in EVALUATIONS:
            continue
        t = f"{lab(f['s'])} {'not ' if f['neg'] else ''}{f['r'].replace('_', ' ')} {lab(f['o'])}"
        if any(e["passage"] == pid for e in f["evidence"]):
            said.append(t)
        elif (f["s"] in here or f["o"] in here) and not f["neg"]:
            links.append((-f["support"], t + f" ({', '.join(sorted(set(e.get('source', '') for e in f['evidence'])))})"))
    scaffold = [f"{lab(x['s'])} is a kind of {lab(x['o'])} (from WordNet, not said by either)"
                for x in mirror.get("scaffold", []) if x["s"] in here or x["o"] in here]
    lines += ["Read in this passage:"] + [f"- {t}" for t in said[:limit]]
    lines += ["Links to the rest of the combined model:"] + [f"- {t}" for _, t in sorted(links)[:limit]]
    lines += [f"- {t}" for t in scaffold[:10]]
    return "\n".join(lines)


BLIND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"choice": {"type": "string", "enum": ["1", "2", "neither"]}, "why": {"type": "string"}},
    "required": ["choice", "why"], "additionalProperties": False,
}

BLIND_SYSTEM = """Two models combine the thought of {subjects} into one. Each is shown as it reads one passage by
{author}: the concepts it finds there (with every name a concept goes by once the two authors' concepts are merged,
and which author each comes from), what it reads the passage as saying, and how those concepts link to the rest.

Which reading does the passage's text support better? Judge only by the text: a concept merged with the other
author's concept should mean what this passage means by it; links should not attribute to the author what the text
does not support; a link marked as from WordNet is a dictionary relation, fine if the text is consistent with it.
Answer "neither" if they are equally good or differ in nothing that matters. `why`: one or two sentences."""


def blind(a: dict[str, Any], b: dict[str, Any], runs: dict[str, dict[str, Any]], n: int, seed: int = 0,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    """Claude's blind choice between the current and the anchored combined Mirror, passage by passage."""
    if not llm.available():
        raise llm.LLMUnavailable("the blind reading needs ANTHROPIC_API_KEY in your local .env")
    rng = random.Random(seed)
    mc, ma = runs["current"]["combo"], runs["anchored"]["combo"]
    text = {p["id"]: p for p in mc["passages"]}
    subjects = f"{a['mirror']['meta']['corpus']['subject']} and {b['mirror']['meta']['corpus']['subject']}"
    out: dict[str, Any] = {"n": n, "authors": {}, "rows": []}
    for src in (a, b):
        author = src["mirror"]["meta"]["corpus"]["subject"]
        pids = [p["id"] for p in mc["passages"] if p.get("source") == src["id"]]
        differ = []
        for pid in pids:
            rc, ra = reading(mc["mirror"], pid), reading(ma["mirror"], pid)
            if rc and ra and rc != ra:
                differ.append((pid, rc, ra))
        pick = rng.sample(differ, min(n, len(differ)))
        tally = {"current": 0, "anchored": 0, "neither": 0}
        log(f"blind reading: {len(pick)} of {len(differ)} {src['id']} passages the two read differently")
        for pid, rc, ra in pick:
            flip = rng.random() < 0.5
            one, two = (ra, rc) if flip else (rc, ra)
            system = BLIND_SYSTEM.replace("{subjects}", subjects).replace("{author}", author)
            user = (f"Passage ({text[pid]['locator']}):\n{text[pid]['text']}\n\n"
                    f"Reading 1:\n{one}\n\nReading 2:\n{two}")
            r = llm._json_call(system, user, BLIND_SCHEMA, effort="medium", max_tokens=2000, cache_system=True)
            won = "neither" if r["choice"] == "neither" else (
                ("anchored" if flip else "current") if r["choice"] == "1" else ("current" if flip else "anchored"))
            tally[won] += 1
            out["rows"].append({"passage": pid, "author": src["id"], "winner": won, "why": r["why"]})
        out["authors"][src["id"]] = {"differing_passages": len(differ), "read": len(pick), **tally}
    return out


# ---------- test 1: two Mirrors ----------


def with_proposed(src: dict[str, Any], root: Path) -> dict[str, Any]:
    """The source with anchors still waiting for approval read as if approved (for a what-if only)."""
    book = anchor_book.load(root)
    anch = dict(anchor_book.approved(book))
    for p in book["proposals"]:
        if p["status"] == "proposed" and p["concept"] not in anch:
            anch[p["concept"]] = p
    return {**src, "anchors": anch}


def compare(data_dir: Path, a_name: str, b_name: str, use_claude: bool = False, include_proposed: bool = False,
            samples: int = 20, blind_n: int = 0, seed: int = 0, log: Callable[[str], None] = print) -> dict[str, Any]:
    a, b = tm.load_source(data_dir, a_name), tm.load_source(data_dir, b_name)
    if include_proposed:
        a, b = (with_proposed(s, data_dir / "mirror" / s["id"]) for s in (a, b))
    if not (a["anchors"] and b["anchors"]):
        raise RuntimeError("both Mirrors need anchors: python -m androidmirror anchors propose --corpus NAME "
                           "(then approve them, or evaluate with --include-proposed)")
    if use_claude and not llm.available():
        raise llm.LLMUnavailable("--claude needs ANTHROPIC_API_KEY in your local .env")
    claude = tm.claude_pairs(a, b) if use_claude else []
    runs = {}
    for mode, on in (("current", False), ("anchored", True)):
        log(f"{mode} alignment: combining and scoring")
        runs[mode] = run(a, b, on, claude, samples=samples)
    report = _report("compare", f"{a_name}+{b_name}", a, b, runs, {"claude_pairs": len(claude),
                     "include_proposed": include_proposed, "samples": samples})
    if blind_n:
        report["blind"] = blind(a, b, runs, blind_n, seed, log)
    return _save(data_dir, report)


# ---------- test 2: split-half with renamed vocabulary ----------


def _inside(pid: str, keep: set[str]) -> bool:
    return pid in keep or pid == "codebook"


def half(src: dict[str, Any], keep: set[str], name: str) -> dict[str, Any]:
    """The part of a Mirror that a set of its passages supports."""
    m = src["mirror"]
    facts = []
    for f in m["facts"]:
        ev = [e for e in f["evidence"] if _inside(e["passage"], keep)]
        if not ev or all(e["passage"] == "codebook" for e in ev):
            continue
        held = any(e.get("attribution") == SELF and e.get("modality") in HELD_MODALITIES for e in ev)
        facts.append({**f, "evidence": ev, "held": held, "support": len({e["passage"] for e in ev})})
    judgments = [j for j in m["judgments"] if j["passage"] in keep]
    values = []
    for v in m["values"]:
        ev = [e for e in v["evidence"] if e["passage"] in keep]
        if ev:
            more = sum(1 for e in ev if e["direction"] > 0)
            less = len(ev) - more
            values.append({**v, "evidence": ev, "more": more, "less": less,
                           "direction": 1 if more > less else -1 if less > more else 0})
    used = set(SHARED)
    for f in facts:
        used |= {f["s"], f["o"]}
    for j in judgments:
        used |= {j["x"], j["y"], j["z"]} - {""}
    used |= {v["concept"] for v in values}
    nodes = {}
    for nid, n in m["nodes"].items():
        mentions = [x for x in n.get("mentions", []) if x["passage"] in keep]
        if nid in used or mentions:
            nodes[nid] = {**n, "mentions": mentions, "support": len({x["passage"] for x in mentions})}
    for nid, n in list(nodes.items()):
        if n.get("pole_of") and n["pole_of"] not in nodes:
            nodes[n["pole_of"]] = {**m["nodes"][n["pole_of"]], "mentions": [], "support": 0}
    meta = {**m["meta"], "corpus": {**m["meta"]["corpus"], "id": name}}
    mirror = {**m, "meta": meta, "nodes": nodes, "facts": facts, "judgments": judgments, "values": values,
              "unresolved": [u for u in m.get("unresolved", []) if u["passage"] in keep],
              "passage_ids": [p for p in m["passage_ids"] if p in keep]}
    return {"mirror": mirror, "passages": [p for p in src["passages"] if p["id"] in keep]}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _sub_word(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE) if old and text else text


def rename(mirror: dict[str, Any], changes: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Give concepts new ids and labels everywhere ({old id: {"id", "label"}}), dropping their aliases (which would
    give the old name away) except those naming an opposite, and replacing the old label in glosses and quotes."""
    idmap = {k: v["id"] for k, v in changes.items()}

    def nid(x: str) -> str:
        if not x:
            return x
        pole = x.endswith("~")
        base = x[:-1] if pole else x
        return idmap.get(base, base) + ("~" if pole else "")

    nodes = {}
    for k, n in mirror["nodes"].items():
        rec = dict(n)
        base = k[:-1] if k.endswith("~") else k
        if base in changes:
            old = n["label"]
            if not k.endswith("~"):
                rec["label"] = changes[base]["label"]
                rec["aliases"] = [a for a in n.get("aliases", []) if a.endswith("(opposite)")]
                rec["gloss"] = _sub_word(n.get("gloss", ""), old, rec["label"])
                rec["mentions"] = [{**x, "quote": _sub_word(x.get("quote", ""), old, rec["label"])}
                                   for x in n.get("mentions", [])]
                rec["renamed_from"] = base
        if rec.get("pole_of"):
            rec["pole_of"] = nid(rec["pole_of"])
        nodes[nid(k)] = rec
    facts = [{**f, "s": nid(f["s"]), "o": nid(f["o"])} for f in mirror["facts"]]
    judgments = []
    for j in mirror["judgments"]:
        rec = {**j, "x": nid(j["x"]), "y": nid(j["y"]), "z": nid(j["z"])}
        if j["kind"] == "ruling":
            t, *rest = j["target"].split("|")
            rec["target"] = "|".join([t] + [nid(r) for r in rest])
        judgments.append(rec)
    values = [{**v, "concept": nid(v["concept"]), "evidence": [{**e, "concept": nid(e["concept"])} for e in v["evidence"]]}
              for v in mirror["values"]]
    return {**mirror, "nodes": nodes, "facts": facts, "judgments": judgments, "values": values}


def _write_mirror(root: Path, data: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, obj in (("mirror.json", data["mirror"]), ("passages.json", data["passages"])):
        (root / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def split_half(data_dir: Path, name: str, rename_share: float = 0.5, homonyms: int = 20, use_claude: bool = False,
               limit: int | None = 300, samples: int = 20, blind_n: int = 0, seed: int = 0,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    from . import wordnet

    rng = random.Random(seed)
    wn = wordnet.load(data_dir, log)
    src = tm.load_source(data_dir, name)
    order = sorted(src["passages"], key=lambda p: p.get("order", 0))
    keep1 = {p["id"] for i, p in enumerate(order) if i % 2 == 0}
    keep2 = {p["id"] for i, p in enumerate(order) if i % 2 == 1}
    n1, n2 = f"{name}-half1", f"{name}-half2"
    h1, h2 = half(src, keep1, n1), half(src, keep2, n2)
    c1, c2 = anchor_book.concepts(h1["mirror"]), anchor_book.concepts(h2["mirror"])
    # rename a share of the second half's concepts to a WordNet synonym of their label
    taken = {n["label"].lower() for n in list(c1.values()) + list(c2.values())}
    renamable = []
    for cid, n in sorted(c2.items()):
        syn = next((s for s in wn.senses(n["label"], ["n"])[:1]), None)
        if not syn:
            continue
        alt = next((w for w in wn.get(syn)["words"] if w.lower() != n["label"].lower()
                    and n["label"].lower() not in w.lower() and w.lower() not in taken and w.islower()), None)
        if alt:
            renamable.append((cid, alt))
    picked, used_alt = [], set()
    for cid, alt in rng.sample(renamable, int(len(renamable) * rename_share)):
        if alt.lower() not in used_alt:  # two concepts never get the same new name
            used_alt.add(alt.lower())
            picked.append((cid, alt))
    changes = {cid: {"id": _slug(alt) + "__r", "label": alt} for cid, alt in picked}
    # plant homonyms: a concept of the second half given the name of a different concept of the first half
    pool1 = [k for k, n in sorted(c1.items()) if len(wn.senses(n["label"], ["n"])) >= 2
             and n["label"].lower() not in {x["label"].lower() for x in c2.values()}]
    pool2 = [k for k in sorted(c2) if k not in changes and c2[k].get("support", 0) >= 1]
    planted = []
    for x, y in zip(rng.sample(pool1, min(homonyms, len(pool1))), rng.sample(pool2, min(homonyms, len(pool2)))):
        if x == y or c1[x]["label"].lower() in used_alt:
            continue
        used_alt.add(c1[x]["label"].lower())
        changes[y] = {"id": _slug(c1[x]["label"]) + "__h", "label": c1[x]["label"]}
        planted.append({"b": changes[y]["id"], "twin": y, "named_after": x, "label": c1[x]["label"]})
    h2 = {**h2, "mirror": rename(h2["mirror"], changes)}
    truth = {changes.get(k, {"id": k})["id"]: k for k in c2}  # half-2 id -> the concept it truly is
    root = data_dir / "evaluations" / f"split-half-{name}"
    for nm, data in ((n1, h1), (n2, h2)):
        r = root / nm
        _write_mirror(r, data)
        if anchor_book.path(r).exists():
            anchor_book.path(r).unlink()
        anchor_book.propose(r, data["mirror"], wn, use_claude, limit, log=log)
        anchor_book.approve(r, ["all"], AUTO)
    a, b = tm.load_root(root / n1, n1), tm.load_root(root / n2, n2)
    claude = tm.claude_pairs(a, b) if use_claude else []
    runs = {}
    for mode, on in (("current", False), ("anchored", True)):
        log(f"{mode} alignment: combining and scoring")
        runs[mode] = run(a, b, on, claude, samples=samples)
        runs[mode]["stats"].update(_truth_score(runs[mode]["alignment"], truth, c1, changes, planted))
    report = _report("split-half", f"split-half-{name}", a, b, runs,
                     {"renamed": len(picked), "renamable": len(renamable), "planted_homonyms": len(planted),
                      "twins": sum(1 for k in c2 if k in c1), "claude_pairs": len(claude), "samples": samples,
                      "seed": seed, "anchoring": "claude" if use_claude else "lexical", "limit": limit})
    report["planted"] = planted
    if blind_n:
        report["blind"] = blind(a, b, runs, blind_n, seed, log)
    return _save(data_dir, report)


def _truth_score(al: dict[str, Any], truth: dict[str, str], c1: dict[str, Any], changes: dict[str, Any],
                 planted: list[dict[str, Any]]) -> dict[str, Any]:
    renamed_ids = {v["id"] for k, v in changes.items() if not v["id"].endswith("__h")}
    planted_ids = {p["b"] for p in planted}
    twins = {b: t for b, t in truth.items() if t in c1}
    right = [p for p in al["pairs"] if twins.get(p["b"]) == p["a"] and p["polarity"] == 1]
    wrong = [p for p in al["pairs"] if not (twins.get(p["b"]) == p["a"] and p["polarity"] == 1)]
    blocked_planted = [x for x in al["pivot"]["blocked"] if x["b"] in planted_ids]
    merged_planted = [p for p in al["pairs"] if p["b"] in planted_ids and twins.get(p["b"]) != p["a"]]
    return {"true_pairs": len(twins), "right_merges": len(right), "wrong_merges": len(wrong),
            "recall": round(len(right) / len(twins), 3) if twins else None,
            "precision": round(len(right) / len(al["pairs"]), 3) if al["pairs"] else None,
            "renamed_recovered": sum(1 for p in right if p["b"] in renamed_ids),
            "renamed_with_twin": sum(1 for b in renamed_ids if b in twins),
            "planted_merged_wrongly": len(merged_planted), "planted_blocked": len(blocked_planted),
            "wrong_examples": [f"{p['a_label']} = {p['b_label']} [{p['how']}]" for p in wrong[:15]]}


# ---------- the report ----------


def _report(test: str, rid: str, a: dict[str, Any], b: dict[str, Any], runs: dict[str, Any], settings: dict[str, Any]
            ) -> dict[str, Any]:
    an = runs["anchored"]["alignment"]["pivot"]
    return {"test": test, "id": rid, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "a": a["id"], "b": b["id"],
            "anchors": {s["id"]: _count(p["predicate"] for p in s["anchors"].values()) for s in (a, b)},
            "settings": settings, "stats": {k: v["stats"] for k, v in runs.items()},
            "blocked": [{k: x.get(k) for k in ("a", "b", "a_label", "b_label", "how", "verdict", "reason")}
                        for x in an["blocked"]],
            "candidates": [{k: x.get(k) for k in ("a", "b", "a_label", "b_label", "why", "how")}
                           for x in an["candidates"][:100]],
            "scaffold": [{k: x.get(k) for k in ("s_label", "o_label", "steps", "lower", "upper")} for x in an["scaffold"][:200]],
            "new_pairs": [f"{p['a_label']} = {p['b_label']} [{p['how']}]" for p in runs["anchored"]["alignment"]["pairs"]
                          if (p["a"], p["b"]) not in {(q["a"], q["b"]) for q in runs["current"]["alignment"]["pairs"]}]}


def _save(data_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    root = data_dir / "evaluations" / report["id"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (root / "report.md").write_text(describe(report), encoding="utf-8")
    return report


ROWS = [("pairs", "concepts merged (pairs)"), ("blocked", "merges blocked as homonyms"),
        ("candidates", "closeMatch candidates (not merged)"), ("scaffold_links", "scaffold is_a links"),
        ("connected_across", "concepts connected across the two Mirrors"), ("largest_part", "largest connected part"),
        ("parts", "separate parts"), ("held_out_predicted", "held-out claims restated or implied"),
        ("held_out_contradicted", "held-out claims contradicted"), ("claim_fitness_mean", "mean claim fitness"),
        ("arrows", "causal arrows"), ("in_tension", "claims in tension"), ("trims", "items fitness would trim"),
        ("true_pairs", "true pairs (concepts in both halves)"), ("right_merges", "right merges"),
        ("wrong_merges", "wrong merges"), ("recall", "recall"), ("precision", "precision"),
        ("renamed_recovered", "renamed concepts recovered"), ("renamed_with_twin", "renamed concepts with a twin"),
        ("planted_merged_wrongly", "planted homonyms merged (wrong)"), ("planted_blocked", "planted homonyms blocked")]


def describe(r: dict[str, Any]) -> str:
    cur, anc = r["stats"]["current"], r["stats"]["anchored"]
    lines = [f"# Anchored vs current alignment: {r['a']} + {r['b']} ({r['test']})", "",
             f"Run {r['created']}. Settings: {json.dumps(r['settings'])}.",
             f"Anchors read: {json.dumps(r['anchors'])}.", "",
             "| measure | current (names) | anchored (WordNet) |", "|---|---|---|"]
    for k, label in ROWS:
        if k in cur or k in anc:
            lines.append(f"| {label} | {cur.get(k, '')} | {anc.get(k, '')} |")
    lines.append(f"| pairs by source | {cur['by_how']} | {anc['by_how']} |")
    if r.get("blind"):
        lines += ["", "## Blind reading", "", "| author | passages read differently | read | current | anchored | neither |",
                  "|---|---|---|---|---|---|"]
        for k, v in r["blind"]["authors"].items():
            lines.append(f"| {k} | {v['differing_passages']} | {v['read']} | {v['current']} | {v['anchored']} | {v['neither']} |")
    if r.get("planted"):
        lines += ["", f"## Planted homonyms ({len(r['planted'])})", ""]
        lines += [f"- {p['twin']} renamed {p['label']!r} (the name of {p['named_after']})" for p in r["planted"]]
        for k in ("current", "anchored"):
            if r["stats"][k].get("wrong_examples"):
                lines += ["", f"Wrong merges, {k}: " + "; ".join(r["stats"][k]["wrong_examples"])]
    lines += ["", f"## Blocked as homonyms ({len(r['blocked'])})", ""]
    lines += [f"- {x['a_label']} / {x['b_label']} [{x['how']}]: {x['reason']}" for x in r["blocked"]]
    lines += ["", f"## Pairs only the anchors make ({len(r['new_pairs'])})", ""] + [f"- {x}" for x in r["new_pairs"][:80]]
    lines += ["", f"## Scaffold links ({len(r['scaffold'])}, first 60)", ""]
    lines += [f"- {x['s_label']} is a kind of {x['o_label']} ({x['lower']['sense_label']} → {x['upper']['sense_label']})"
              for x in r["scaffold"][:60]]
    lines += ["", "## Candidates (first 40)", ""] + [f"- {x['a_label']} ~ {x['b_label']}: {x['why']}" for x in r["candidates"][:40]]
    return "\n".join(lines) + "\n"
