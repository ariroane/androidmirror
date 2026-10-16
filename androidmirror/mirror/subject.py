"""The subject transformation: Claude standing in for the subject of the corpus.

Reading between the lines means saying how a variable stands in a situation
the text leaves it unsaid, and answering vignettes the corpus never wrote. Done
naively that is Claude's own world model talking. The transformation holds
Claude's priors in check by making it answer as the subject, from a profile
built from the corpus alone (nothing here is specific to Marcus; the same code
stands in for the FOMC or a diarist):

- vocabulary: the causal variables, with their meanings and aliases;
- beliefs: what the subject holds (the knowledge graph, stated and derived),
  its stated causal claims, and denials;
- usual levels: how each variable usually stands where the subject states it;
- voice: a few representative passages;
- attribution: whose views the subject only reports, so they are never used.

Before each answer the passages most like the situation are retrieved, and
every value must cite a passage or a belief. Three checks keep it honest:

- held-out prediction (`validate`): hide some real situations, reveal part of
  what each states and predict the rest, as the subject, as plain Claude asked
  to imitate the subject without the profile, and as an ordinary educated
  adult; only variables the transformation predicts well are filled in;
- citation knockout: drop the cited passage and ask again; an answer that does
  not change came from Claude's priors, not the subject;
- generic-subject diff: an uncited answer an ordinary adult gives too is
  common sense, not the subject.

The share of filled values that fail a check is reported as the prior-leak rate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .. import llm
from . import situations as sit
from .kg import KnowledgeBase
from .schema import EVALUATIONS, SELF

LEVEL_WORD = {-1: "low", 0: "typical", 1: "high"}

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"variable": {"type": "string"}, "known": {"type": "boolean"},
                       "level": {"type": "integer", "enum": [-1, 0, 1]},
                       "basis": {"type": "string", "enum": ["passage", "belief", "common sense", "none"]},
                       "cite": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["variable", "known", "level", "basis", "cite", "reason"], "additionalProperties": False}}},
    "required": ["answers"],
    "additionalProperties": False,
}

SUBJECT_SYSTEM = """You are standing in for {subject}, the subject of a corpus, so that {subject}'s own model of the world can be read off. Answer as {subject} would, from {subject}'s own words below, never from what you yourself believe and never from modern common sense.

For each variable you are asked about, say how it stands in the situation: low (-1), typical (0) or high (1).
1. Only the subject's vocabulary. If the value cannot be settled in the subject's own terms, answer known = false.
2. Only the subject's reasons. Every value cites a passage (basis "passage", cite its locator) or one of the subject's beliefs listed below (basis "belief", cite it). If you cannot cite either, answer known = false, or give your best reading with basis "none".
3. If the situation does not bear on a variable, use the subject's usual level where one is listed (basis "belief", cite "usual level"); otherwise known = false.
4. Views the subject only reports from others ({others}) are not the subject's views.
Keep each reason to one short line, in the subject's terms.

{profile}"""

GENERIC_SYSTEM = """You are an ordinary educated adult of today, with no particular philosophy or school. For each variable you are asked about, say how it would stand in the situation, from common sense: low (-1), typical (0) or high (1). Use basis "common sense" and leave cite empty. If common sense does not settle it, answer known = false. Keep each reason to one short line.

Variables:
{vocabulary}"""

PLAIN_SYSTEM = """Say how {subject} would see the situation. For each variable you are asked about, say how it would stand: low (-1), typical (0) or high (1), with your best reason, using basis "none" unless you can cite something specific. If you cannot tell, answer known = false. Keep each reason to one short line.

Variables:
{vocabulary}"""


# ---------- the profile ----------


def build_profile(subject: str, nodes: dict[str, Any], facts: list[dict[str, Any]], kb: KnowledgeBase,
                  claims: list[dict[str, Any]], situations: list[dict[str, Any]], passages: dict[str, dict[str, Any]],
                  variables: list[str], exclude: set[str] | frozenset = frozenset()) -> dict[str, Any]:
    """Everything the transformation knows about the subject, from the corpus alone (minus `exclude` passages)."""
    label = lambda n: nodes.get(n, {}).get("label", n)  # noqa: E731

    def usable(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [e for e in evidence if e.get("passage") not in exclude]

    vocab = [{"id": v, "label": label(v), "gloss": nodes.get(v, {}).get("gloss", ""),
              "aliases": [a for a in nodes.get(v, {}).get("aliases", []) if "(opposite)" not in a][:4]} for v in variables]
    beliefs, denials = [], []
    for f in sorted(facts, key=lambda f: -f["support"]):
        ev = usable(f["evidence"])
        if not f["held"] or not ev or f["s"] in EVALUATIONS:
            continue
        text = f"{label(f['s'])} {f['r'].replace('_', ' ')} {label(f['o'])}"
        rec = {"text": ("not: " if f["neg"] else "") + text, "cite": ev[0].get("locator", ""), "support": len(ev)}
        (denials if f["neg"] else beliefs).append(rec)
    vset = set(variables)
    derived = []
    for (s, r, o), proof in kb.closure.items():
        if proof["rule"] != "stated" and (s in vset or o in vset) and o not in EVALUATIONS and s != SELF:
            if kb.stated_support((s, r, o)) - set(exclude):
                derived.append({"text": f"{label(s)} {r.replace('_', ' ')} {label(o)}", "cite": proof["rule"]})
    causal = []
    for c in claims:
        ev = usable(c["evidence"])
        if c.get("held") and ev:
            causal.append({"text": f"{label(c['cause'])} {'raises' if c['sign'] > 0 else 'lowers'} {label(c['effect'])}",
                           "cite": ev[0]["locator"], "support": len(ev)})
    kept = [s for s in situations if s["origin"] == "corpus" and s["passage"] not in exclude]
    typical = sit.typical_levels(kept, variables)
    voice = _voice(kept, passages, exclude)
    others = Counter()
    for f in facts:
        for e in f["evidence"]:
            a = e.get("attribution", SELF)
            if a and a != SELF and e.get("passage") not in exclude:
                others[a] += 1
    return {"subject": subject, "vocabulary": vocab, "beliefs": beliefs[:60], "denials": denials[:15],
            "derived": derived[:25], "causal": sorted(causal, key=lambda c: -c["support"])[:40],
            "typical": {v: {**t, "label": label(v)} for v, t in typical.items()}, "voice": voice,
            "others": [a for a, _ in others.most_common(10)], "excluded": len(exclude)}


def _voice(kept: list[dict[str, Any]], passages: dict[str, dict[str, Any]], exclude, k: int = 5) -> list[dict[str, str]]:
    score = Counter()
    for s in kept:
        score[s["passage"]] += sum(1 for c in s["cells"].values() if c["tier"] == "stated")
    out, groups = [], set()
    for pid, _ in score.most_common():
        p = passages.get(pid)
        if not p or pid in exclude or p.get("group") in groups:
            continue
        groups.add(p.get("group"))
        out.append({"locator": p["locator"], "text": p["text"][:800]})
        if len(out) >= k:
            break
    return out


def render_profile(profile: dict[str, Any]) -> str:
    lines = ["Variables (id: label - meaning; also called):"]
    lines += [f"- {v['id']}: {v['label']} - {v['gloss']}" + (f" (also: {', '.join(v['aliases'])})" if v["aliases"] else "")
              for v in profile["vocabulary"]]
    if profile["beliefs"]:
        lines += ["", "What the subject holds (with a passage that says so):"]
        lines += [f"- {b['text']} [{b['cite']}]" for b in profile["beliefs"]]
    if profile["denials"]:
        lines += ["", "What the subject denies:"] + [f"- {b['text']} [{b['cite']}]" for b in profile["denials"]]
    if profile["derived"]:
        lines += ["", "What follows from those beliefs:"] + [f"- {b['text']} ({b['cite']})" for b in profile["derived"]]
    if profile["causal"]:
        lines += ["", "What the subject says brings about what:"] + [f"- {c['text']} [{c['cite']}]" for c in profile["causal"]]
    if profile["typical"]:
        lines += ["", "Usual levels, where the subject states them:"]
        lines += [f"- {t['label']}: usually {LEVEL_WORD[t['level']]} ({t['n']} situations)" for t in profile["typical"].values()]
    if profile["voice"]:
        lines += ["", "In the subject's own words:"] + [f"[{p['locator']}] {p['text']}" for p in profile["voice"]]
    return "\n".join(lines)


def profile_hash(profile: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:12]


# ---------- retrieval ----------

_WORD = re.compile(r"[a-z]{3,}")
_STOP = set("the and that this with for are was were not but his her its from have has had they them their thee thou "
            "thy thine which what who whom shall will would can could may might all any one into than then there these "
            "those also such upon unto him she you your our ours own being been".split())


class Index:
    """BM25 over passages, for finding what the subject said about situations like this one."""

    def __init__(self, passages: dict[str, dict[str, Any]], exclude: set[str] | frozenset = frozenset()) -> None:
        self.docs = {pid: Counter(self.tokens(p["text"])) for pid, p in passages.items() if pid not in exclude}
        self.passages = passages
        n = len(self.docs) or 1
        df = Counter(t for d in self.docs.values() for t in d)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.avg = sum(sum(d.values()) for d in self.docs.values()) / n

    @staticmethod
    def tokens(text: str) -> list[str]:
        return [w for w in _WORD.findall(text.lower()) if w not in _STOP]

    def search(self, query: str, k: int = 4, skip: set[str] | frozenset = frozenset()) -> list[str]:
        q = set(self.tokens(query))
        scores = []
        for pid, d in self.docs.items():
            if pid in skip:
                continue
            length = sum(d.values()) or 1
            s = sum(self.idf.get(t, 0) * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * length / self.avg)) for t in q if t in d)
            if s > 0:
                scores.append((s, pid))
        return [pid for _, pid in sorted(scores, reverse=True)[:k]]


# ---------- one question to the stand-in ----------


def request(mode: str, profile: dict[str, Any], situation: dict[str, Any], ask: list[str], index: Index | None,
            labels: dict[str, str], own_passage: dict[str, Any] | None = None, k: int = 4,
            skip: set[str] | frozenset = frozenset()) -> dict[str, Any]:
    """Everything one answer needs, as data (so an offline stand-in can answer it too)."""
    known = [{"variable": v, "label": labels.get(v, v), "level": c["level"]} for v, c in situation["cells"].items()
             if c["tier"] in ("set", "stated") and v not in ask]
    query = situation.get("description", "") + " " + " ".join(k_["label"] for k_ in known)
    retrieved = []
    if mode == "subject" and index is not None:
        skip = set(skip) | ({own_passage["id"]} if own_passage else set())
        retrieved = [{"id": pid, "locator": index.passages[pid]["locator"], "text": index.passages[pid]["text"][:900]}
                     for pid in index.search(query, k, skip)]
    return {"mode": mode, "subject": profile["subject"], "profile": profile if mode == "subject" else None,
            "vocabulary": profile["vocabulary"], "situation": {"id": situation.get("id"), "description": situation.get("description", ""),
                                                               "kind": situation.get("kind", ""), "known": known,
                                                               "target": situation.get("target")},
            "own_passage": {"locator": own_passage["locator"], "text": own_passage["text"][:1500]} if own_passage and mode == "subject" else None,
            "retrieved": retrieved, "ask": [{"id": v, "label": labels.get(v, v)} for v in ask]}


def render_request(req: dict[str, Any]) -> tuple[str, str]:
    vocab = "\n".join(f"- {v['id']}: {v['label']} - {v['gloss']}" for v in req["vocabulary"])
    if req["mode"] == "subject":
        prof = req["profile"]
        system = SUBJECT_SYSTEM.replace("{subject}", req["subject"]).replace(
            "{others}", ", ".join(prof["others"]) or "none recorded").replace("{profile}", render_profile(prof))
    elif req["mode"] == "generic":
        system = GENERIC_SYSTEM.replace("{vocabulary}", vocab)
    else:
        system = PLAIN_SYSTEM.replace("{subject}", req["subject"]).replace("{vocabulary}", vocab)
    s = req["situation"]
    parts = [f"Situation ({s['kind'] or 'unspecified'}): {s['description'] or '(no description)'}"]
    if s["known"]:
        parts.append("Known in this situation: " + "; ".join(f"{k['label']} is {LEVEL_WORD[k['level']]}" for k in s["known"]))
    if req.get("own_passage"):
        parts.append(f"The passage this situation comes from [{req['own_passage']['locator']}]:\n{req['own_passage']['text']}")
    if req["retrieved"]:
        parts.append("Passages about situations like this one:\n" + "\n\n".join(f"[{r['locator']}] {r['text']}" for r in req["retrieved"]))
    parts.append("Say how each of these stands in the situation:\n" + "\n".join(f"- {a['id']}: {a['label']}" for a in req["ask"]))
    return system, "\n\n".join(parts)


def claude_answer(req: dict[str, Any]) -> list[dict[str, Any]]:
    system, user = render_request(req)
    return llm._json_call(system, user, ANSWER_SCHEMA, effort="medium")["answers"]


def _answers(coder: Any, req: dict[str, Any]) -> dict[str, dict[str, Any]]:
    wanted = {a["id"] for a in req["ask"]}
    return {a["variable"]: a for a in coder.answer(req) if a["variable"] in wanted and a.get("known")}


def _cell(a: dict[str, Any], source: str) -> dict[str, Any]:
    cited = a["basis"] in ("passage", "belief") and a.get("cite", "").strip()
    return {"level": int(a["level"]), "tier": "implied" if cited else "reading", "source": source,
            "basis": a["basis"], "cite": a.get("cite", ""), "reason": a.get("reason", "")}


def _pmap(fn: Callable, items: list, workers: int) -> list:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(fn, items))


# ---------- held-out validation ----------


def validate(ctx: dict[str, Any], coder: Any, n: int = 40, frac: float = 0.15, seed: int = 11, workers: int = 6,
             log: Callable[[str], None] = print, min_precision: float = 0.6) -> dict[str, Any]:
    """Hide real situations, reveal part of each, and score three stand-ins at predicting the rest."""
    variables, labels = ctx["variables"], ctx["labels"]
    vset = set(variables)
    eligible = [s for s in ctx["situations"] if s["origin"] == "corpus"
                and sum(1 for v, c in s["cells"].items() if v in vset and c["tier"] == "stated") >= 2]
    rng = random.Random(seed)
    held = rng.sample(eligible, min(len(eligible), n, max(1, math.ceil(frac * len(eligible))))) if eligible else []
    exclude = {s["passage"] for s in held}
    profile = ctx["profile_fn"](exclude)
    index = Index(ctx["passages"], exclude)
    log(f"validating on {len(held)} held-out situations ({3 * len(held)} Claude calls)")
    jobs = []
    for s in held:
        stated = sorted(v for v, c in s["cells"].items() if v in vset and c["tier"] == "stated")
        rng.shuffle(stated)
        cut = max(1, len(stated) // 2)
        shown, hidden = stated[:cut], stated[cut:]
        probe = {**s, "cells": {v: s["cells"][v] for v in shown}, "description": ""}  # the description could give it away
        for mode in ("subject", "plain", "generic"):
            jobs.append((s, mode, request(mode, profile, probe, hidden, index, labels), hidden))

    def run(job):
        s, mode, req, hidden = job
        try:
            return s, mode, hidden, _answers(coder, req)
        except llm.LLMUnavailable:
            raise
        except Exception as e:  # one failed answer should not sink the run
            log(f"  {s['id']} ({mode}): {e}")
            return s, mode, hidden, {}

    per_var: dict[str, dict[str, Counter]] = {v: {m: Counter() for m in ("subject", "plain", "generic")} for v in variables}
    examples = []
    for s, mode, hidden, ans in _pmap(run, jobs, workers):
        for v in hidden:
            tally = per_var[v][mode]
            tally["asked"] += 1
            a = ans.get(v)
            if a is None:
                continue
            tally["answered"] += 1
            tally["correct"] += int(a["level"]) == s["cells"][v]["level"]
            if mode == "subject" and len(examples) < 30:
                examples.append({"situation": s["id"], "locator": s["locator"], "variable": v, "actual": s["cells"][v]["level"],
                                 "predicted": int(a["level"]), "cite": a.get("cite", ""), "reason": a.get("reason", "")})

    def score(c: Counter) -> dict[str, Any]:
        return {"asked": c["asked"], "answered": c["answered"], "correct": c["correct"],
                "precision": round(c["correct"] / c["answered"], 3) if c["answered"] else None,
                "accuracy": round(c["correct"] / c["asked"], 3) if c["asked"] else None}

    overall = {m: score(sum((per_var[v][m] for v in variables), Counter())) for m in ("subject", "plain", "generic")}
    rows, trusted = [], []
    for v in variables:
        sc = {m: score(per_var[v][m]) for m in ("subject", "plain", "generic")}
        subj = sc["subject"]
        if subj["answered"] >= 3:
            ok = subj["precision"] >= min_precision
            why = f"{subj['correct']} of {subj['answered']} right" + ("" if ok else f", below {min_precision:.0%}")
        else:
            op = overall["subject"]["precision"]
            gp = overall["generic"]["precision"] or 0
            ok = op is not None and op >= min_precision + 0.1 and op >= gp
            why = f"too few held-out values ({subj['answered']}); " + ("trusted on the overall score" if ok else "not trusted")
        if ok:
            trusted.append(v)
        rows.append({"id": v, "label": labels.get(v, v), "trusted": ok, "why": why, **sc})
    log(f"held-out precision: subject {overall['subject']['precision']}, plain {overall['plain']['precision']}, "
        f"generic {overall['generic']['precision']}; {len(trusted)} of {len(variables)} variables trusted")
    return {"held_out": len(held), "overall": overall, "variables": rows, "trusted": trusted, "examples": examples,
            "settings": {"n": n, "frac": frac, "seed": seed, "min_precision": min_precision}}


# ---------- filling in corpus situations ----------


def fill(ctx: dict[str, Any], coder: Any, trusted: list[str], leak_sample: int = 30, seed: int = 13, workers: int = 6,
         cache: Callable[[str], Any] | None = None, save: Callable[[str, Any], None] | None = None,
         log: Callable[[str], None] = print) -> dict[str, Any]:
    """Fill unknown cells of trusted variables in every corpus situation, through the transformation."""
    labels, passages = ctx["labels"], ctx["passages"]
    profile = ctx["profile_fn"](set())
    index = Index(passages)
    tset = set(trusted)
    ph = profile_hash(profile)
    todo = [s for s in ctx["situations"] if s["origin"] == "corpus" and tset - set(s["cells"])]
    log(f"filling {len(todo)} situations through the subject transformation (up to {len(todo)} Claude calls, "
        f"plus about {2 * leak_sample} for leak checks)")

    def one(s):
        ask = sorted(tset - set(s["cells"]))
        hit = cache(s["id"]) if cache else None
        if hit and hit.get("_key") == ph and set(ask) <= set(hit.get("asked", [])):
            return s["id"], hit
        try:
            ans = _answers(coder, request("subject", profile, s, ask, index, labels, passages.get(s["passage"])))
        except llm.LLMUnavailable:
            raise
        except Exception as e:
            log(f"  {s['locator']}: {e}")
            return s["id"], None
        rec = {"_key": ph, "asked": ask, "answers": ans}
        if save:
            save(s["id"], rec)
        return s["id"], rec

    filled: dict[str, dict[str, Any]] = {}
    for sid, rec in _pmap(one, todo, workers):
        if rec:
            filled[sid] = {v: _cell(a, "transformation") for v, a in rec["answers"].items()}
    checks = leak_checks(ctx, coder, filled, profile, index, leak_sample, seed, workers, log)
    for c in checks:
        if c["leak"]:
            cell = filled[c["situation"]][c["variable"]]
            cell.update({"tier": "reading", "leak": c["why"]})
    n_vals = sum(len(v) for v in filled.values())
    rate = round(sum(1 for c in checks if c["leak"]) / len(checks), 3) if checks else None
    log(f"filled {n_vals} values in {len(filled)} situations; prior-leak rate {rate} on {len(checks)} checked")
    return {"key": ph, "trusted": trusted, "cells": filled, "leak_checks": checks, "prior_leak_rate": rate}


def leak_checks(ctx, coder, filled, profile, index, sample: int, seed: int, workers: int, log) -> list[dict[str, Any]]:
    """Citation knockout and generic-subject diff on a sample of the cited filled values."""
    labels, passages = ctx["labels"], ctx["passages"]
    by_loc = {p["locator"]: pid for pid, p in passages.items()}
    sits = {s["id"]: s for s in ctx["situations"]}
    cited = [(sid, v, c) for sid, cells in filled.items() for v, c in cells.items() if c["tier"] == "implied"]
    rng = random.Random(seed)
    picks = rng.sample(cited, min(sample, len(cited)))

    def one(item):
        sid, v, c = item
        s = sits[sid]
        own = passages.get(s["passage"])
        generic = _answers(coder, request("generic", profile, s, [v], None, labels)).get(v)
        same_as_generic = generic is not None and int(generic["level"]) == c["level"]
        pid = next((p for loc, p in by_loc.items() if loc and loc in c["cite"]), None)
        knocked = None
        if pid and pid != s["passage"]:
            prof2 = ctx["profile_fn"]({pid})
            again = _answers(coder, request("subject", prof2, s, [v], index, labels, own, skip={pid})).get(v)
            knocked = again is not None and int(again["level"]) == c["level"]
        weak = "usual level" in c["cite"] or (c["basis"] == "passage" and pid is None)
        leak = bool(knocked) or (same_as_generic and weak)
        why = ("unchanged without the passage it cites" if knocked else
               "an ordinary adult says the same, and its citation is weak" if leak else "")
        return {"situation": sid, "variable": v, "level": c["level"], "cite": c["cite"], "same_as_generic": same_as_generic,
                "knockout_unchanged": knocked, "leak": leak, "why": why}

    def safe(item):
        try:
            return one(item)
        except llm.LLMUnavailable:
            raise
        except Exception as e:
            log(f"  leak check {item[0]}: {e}")
            return None

    return [r for r in _pmap(safe, picks, workers) if r]


# ---------- vignettes ----------


def vignettes(ctx: dict[str, Any], coder: Any, trusted: list[str], causal: dict[str, Any], per_level: int = 3,
              max_targets: int = 8, seed: int = 17, workers: int = 6, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Factorial vignettes: set one variable low or high in a situation like the subject's, ask about the rest.

    Targets are trusted variables on edges the corpus leaves unoriented first,
    then on any edge. Each vignette is answered as the subject and as an
    ordinary adult; an uncited answer the adult gives too is common sense.
    """
    labels, passages, variables = ctx["labels"], ctx["passages"], ctx["variables"]
    tset = set(trusted)
    reversible = Counter()
    linked = Counter()
    for e in causal.get("edges", []):
        for v in (e["a"], e["b"]):
            linked[v] += 1
            if not e["directed"]:
                reversible[v] += 1
    targets = sorted((v for v in tset if linked[v]), key=lambda v: (-reversible[v], -linked[v], v))[:max_targets]
    profile = ctx["profile_fn"](set())
    index = Index(passages)
    rng = random.Random(seed)
    corpus = [s for s in ctx["situations"] if s["origin"] == "corpus" and s.get("description")]
    jobs = []
    for t in targets:
        pool = [s for s in corpus if t not in s["cells"]] or corpus
        for level in (-1, 1):
            for _ in range(per_level):
                base = rng.choice(pool)
                desc = f"A situation like this one: {base['description']} In it, {labels.get(t, t)} is {LEVEL_WORD[level]}."
                v = {"id": f"vig-{t}-{'hi' if level > 0 else 'lo'}-{len(jobs) + 1}", "origin": "vignette", "kind": "vignette",
                     "target": t, "description": desc, "based_on": base["id"], "passage": None, "locator": "vignette",
                     "order": 10**9 + len(jobs), "contrast": None,
                     "cells": {t: {"level": level, "tier": "set", "quote": ""}}}
                jobs.append(v)
    log(f"{len(jobs)} vignettes on {len(targets)} variables ({2 * len(jobs)} Claude calls)")

    def one(v):
        ask = [x for x in variables if x != v["target"]]
        try:
            subj = _answers(coder, request("subject", profile, v, ask, index, labels))
            gen = _answers(coder, request("generic", profile, v, ask, None, labels))
        except llm.LLMUnavailable:
            raise
        except Exception as e:
            log(f"  {v['id']}: {e}")
            return None
        for x, a in subj.items():
            cell = _cell(a, "vignette")
            g = gen.get(x)
            cell["generic"] = int(g["level"]) if g else None
            if g is not None and int(g["level"]) == cell["level"]:
                if cell["tier"] == "reading":
                    cell["tier"] = "common sense"
                else:
                    cell["agrees_with_common_sense"] = True
            v["cells"][x] = cell
        return v

    done = [v for v in _pmap(one, jobs, workers) if v]
    return {"key": profile_hash(profile), "targets": targets, "trusted": trusted, "situations": done,
            "settings": {"per_level": per_level, "max_targets": max_targets, "seed": seed}}
