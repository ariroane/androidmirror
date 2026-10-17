"""Build a Mirror from a whole corpus: code, consolidate, read situations, reason, discover.

Every Claude call is cached per passage under data/mirror/<corpus>/, so a build
that stops halfway resumes where it left off, and re-running only the causal
step (`python -m androidmirror causal`) costs nothing.

The subject transformation's three optional steps (`validate`, `fill`,
`vignettes`, in subject.py) save their results beside the build; `finalize`
folds whatever is there into the situations and re-runs discovery.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .. import llm
from . import codebook as cb
from . import coding
from . import situations as sit
from . import topics as tp
from . import subject as subj
from .causal import discover
from .corpus import CorpusSpec, Passage, fetch, split
from .kg import KnowledgeBase, aggregate
from .schema import EVALUATIONS, HELD_MODALITIES, SELF

MAX_CANDIDATES = 60  # candidate variables shown to the situations call


class ClaudeCoder:
    """The real coder: the structured Claude calls in coding.py and subject.py."""

    uses_llm = True

    def code(self, text: str, locator: str, subject: str) -> dict[str, Any]:
        return coding.code_passage(text, locator, subject)

    def consolidate(self, lines: list[str], subject: str) -> list[dict[str, Any]]:
        return coding.consolidate(lines, subject)

    def situations(self, text: str, locator: str, subject: str, variables: list[dict[str, str]]) -> dict[str, Any]:
        return coding.situations_passage(text, locator, subject, variables)

    def answer(self, req: dict[str, Any]) -> list[dict[str, Any]]:
        return subj.claude_answer(req)


def mirror_dir(data_dir: Path, corpus_id: str) -> Path:
    return data_dir / "mirror" / corpus_id


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _cached_map(items: list[Passage], path_of: Callable[[Passage], Path], fresh: Callable[[Any], bool],
                run: Callable[[Passage], Any], workers: int, log: Callable[[str], None], what: str) -> dict[str, Any]:
    out, todo = {}, []
    for p in items:
        hit = _read(path_of(p))
        if hit is not None and fresh(hit):
            out[p.id] = hit
        else:
            todo.append(p)
    if todo:
        log(f"{what}: {len(todo)} passages to do, {len(out)} cached")
    done = 0

    def one(p: Passage) -> tuple[str, Any]:
        for attempt in range(3):
            try:
                res = run(p)
                _write(path_of(p), res)
                return p.id, res
            except llm.LLMUnavailable:
                raise
            except Exception as e:  # transient API trouble: back off and retry
                if attempt == 2:
                    log(f"  {p.locator}: gave up ({e})")
                    return p.id, None
                time.sleep(2 ** attempt * 3)
        return p.id, None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for pid, res in pool.map(one, todo):
            done += 1
            if res is not None:
                out[pid] = res
            if done % 25 == 0 or done == len(todo):
                log(f"  {what}: {done}/{len(todo)}")
    return out


def build(spec: CorpusSpec, data_dir: Path, coder: Any = None, limit: int | None = None, workers: int = 6,
          read_situations: bool = True, settings: dict[str, Any] | None = None, n_topics: int | None = None,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    coder = coder or ClaudeCoder()
    root = mirror_dir(data_dir, spec.id)
    passages = split(spec, fetch(spec, data_dir))
    if limit:
        passages = passages[:limit]
    log(f"{spec.title}: {len(passages)} passages")
    _write(root / "passages.json", [vars(p) for p in passages])
    pv = coding.prompt_version()

    # 1. open coding
    codings = _cached_map(passages, lambda p: root / "coding" / f"{p.id}.json", lambda h: h.get("_v") == pv,
                          lambda p: {**coder.code(p.text, p.locator, spec.subject), "_v": pv}, workers, log, "coding")

    # 2. codebook
    key = _hash([pv, sorted(codings)])
    book = _read(root / "codebook.json")
    if not book or book.get("_key") != key:
        log("consolidating the codebook")
        use_llm = getattr(coder, "uses_llm", False) and llm.available() or not getattr(coder, "uses_llm", False)
        book = cb.build(codings, spec.subject, coder.consolidate if use_llm else None, workers=workers, log=log)
        book["_key"] = key
        _write(root / "codebook.json", book)
    concepts = book["concepts"]

    # 3. candidate variables: the knowledge graph's quantities, by how often they come up
    mentions = _mentions(passages, codings, book)
    support = Counter({cid: len(ps) for cid, ps in mentions.items()})
    candidates = [cid for cid, _ in support.most_common() if concepts.get(cid, {}).get("variable")][:MAX_CANDIDATES]
    cand_info = [{"id": c, "label": concepts[c]["label"], "gloss": concepts[c]["gloss"]} for c in candidates]

    # topics: what each passage is about, from its words and coded concepts (no Claude calls)
    _write(root / "topics.json", _topics(passages, codings, mentions, concepts, n_topics))

    # 4. situations: what each passage reasons about, and how the variables stand there
    if read_situations:
        sv = coding.situation_version()
        readings = _cached_map(
            passages, lambda p: root / "situations" / f"{p.id}.json",
            lambda h: h.get("_v") == sv and set(candidates) <= set(h.get("variables", [])),
            lambda p: {**coder.situations(p.text, p.locator, spec.subject, cand_info), "variables": candidates, "_v": sv},
            workers, log, "reading situations")
    else:
        readings = {}

    # 5. statements and claims, and the situations they come with
    facts, poles = _facts(passages, codings, book)
    claims = _claims(passages, codings, book)
    nodes = _nodes(concepts, mentions, poles, passages)
    pdicts = {p.id: vars(p) for p in passages}
    cset = set(candidates)
    base, stated_model = [], []
    for p in passages:
        if p.id in readings:
            base += sit.from_reading(pdicts[p.id], readings[p.id], cset)
            stated_model += sit.model_claims(pdicts[p.id], readings[p.id], cset)
        elif not read_situations:  # a cheap fallback: one situation per passage from the open coding
            cells = {}
            for cid in candidates:
                items = mentions.get(cid, {}).get(p.id)
                if items:
                    level = sum(lv for lv, _, _ in items)
                    cells[cid] = {"level": 1 if level >= 0 else -1, "tier": "stated", "quote": items[0][1]}
            if cells:
                base.append({"id": f"{p.id}:1", "passage": p.id, "locator": p.locator, "order": p.order * 100,
                             "origin": "corpus", "kind": "actual", "description": "", "cells": cells, "contrast": None})
    _write(root / "situations_base.json", {"situations": base, "stated_model": stated_model, "candidates": candidates})

    mirror = {
        "meta": {"corpus": spec.to_dict(), "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "prompt_version": pv,
                 "model": llm.DEFAULT_MODEL if getattr(coder, "uses_llm", False) else type(coder).__name__,
                 "passages": len(passages), "coded": len(codings), "read": len(readings),
                 "units": "situations" if read_situations else "passages (open coding only)"},
        "passage_ids": [p.id for p in passages],
        "nodes": nodes, "facts": facts, "claims": claims, "candidates": candidates,
        "causal": {"settings": settings or {}},
    }
    mirror = finalize(mirror, root, settings, log)
    log(f"built: {len(nodes)} concepts, {len(facts)} facts, {len(claims)} causal claims, "
        f"{len(mirror['situations'])} situations, {len(mirror['causal'].get('variables', []))} causal variables")
    return mirror


def finalize(mirror: dict[str, Any], root: Path, settings: dict[str, Any] | None = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Situations with every tier filled in (beliefs, the transformation's fills, usual levels, vignettes), then discovery."""
    base = _read(root / "situations_base.json") or {"situations": [], "stated_model": [], "candidates": []}
    candidates = base["candidates"]
    cset = set(candidates)
    labels = {k: v["label"] for k, v in mirror["nodes"].items()}
    kb = KnowledgeBase(mirror["nodes"], mirror["facts"])
    rules = sit.implications(kb, cset, labels)
    situations = json.loads(json.dumps(base["situations"]))
    topics = _read(root / "topics.json") or {"passages": {}, "topics": []}
    for s in situations:
        s["topic"] = topics["passages"].get(s["passage"])
        sit.propagate(s, rules)
    filled = _read(root / "fill.json")
    if filled:
        for s in situations:
            for v, cell in filled["cells"].get(s["id"], {}).items():
                if v in cset and v not in s["cells"]:
                    s["cells"][v] = cell
            sit.propagate(s, rules)
    typical = sit.typical_levels(situations, candidates)
    sit.add_defaults(situations, typical)
    vig = _read(root / "vignettes.json")
    if vig:
        for v in vig["situations"]:
            v = {**v, "cells": {k: c for k, c in v["cells"].items() if k in cset},
                 "topic": topics["passages"].get((v.get("based_on") or "").split(":")[0])}  # its source situation's topic
            sit.propagate(v, rules)
            situations.append(v)
    concepts = mirror["nodes"]
    old = (mirror.get("causal") or {}).get("settings", {})
    cfg = {**old, **(settings or {})}
    mirror["situations"] = situations
    mirror["stated_model"] = base["stated_model"]
    mirror["typical"] = {v: {**t, "label": labels.get(v, v)} for v, t in typical.items()}
    mirror["rules"] = len(rules)
    mirror["topics"] = {"k": len(topics["topics"]), "topics": topics["topics"], "note": topics.get("note")}
    mirror["transformation"] = {"validation": _summary(_read(root / "validation.json")),
                                "fill": {k: filled[k] for k in ("trusted", "prior_leak_rate", "leak_checks")} if filled else None,
                                "vignettes": {k: vig[k] for k in ("targets", "trusted", "settings")} | {"n": len(vig["situations"])} if vig else None}
    mirror["causal"] = discover(situations, candidates, concepts, mirror["claims"], base["stated_model"], cfg)
    _write(root / "mirror.json", mirror)
    return mirror


def _summary(validation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not validation:
        return None
    return {k: validation[k] for k in ("held_out", "overall", "variables", "trusted", "examples", "settings")}


def rediscover(data_dir: Path, corpus_id: str, settings: dict[str, Any] | None = None,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """Re-run only the causal layer, with new settings, from a saved Mirror."""
    root = mirror_dir(data_dir, corpus_id)
    mirror = _read(root / "mirror.json")
    if mirror is None:
        raise FileNotFoundError(f"no Mirror built at {root / 'mirror.json'}")
    return finalize(mirror, root, settings, log)


# ---------- the subject transformation's steps ----------


def context(mirror: dict[str, Any], root: Path) -> dict[str, Any]:
    """What subject.py needs from a built Mirror."""
    passages = {p["id"]: p for p in _read(root / "passages.json")}
    nodes, facts, claims = mirror["nodes"], mirror["facts"], mirror["claims"]
    kb = KnowledgeBase(nodes, facts)
    variables = mirror["causal"].get("variables") or mirror.get("candidates", [])[:25]
    situations = [s for s in mirror["situations"] if s["origin"] == "corpus"]
    subject = nodes.get(SELF, {}).get("label", "the subject")

    def profile_fn(exclude) -> dict[str, Any]:
        return subj.build_profile(subject, nodes, facts, kb, claims, situations, passages, variables, set(exclude))

    return {"subject": subject, "situations": situations, "passages": passages, "variables": variables,
            "labels": {k: v["label"] for k, v in nodes.items()}, "profile_fn": profile_fn}


def run_step(step: str, data_dir: Path, corpus_id: str, coder: Any = None, workers: int = 6,
             log: Callable[[str], None] = print, **opts: Any) -> dict[str, Any]:
    """validate, fill or vignettes on a built Mirror; saves the result and re-runs discovery."""
    coder = coder or ClaudeCoder()
    root = mirror_dir(data_dir, corpus_id)
    mirror = _read(root / "mirror.json")
    if mirror is None:
        raise FileNotFoundError(f"no Mirror built at {root / 'mirror.json'}")
    ctx = context(mirror, root)
    validation = _read(root / "validation.json")
    if step == "validate":
        out = subj.validate(ctx, coder, workers=workers, log=log, **opts)
        _write(root / "validation.json", out)
    else:
        if not validation:
            raise RuntimeError("run `validate` first: only variables the transformation predicts well are filled in")
        trusted = [v for v in validation["trusted"] if v in ctx["variables"]]
        if not trusted:
            raise RuntimeError("validation trusted no variable, so there is nothing the transformation may fill in")
        if step == "fill":
            out = subj.fill(ctx, coder, trusted, workers=workers, log=log,
                            cache=lambda sid: _read(root / "fill" / f"{_safe(sid)}.json"),
                            save=lambda sid, rec: _write(root / "fill" / f"{_safe(sid)}.json", rec), **opts)
            _write(root / "fill.json", out)
        elif step == "vignettes":
            out = subj.vignettes(ctx, coder, trusted, mirror["causal"], workers=workers, log=log, **opts)
            _write(root / "vignettes.json", out)
        else:
            raise ValueError(step)
    finalize(mirror, root, None, log)
    return out


def _safe(sid: str) -> str:
    return sid.replace(":", "_")


# ---------- assembling the pieces ----------


def _topics(passages: list[Passage], codings: dict[str, Any], mentions: dict[str, dict[str, list]],
            concepts: dict[str, Any], k: int | None) -> dict[str, Any]:
    if k == 0 or len(passages) < 30:
        return {"k": 0, "topics": [], "passages": {}, "note": "off" if k == 0 else "too few passages to group"}
    k = k or tp.default_k(len(passages))
    by_passage: dict[str, list[str]] = defaultdict(list)
    for cid, ps in mentions.items():
        for pid in ps:
            by_passage[pid].append(cid)
    found = tp.cluster([tp.features(p.text, by_passage.get(p.id, [])) for p in passages], k)
    labels = {cid: c["label"] for cid, c in concepts.items()}
    topics = [{**t, "name": f"t{t['id'] + 1}", "label": tp.label(t, labels)} for t in found["topics"]]
    return {"k": found["k"], "topics": topics,
            "passages": {p.id: f"t{lab + 1}" for p, lab in zip(passages, found["labels"])}}


def _mentions(passages: list[Passage], codings: dict[str, Any], book: dict[str, Any]) -> dict[str, dict[str, list]]:
    """concept id -> passage id -> [(level, quote, label)] from open coding."""
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in passages:
        for m in codings.get(p.id, {}).get("concepts", []):
            hit = cb.resolve(book, m["label"])
            if hit and hit[0] not in (SELF, *EVALUATIONS):
                out[hit[0]][p.id].append((m["level"] * hit[1], m.get("quote", ""), m["label"]))
    return out


def _endpoint(book, label, poles) -> str | None:
    hit = cb.resolve(book, label)
    if not hit:
        return None
    cid, pol = hit
    if pol == 1:
        return cid
    pole = f"{cid}~"
    poles.setdefault(pole, {"of": cid, "labels": Counter()})["labels"][label.strip().lower()] += 1
    return pole


def _facts(passages, codings, book) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    poles: dict[str, Any] = {}
    stmts = []
    for p in passages:
        for s in codings.get(p.id, {}).get("statements", []):
            a, b = _endpoint(book, s["subject"], poles), _endpoint(book, s["object"], poles)
            if not a or not b:
                continue
            attribution = SELF if s["attribution"].strip().lower() in (SELF, book["concepts"][SELF]["label"].lower()) \
                else s["attribution"]
            stmts.append({"s": a, "r": s["relation"], "o": b, "neg": bool(s["negated"]), "passage": p.id,
                          "locator": p.locator, "quote": s["quote"], "modality": s["modality"],
                          "attribution": attribution, "qualifier": s.get("qualifier", ""), "wording": s.get("wording", "")})
    for pole, info in poles.items():  # an opposite pole is the contrary of its concept, by construction
        stmts.append({"s": pole, "r": "opposes", "o": info["of"], "neg": False, "passage": "codebook",
                      "locator": "codebook", "quote": "", "modality": "asserts", "attribution": SELF})
    return aggregate(stmts), poles


def _claims(passages, codings, book) -> list[dict[str, Any]]:
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for p in passages:
        for c in codings.get(p.id, {}).get("causal_claims", []):
            ca, ef = cb.resolve(book, c["cause"]), cb.resolve(book, c["effect"])
            if not ca or not ef or ca[0] == ef[0]:
                continue
            sign = c["sign"] * ca[1] * ef[1]
            rec = agg.setdefault((ca[0], ef[0]), {"cause": ca[0], "effect": ef[0], "evidence": [], "signs": Counter(),
                                                  "bases": Counter(), "scopes": Counter(), "held": False})
            rec["evidence"].append({"passage": p.id, "locator": p.locator, "quote": c["quote"], "sign": sign,
                                    "direction_basis": c["direction_basis"], "strength_word": c.get("strength_word", ""),
                                    "modality": c.get("modality", "asserts"), "attribution": c.get("attribution", SELF)})
            rec["signs"][sign] += 1
            rec["bases"][c["direction_basis"]] += 1
            rec["scopes"][c["scope"]] += 1
            held = c.get("attribution", SELF).strip().lower() in (SELF, book["concepts"][SELF]["label"].lower()) \
                and c.get("modality", "asserts") in HELD_MODALITIES
            rec["held"] = rec["held"] or held
    out = []
    for rec in agg.values():
        rec["support"] = len({e["passage"] for e in rec["evidence"]})
        rec["sign"] = 1 if rec["signs"][1] >= rec["signs"][-1] else -1
        rec["sign_conflict"] = bool(rec["signs"][1] and rec["signs"][-1])
        rec["signs"], rec["bases"], rec["scopes"] = (dict(rec["signs"]), dict(rec["bases"]), dict(rec["scopes"]))
        rec["signs"] = {str(k): v for k, v in rec["signs"].items()}
        out.append(rec)
    return sorted(out, key=lambda r: -r["support"])


def _nodes(concepts, mentions, poles, passages) -> dict[str, dict[str, Any]]:
    loc = {p.id: p.locator for p in passages}
    nodes = {}
    for cid, c in concepts.items():
        ev = []
        for pid, items in mentions.get(cid, {}).items():
            for level, quote, label in items[:1]:
                ev.append({"passage": pid, "locator": loc[pid], "level": level, "quote": quote, "label": label})
        order = {p.id: p.order for p in passages}
        ev.sort(key=lambda e: order.get(e["passage"], 0))
        nodes[cid] = {**c, "mentions": ev, "support": len({e["passage"] for e in ev})}
    for pole, info in poles.items():
        base = concepts[info["of"]]
        nodes[pole] = {"label": info["labels"].most_common(1)[0][0], "kind": base["kind"], "variable": False,
                       "gloss": f"the opposite pole of {base['label']}", "aliases": [], "pole_of": info["of"],
                       "mentions": [], "support": 0}
    return nodes
