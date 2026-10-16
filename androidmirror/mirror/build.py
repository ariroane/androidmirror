"""Build a Mirror from a whole corpus: code, consolidate, measure, reason, discover.

Every Claude call is cached per passage under data/mirror/<corpus>/, so a build
that stops halfway resumes where it left off, and re-running only the causal
step (`python -m androidmirror causal`) costs nothing.
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
from .causal import discover
from .corpus import CorpusSpec, Passage, fetch, split
from .kg import aggregate
from .schema import EVALUATIONS, HELD_MODALITIES, SELF

MEASURE_MAX = 60  # candidate variables shown to the measuring call


class ClaudeCoder:
    """The real coder: three structured Claude calls (see coding.py)."""

    uses_llm = True

    def code(self, text: str, locator: str, subject: str) -> dict[str, Any]:
        return coding.code_passage(text, locator, subject)

    def consolidate(self, lines: list[str], subject: str) -> list[dict[str, Any]]:
        return coding.consolidate(lines, subject)

    def measure(self, text: str, locator: str, subject: str, variables: list[dict[str, str]]) -> list[dict[str, Any]]:
        return coding.measure_passage(text, locator, subject, variables)


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
          measure: bool = True, settings: dict[str, Any] | None = None,
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
        book = cb.build(codings, spec.subject, coder.consolidate if use_llm else None)
        book["_key"] = key
        _write(root / "codebook.json", book)
    concepts = book["concepts"]

    # 3. measurement against the fixed codebook
    mentions = _mentions(passages, codings, book)
    support = Counter({cid: len(ps) for cid, ps in mentions.items()})
    candidates = [cid for cid, _ in support.most_common() if concepts.get(cid, {}).get("variable")][:MEASURE_MAX]
    cand_info = [{"id": c, "label": concepts[c]["label"], "gloss": concepts[c]["gloss"]} for c in candidates]
    if measure:
        ckey = _hash(candidates)
        measured = _cached_map(
            passages, lambda p: root / "measure" / f"{p.id}.json",
            lambda h: h.get("_v") == pv and set(candidates) <= set(h.get("variables", [])),
            lambda p: {"levels": coder.measure(p.text, p.locator, spec.subject, cand_info), "variables": candidates,
                       "_v": pv, "_key": ckey}, workers, log, "measuring")
    else:
        measured = {}
    matrix = _matrix(passages, mentions, measured, candidates)

    # 4. statements and claims
    facts, poles = _facts(passages, codings, book)
    claims = _claims(passages, codings, book)
    nodes = _nodes(concepts, mentions, poles, measured, passages)

    mirror = {
        "meta": {"corpus": spec.to_dict(), "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "prompt_version": pv,
                 "model": llm.DEFAULT_MODEL if getattr(coder, "uses_llm", False) else type(coder).__name__,
                 "passages": len(passages), "coded": len(codings), "measured": len(measured),
                 "measurement": "codebook" if measure else "open coding"},
        "passage_ids": [p.id for p in passages],
        "nodes": nodes, "facts": facts, "claims": claims, "matrix": matrix,
    }
    mirror["causal"] = discover(matrix, concepts, claims, settings)
    _write(root / "mirror.json", mirror)
    log(f"built: {len(nodes)} concepts, {len(facts)} facts, {len(claims)} causal claims, "
        f"{len(mirror['causal'].get('variables', []))} causal variables")
    return mirror


def rediscover(data_dir: Path, corpus_id: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-run only the causal layer, with new settings, from a saved Mirror."""
    path = mirror_dir(data_dir, corpus_id) / "mirror.json"
    mirror = _read(path)
    if mirror is None:
        raise FileNotFoundError(f"no Mirror built at {path}")
    concepts = {k: v for k, v in mirror["nodes"].items()}
    old = mirror.get("causal", {}).get("settings", {})
    mirror["causal"] = discover(mirror["matrix"], concepts, mirror["claims"], {**old, **(settings or {})})
    _write(path, mirror)
    return mirror


# ---------- assembling the pieces ----------


def _mentions(passages: list[Passage], codings: dict[str, Any], book: dict[str, Any]) -> dict[str, dict[str, list]]:
    """concept id -> passage id -> [(level, quote, label)] from open coding."""
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in passages:
        for m in codings.get(p.id, {}).get("concepts", []):
            hit = cb.resolve(book, m["label"])
            if hit and hit[0] not in (SELF, *EVALUATIONS):
                out[hit[0]][p.id].append((m["level"] * hit[1], m.get("quote", ""), m["label"]))
    return out


def _matrix(passages, mentions, measured, candidates) -> dict[str, list[int]]:
    matrix = {c: [0] * len(passages) for c in candidates}
    for i, p in enumerate(passages):
        if p.id in measured:
            for lv in measured[p.id]["levels"]:
                if lv["variable"] in matrix:
                    matrix[lv["variable"]][i] = 1 if lv["level"] > 0 else -1
        else:
            for c in candidates:
                levels = [lv for lv, _, _ in mentions.get(c, {}).get(p.id, [])]
                if levels:
                    s = sum(levels)
                    matrix[c][i] = 1 if s >= 0 else -1
    return matrix


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


def _nodes(concepts, mentions, poles, measured, passages) -> dict[str, dict[str, Any]]:
    loc = {p.id: p.locator for p in passages}
    nodes = {}
    for cid, c in concepts.items():
        ev = []
        for pid, items in mentions.get(cid, {}).items():
            for level, quote, label in items[:1]:
                ev.append({"passage": pid, "locator": loc[pid], "level": level, "quote": quote, "label": label})
        for pid, m in measured.items():
            for lv in m["levels"]:
                if lv["variable"] == cid and pid not in mentions.get(cid, {}):
                    ev.append({"passage": pid, "locator": loc.get(pid, pid), "level": lv["level"], "quote": lv["quote"],
                               "label": "(measured)"})
        order = {p.id: p.order for p in passages}
        ev.sort(key=lambda e: order.get(e["passage"], 0))
        nodes[cid] = {**c, "mentions": ev, "support": len({e["passage"] for e in ev})}
    for pole, info in poles.items():
        base = concepts[info["of"]]
        nodes[pole] = {"label": info["labels"].most_common(1)[0][0], "kind": base["kind"], "variable": False,
                       "gloss": f"the opposite pole of {base['label']}", "aliases": [], "pole_of": info["of"],
                       "mentions": [], "support": 0}
    return nodes
