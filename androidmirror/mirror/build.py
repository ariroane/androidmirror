"""Build a Mirror from a whole corpus: code, consolidate, read judgments, reason, fit.

Every Claude call is cached per passage under data/mirror/<corpus>/, so a build
that stops halfway resumes where it left off, and re-fitting only the causal
layer (`python -m androidmirror causal`) costs nothing.
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
from . import coding, mutate
from . import judgments as jd
from .causal import fit, holdout, public
from .corpus import CorpusSpec, Passage, fetch, split
from .kg import aggregate
from .schema import EVALUATIONS, SELF

MAX_VOCABULARY = 150  # concepts listed for the judgments call, most often mentioned first


class ClaudeCoder:
    """The real coder: the structured Claude calls in coding.py."""

    uses_llm = True

    def code(self, text: str, locator: str, subject: str) -> dict[str, Any]:
        return coding.code_passage(text, locator, subject)

    def consolidate(self, lines: list[str], subject: str) -> list[dict[str, Any]]:
        return coding.consolidate(lines, subject)

    def judgments(self, text: str, locator: str, subject: str, concepts: list[dict[str, str]]) -> dict[str, Any]:
        return coding.judgments_passage(text, locator, subject, concepts)


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
          settings: dict[str, Any] | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
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

    # 3. the vocabulary the judgments are read in: the codebook's concepts, most often mentioned first
    mentions = _mentions(passages, codings, book)
    support = Counter({cid: len(ps) for cid, ps in mentions.items()})
    vocab_ids = [cid for cid, _ in support.most_common() if cid in concepts][:MAX_VOCABULARY]
    vocab = [{"id": c, "label": concepts[c]["label"], "gloss": concepts[c]["gloss"]} for c in vocab_ids]

    # 4. the subject's causal judgments and values, one call per passage
    jv = coding.judgment_version() + _hash(vocab)
    readings = _cached_map(passages, lambda p: root / "judgments" / f"{p.id}.json", lambda h: h.get("_v") == jv,
                           lambda p: {**coder.judgments(p.text, p.locator, spec.subject, vocab), "_v": jv},
                           workers, log, "reading judgments")

    # 5. facts, judgments, values
    facts, poles = _facts(passages, codings, book)
    nodes = _nodes(concepts, mentions, poles, passages)
    pdicts = {p.id: vars(p) for p in passages}
    judgments, values, unresolved = [], [], []
    for p in passages:
        if p.id in readings:
            r = jd.read(pdicts[p.id], readings[p.id], book, spec.subject)
            judgments += r["judgments"]
            values += r["values"]
            unresolved += r["unresolved"]
    mirror = {
        "meta": {"corpus": spec.to_dict(), "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "prompt_version": pv,
                 "judgment_version": jv,
                 "model": llm.DEFAULT_MODEL if getattr(coder, "uses_llm", False) else type(coder).__name__,
                 "passages": len(passages), "coded": len(codings), "read": len(readings), "prototype": 3},
        "passage_ids": [p.id for p in passages],
        "nodes": nodes, "facts": facts, "judgments": judgments, "unresolved": unresolved,
        "values": jd.values(values), "value_records": values,
        "causal": {"settings": settings or {}},
    }
    mirror = mutate.replay(mirror, root, log)  # the mutations the user approved, made again on the new reading
    mirror = finalize(mirror, root, settings, log)
    st = mirror["causal"]["stats"]
    log(f"built: {len(nodes)} concepts, {len(facts)} facts, {st['held_judgments']} of the subject's causal judgments "
        f"({len(unresolved)} outside the codebook), {st['arrows']} arrows, {st['rejected']} constraints in tension, "
        f"{len(mirror['values'])} values")
    return mirror


def finalize(mirror: dict[str, Any], root: Path, settings: dict[str, Any] | None = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Fit the causal model to the judgments and check it on held-out passages (no Claude calls)."""
    old = (mirror.get("causal") or {}).get("settings", {})
    cfg = {**old, **(settings or {})}
    model = fit(mirror["judgments"], cfg)
    mirror["causal"] = {**public(model), "holdout": holdout(mirror["judgments"], cfg)}
    _write(root / "mirror.json", mirror)
    return mirror


def refit(data_dir: Path, corpus_id: str, settings: dict[str, Any] | None = None,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    """Re-fit only the causal layer, with new settings, from a saved Mirror."""
    root = mirror_dir(data_dir, corpus_id)
    mirror = _read(root / "mirror.json")
    if mirror is None:
        raise FileNotFoundError(f"no Mirror built at {root / 'mirror.json'}")
    if "judgments" not in mirror:
        raise RuntimeError("this Mirror was built by Prototype 2; run build again to read the judgments")
    return finalize(mirror, root, settings, log)


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
