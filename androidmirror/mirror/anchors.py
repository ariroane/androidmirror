"""Anchors: what each concept of a Mirror means, as a sense in Open English WordNet.

An anchor says "when this subject says X, they mean WordNet sense S", with one
of the SKOS mapping relations:

- exactMatch: X is that sense;
- closeMatch: X is near that sense but not quite it (a Stoic *passion* is not
  WordNet's "strong feeling", but close);
- broadMatch: X is a kind of that sense (SKOS: the sense is broader than X);
- narrowMatch: that sense is a kind of X;
- relatedMatch: X is related to it, nothing more;
- noMatch: none of the senses listed for X's name is what the subject means
  (common for technical terms; the concept stays the Mirror's own).

`polarity` -1 says X is the opposite of the sense (a Mirror's "trouble" may be
anchored as the opposite of "tranquillity").

Anchors are proposed and approved the way answers to the Question engine are,
and nothing changes until a named person approves. Proposals come from:

1. lexical match: the concept's label and aliases looked up among WordNet's
   lemmas (and, for phrases WordNet lacks, the head word, as a broader sense);
2. Claude (optional), choosing among the listed senses only, with the
   concept's gloss and two of its quotes in view; it may say none fits;
3. you: `anchors set CONCEPT SENSE`.

The book is anchors.json beside mirror.json; approved anchors are also written
as anchors.sssom.tsv (the SSSOM mapping format) for use outside the app. Each
approved anchor carries what Transmutate needs from WordNet (the sense's
hypernyms with their distances, and its antonyms), so combining two Mirrors
does not need WordNet loaded. An anchor is never evidence of what the subject
holds: it changes which merges Transmutate proposes, and with what relation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .. import llm
from .schema import EVALUATIONS, SELF
from .wordnet import CITATION, PREFIX, WordNet, curie, uncurie

PREDICATES = ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch", "noMatch")
SHARED = {SELF, *EVALUATIONS}
BATCH = 25  # concepts per Claude call
MAX_SENSES = 8  # senses listed per name
UP_DEPTH = 12  # hypernym steps kept with an approved anchor

# Which WordNet lexicographer files a Mirror kind usually lands in: a check, never a rule.
KIND_LEX = {
    "agent": {"noun.person", "noun.group", "noun.Tops"},
    "place": {"noun.location", "noun.object", "noun.artifact"},
    "event": {"noun.event", "noun.act", "noun.process", "noun.time", "noun.phenomenon", "noun.state"},
    "object": {"noun.object", "noun.artifact", "noun.body", "noun.substance", "noun.food", "noun.plant",
               "noun.animal", "noun.possession", "noun.communication"},
    "disposition": {"noun.attribute", "noun.state", "noun.feeling", "noun.cognition"},
    "emotion": {"noun.feeling", "noun.state", "noun.attribute"},
    "practice": {"noun.act", "noun.cognition", "noun.communication", "noun.process"},
    "belief": {"noun.cognition", "noun.communication"},
    "state": {"noun.state", "noun.feeling", "noun.attribute", "noun.cognition"},
    "faculty": {"noun.cognition", "noun.attribute", "noun.Tops", "noun.state"},
}


def path(root: Path) -> Path:
    return root / "anchors.json"


def tsv_path(root: Path) -> Path:
    return root / "anchors.sssom.tsv"


def load(root: Path) -> dict[str, Any]:
    p = path(root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"proposals": []}


def save(root: Path, book: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = path(root).with_suffix(".tmp")
    tmp.write_text(json.dumps(book, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path(root))
    write_sssom(root, book)


def approved(book: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The approved anchor of each concept (one per concept; a later approval supersedes)."""
    return {p["concept"]: p for p in book.get("proposals", []) if p["status"] == "approved"}


def digest(root: Path) -> bytes:
    """What of the anchors a combination rests on, for Transmutate's staleness check."""
    book = load(root)
    rows = sorted((c, p["predicate"], p.get("object", ""), p.get("polarity", 1)) for c, p in approved(book).items())
    return json.dumps(rows).encode() if rows else b""


def concepts(mirror: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in mirror["nodes"].items() if k not in SHARED and not v.get("pole_of")}


# ---------- 1. lexical candidates ----------


def _pos_for(kind: str) -> list[list[str]]:
    """Parts of speech to try, in order: nouns first; a practice may be a verb; a state may be an adjective."""
    if kind == "practice":
        return [["n"], ["v"], ["a"]]
    return [["n"], ["a"], ["v"]]


def lexical(wn: WordNet, node: dict[str, Any]) -> list[dict[str, Any]]:
    """The senses a concept's names point to: [{synset, via, polarity, head}], best first, without repeats."""
    names = [(node.get("label", ""), 1)]
    for al in node.get("aliases", []):
        names.append((al[: -len("(opposite)")].strip(), -1) if al.endswith("(opposite)") else (al, 1))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    tiers = _pos_for(node.get("kind", ""))
    for head in (False, True):
        for name, pol in names:
            word = name.split()[-1] if head else name
            if head and len(name.split()) < 2:
                continue
            for pos in tiers:
                syns = wn.senses(word, pos)
                if syns:
                    break
            for s in syns[:MAX_SENSES]:
                if s not in seen:
                    seen.add(s)
                    out.append({"synset": s, "via": word, "polarity": pol, "head": head})
        if out:
            break  # a phrase's head word is tried only if no name of it is in WordNet as a whole
    return out


def _kind_fits(kind: str, lex: str) -> bool:
    want = KIND_LEX.get(kind)
    return True if not want or not lex.startswith("noun.") else lex in want


def _lexical_proposal(wn: WordNet, cid: str, node: dict[str, Any], cands: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Without Claude: the first sense of the label itself; exact if WordNet knows only one, close otherwise."""
    own = [c for c in cands if not c["head"] and c["polarity"] == 1]
    if not own:
        return None
    first = next((c for c in own if _kind_fits(node.get("kind", ""), wn.get(c["synset"])["lex"])), own[0])
    single = len({c["synset"] for c in own if c["via"] == first["via"]}) == 1
    return {"predicate": "exactMatch" if single else "closeMatch", "synset": first["synset"], "polarity": 1,
            "confidence": 0.6 if single else 0.3, "justification": "lexical",
            "why": f"{first['via']!r} has {'one sense' if single else 'several senses; the first is taken'} in WordNet"}


# ---------- 2. Claude chooses among the listed senses ----------

CHOOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"anchors": {"type": "array", "items": {
        "type": "object",
        "properties": {"concept": {"type": "string"},
                       "sense": {"type": "string", "description": "one of the listed sense ids, or 'none'"},
                       "predicate": {"type": "string", "enum": list(PREDICATES)},
                       "polarity": {"type": "integer", "enum": [1, -1]},
                       "confidence": {"type": "number"},
                       "why": {"type": "string"}},
        "required": ["concept", "sense", "predicate", "polarity", "confidence", "why"],
        "additionalProperties": False}}},
    "required": ["anchors"],
    "additionalProperties": False,
}

CHOOSE_SYSTEM = """You are anchoring the concepts of an Android Mirror, a model of {subject}'s subjective world built
from their own words, to word senses in Open English WordNet.

For each concept you get its id, label, kind, gloss (how the subject uses it), up to two quotes, and the WordNet senses
its names point to, each as `sense_id: words (lexicographer file): definition [kind of: ...]`. Choose ONE listed sense
and say how the subject's concept relates to it:
- exactMatch: what the subject means is that sense;
- closeMatch: near it, but the subject's use differs in a way that matters (a technical or period sense);
- broadMatch: the subject's concept is a narrower kind of that sense (use this for head-word senses of a phrase,
  such as "fear of death" -> fear);
- narrowMatch: the sense is a narrower kind of the subject's concept;
- relatedMatch: only related.
Or answer sense "none" with predicate noMatch when no listed sense is what the subject means. Do not invent senses:
use the ids exactly as listed. polarity -1 means the concept is the opposite of the sense (rare: usually it is 1).
Read the subject on their own terms: an ancient or technical meaning is not the modern everyday one, and then
closeMatch or noMatch is the honest answer. confidence is 0 to 1. `why`: one short sentence."""


def _concept_block(wn: WordNet, cid: str, node: dict[str, Any], cands: list[dict[str, Any]]) -> str:
    quotes = [m["quote"] for m in node.get("mentions", [])[:2] if m.get("quote")]
    lines = [f"## {cid}: {node.get('label', '')} ({node.get('kind', '')}) - {node.get('gloss', '')}"]
    if node.get("aliases"):
        lines.append("also called: " + ", ".join(node["aliases"][:8]))
    lines += [f'quote: "{q}"' for q in quotes]
    for c in cands:
        tag = " (head word of the label)" if c["head"] else (" (the name is the opposite)" if c["polarity"] < 0 else "")
        lines.append(f"- {curie(c['synset'])}: {wn.describe(c['synset'])}{tag}")
    return "\n".join(lines)


def claude_choose(wn: WordNet, subject: str, batch: list[tuple[str, dict[str, Any], list[dict[str, Any]]]]
                  ) -> dict[str, dict[str, Any]]:
    """One call: Claude's choice for each concept of the batch, only among its listed senses."""
    system = CHOOSE_SYSTEM.replace("{subject}", subject)
    user = "\n\n".join(_concept_block(wn, cid, n, c) for cid, n, c in batch)
    res = llm._json_call(system, user, CHOOSE_SCHEMA, effort="medium", max_tokens=12000, cache_system=True)
    allowed = {cid: {curie(c["synset"]) for c in cands} for cid, _, cands in batch}
    out = {}
    for r in res["anchors"]:
        cid = r["concept"]
        if cid not in allowed:
            continue
        if r["predicate"] == "noMatch" or r["sense"] == "none":
            out[cid] = {"predicate": "noMatch", "synset": "", "polarity": 1, "confidence": r["confidence"],
                        "justification": "claude", "why": r["why"]}
        elif r["sense"] in allowed[cid]:
            out[cid] = {"predicate": r["predicate"], "synset": uncurie(r["sense"]), "polarity": r["polarity"],
                        "confidence": max(0.0, min(1.0, r["confidence"])), "justification": "claude", "why": r["why"]}
    return out


# ---------- proposals ----------


def _pid(concept: str, predicate: str, obj: str, polarity: int) -> str:
    return "a-" + hashlib.sha1(f"{concept}|{predicate}|{obj}|{polarity}".encode()).hexdigest()[:8]


def make(wn: WordNet | None, cid: str, node: dict[str, Any], choice: dict[str, Any],
         cands: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A proposal record, with what Transmutate needs from WordNet carried along."""
    syn = choice.get("synset", "")
    rec = {"id": _pid(cid, choice["predicate"], curie(syn) if syn else "", choice.get("polarity", 1)),
           "concept": cid, "label": node.get("label", cid), "kind": node.get("kind", ""),
           "predicate": choice["predicate"], "object": curie(syn) if syn else "", "polarity": choice.get("polarity", 1),
           "confidence": round(float(choice.get("confidence", 0.5)), 2), "justification": choice["justification"],
           "why": choice.get("why", ""), "candidates": [curie(c["synset"]) for c in cands or []],
           "status": "proposed", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if syn and wn is not None:
        s = wn.get(syn)
        rec.update({"object_label": wn.name(syn), "object_gloss": s["def"], "lexfile": s["lex"],
                    "up": {curie(k): d for k, d in wn.ancestors(syn).items() if 0 < d <= UP_DEPTH},
                    "depth": max(wn.ancestors(syn).values()),
                    "antonyms": sorted(curie(a) for a in wn.antonyms(syn))})
        if not _kind_fits(node.get("kind", ""), s["lex"]):
            rec["check"] = f"kind {node.get('kind')} but WordNet files the sense under {s['lex']}"
    return rec


def propose(root: Path, mirror: dict[str, Any], wn: WordNet, use_claude: bool = False, limit: int | None = None,
            redo: bool = False, workers: int = 4, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Propose an anchor for each concept that has none yet (proposed or approved). Changes nothing else."""
    if use_claude and not llm.available():
        raise llm.LLMUnavailable("anchoring with Claude needs ANTHROPIC_API_KEY in your local .env")
    book = load(root)
    taken = {p["concept"] for p in book["proposals"] if p["status"] in ("proposed", "approved")}
    todo = []
    for cid, node in sorted(concepts(mirror).items(), key=lambda kv: -kv[1].get("support", 0)):
        if cid in taken and not redo:
            continue
        cands = lexical(wn, node)
        if cands:
            todo.append((cid, node, cands))
    if limit:
        todo = todo[:limit]
    choices: dict[str, dict[str, Any]] = {}
    if use_claude:
        subject = mirror["meta"]["corpus"]["subject"]
        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        log(f"Claude chooses senses for {len(todo)} concepts ({len(batches)} calls)")
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for i, got in enumerate(ex.map(lambda b: claude_choose(wn, subject, b), batches)):
                choices.update(got)
                log(f"  {min((i + 1) * BATCH, len(todo))}/{len(todo)}")
    made = []
    for cid, node, cands in todo:
        choice = choices.get(cid) or _lexical_proposal(wn, cid, node, cands)
        if choice is None:
            continue
        rec = make(wn, cid, node, choice, cands)
        if redo:
            for p in book["proposals"]:
                if p["concept"] == cid and p["status"] == "proposed":
                    p["status"] = "rejected"
        if not any(p["id"] == rec["id"] and p["status"] in ("proposed", "approved") for p in book["proposals"]):
            book["proposals"].append(rec)
            made.append(rec)
    book["wordnet"] = CITATION
    save(root, book)
    log(f"{len(made)} anchors proposed; nothing is used until approved")
    return made


def set_anchor(root: Path, mirror: dict[str, Any], wn: WordNet, concept: str, sense: str, predicate: str = "exactMatch",
               polarity: int = 1, approver: str = "") -> dict[str, Any]:
    """Your own anchor: SENSE is a WordNet id (oewn:07531593-n) or 'none'. Approved at once if you name yourself."""
    nodes = concepts(mirror)
    if concept not in nodes:
        raise KeyError(concept)
    if predicate not in PREDICATES:
        raise ValueError(f"predicate is one of {', '.join(PREDICATES)}")
    syn = "" if sense in ("none", "") or predicate == "noMatch" else uncurie(sense)
    if syn and syn not in wn.synsets:
        raise KeyError(f"no WordNet sense {sense}")
    rec = make(wn, concept, nodes[concept], {"predicate": "noMatch" if not syn else predicate, "synset": syn,
                                             "polarity": polarity, "confidence": 1.0, "justification": "you",
                                             "why": "set by you"})
    book = load(root)
    book["proposals"] = [p for p in book["proposals"] if not (p["id"] == rec["id"] and p["status"] == "proposed")]
    book["proposals"].append(rec)
    save(root, book)
    return approve(root, [rec["id"]], approver)[0] if approver.strip() else rec


def proposals(root: Path, status: str | None = None) -> list[dict[str, Any]]:
    return [p for p in load(root)["proposals"] if status is None or p["status"] == status]


def approve(root: Path, ids: list[str], approver: str, min_confidence: float | None = None) -> list[dict[str, Any]]:
    """Approve proposals by id, or every pending one with ids ['all'] (optionally only those at min_confidence or above).
    A named approver is required, as for answers. A concept's earlier approved anchor is superseded."""
    if not approver.strip():
        raise ValueError("approving needs the approver's name")
    book = load(root)
    pending = {p["id"]: p for p in book["proposals"] if p["status"] == "proposed"}
    if ids == ["all"]:
        chosen = [p for p in pending.values() if min_confidence is None or p["confidence"] >= min_confidence]
    else:
        missing = [i for i in ids if i not in pending]
        if missing:
            raise KeyError(f"not pending: {', '.join(missing)}")
        chosen = [pending[i] for i in ids]
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for p in chosen:
        for q in book["proposals"]:
            if q["concept"] == p["concept"] and q["status"] == "approved":
                q["status"], q["superseded_by"] = "superseded", p["id"]
        p.update(status="approved", approved_by=approver.strip(), approved_at=now)
    save(root, book)
    return chosen


def _set_status(root: Path, ids: list[str], frm: str, to: str) -> list[dict[str, Any]]:
    book = load(root)
    rows = [p for p in book["proposals"] if p["status"] == frm and (ids == ["all"] or p["id"] in ids)]
    if ids != ["all"] and len(rows) != len(set(ids)):
        raise KeyError(f"not {frm}: {', '.join(sorted(set(ids) - {p['id'] for p in rows}))}")
    for p in rows:
        p["status"] = to
        p[f"{to}_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(root, book)
    return rows


def reject(root: Path, ids: list[str]) -> list[dict[str, Any]]:
    return _set_status(root, ids, "proposed", "rejected")


def revoke(root: Path, ids: list[str]) -> list[dict[str, Any]]:
    return _set_status(root, ids, "approved", "revoked")


# ---------- SSSOM export ----------

_JUSTIFICATION = {"lexical": "semapv:LexicalMatching", "claude": "semapv:UnspecifiedMatching",
                  "you": "semapv:ManualMappingCuration"}
COLUMNS = ["subject_id", "subject_label", "predicate_id", "object_id", "object_label", "mapping_justification",
           "confidence", "author_label", "mapping_date", "subject_category", "comment", "other"]


def write_sssom(root: Path, book: dict[str, Any]) -> None:
    rows = sorted(approved(book).values(), key=lambda p: p["concept"])
    name = root.name
    head = ["# curie_map:", "#   mirror: https://github.com/ariroane/androidmirror/concept/",
            "#   oewn: https://en-word.net/id/oewn-", "#   skos: http://www.w3.org/2004/02/skos/core#",
            "#   semapv: https://w3id.org/semapv/vocab/", "#   sssom: https://w3id.org/sssom/",
            f"# mapping_set_id: https://github.com/ariroane/androidmirror/anchors/{name}",
            "# license: https://creativecommons.org/licenses/by/4.0/",
            f"# comment: anchors of the {name} Mirror to {CITATION}; approved rows only"]
    out = head + ["\t".join(COLUMNS)]
    for p in rows:
        pred = "skos:exactMatch" if p["predicate"] == "noMatch" else f"skos:{p['predicate']}"
        obj = "sssom:NoTermFound" if p["predicate"] == "noMatch" else p["object"]
        how = _JUSTIFICATION.get(p["justification"], "semapv:UnspecifiedMatching")
        comment = ("chosen by Claude among the listed senses; " if p["justification"] == "claude" else "") + p.get("why", "")
        vals = [f"mirror:{name}/{p['concept']}", p["label"], pred, obj, p.get("object_label", ""), how,
                str(p["confidence"]), p.get("approved_by", ""), (p.get("approved_at") or "")[:10], p["kind"],
                comment.replace("\t", " ").replace("\n", " "), f"polarity={p['polarity']}"]
        out.append("\t".join(vals))
    tsv_path(root).write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------- in words ----------


def describe(p: dict[str, Any]) -> str:
    if p["predicate"] == "noMatch":
        target = "no WordNet sense"
    else:
        rel = {"exactMatch": "is", "closeMatch": "is close to", "broadMatch": "is a kind of",
               "narrowMatch": "has as a kind", "relatedMatch": "is related to"}[p["predicate"]]
        target = f"{rel}{' the opposite of' if p['polarity'] < 0 else ''} {p['object']} {p.get('object_label', '')}: " \
                 f"{p.get('object_gloss', '')}"
    flag = f"  [check: {p['check']}]" if p.get("check") else ""
    return (f"{p['id']}  {p['label']} ({p['concept']}) {target}  "
            f"({p['justification']}, {p['confidence']:.0%}){flag}\n    {p.get('why', '')}")


def summary(book: dict[str, Any]) -> dict[str, Any]:
    from collections import Counter

    by = Counter(p["status"] for p in book["proposals"])
    pred = Counter(p["predicate"] for p in approved(book).values())
    return {"status": dict(by), "approved_by_predicate": dict(pred)}


# ---------- command line and web ----------


class AnchorsIn(BaseModel):
    mirror: str | None = None  # the Mirror to anchor (any folder in data/mirror/); default: the one shown
    corpus: str | None = None
    ids: list[str] = []
    approver: str = ""
    claude: bool = False
    limit: int | None = None
    min_confidence: float | None = None


def cli(args: Any, root: Path, data_dir: Path) -> None:
    from . import wordnet

    mirror_path = root / "mirror.json"
    if not mirror_path.exists() and args.action != "senses":
        raise SystemExit(f"no Mirror built at {root}")
    try:
        if args.action == "propose":
            mirror = json.loads(mirror_path.read_text(encoding="utf-8"))
            made = propose(root, mirror, wordnet.load(data_dir), args.claude, args.limit, args.redo)
            for p in made[: args.show]:
                print(describe(p))
            print(f"\n{len(made)} proposed. Approve: anchors approve all --by NAME (or IDs; --min-confidence 0.7)")
        elif args.action == "list":
            for p in proposals(root, args.status):
                print(f"[{p['status']}] " + describe(p))
            print(json.dumps(summary(load(root))))
        elif args.action == "approve":
            rows = approve(root, args.ids or ["all"], args.by, args.min_confidence)
            print(f"approved {len(rows)} anchors; Transmutate reads them from now on")
        elif args.action == "reject":
            print(f"rejected {len(reject(root, args.ids or ['all']))}")
        elif args.action == "revoke":
            print(f"revoked {len(revoke(root, args.ids))}")
        elif args.action == "set":
            if len(args.ids) != 2:
                raise ValueError("anchors set CONCEPT SENSE (a WordNet id such as oewn:07531593-n, or none)")
            mirror = json.loads(mirror_path.read_text(encoding="utf-8"))
            p = set_anchor(root, mirror, wordnet.load(data_dir), args.ids[0], args.ids[1], args.predicate,
                           -1 if args.opposite else 1, args.by)
            print(f"[{p['status']}] " + describe(p))
        elif args.action == "senses":
            wn = wordnet.load(data_dir)
            for word in args.ids:
                for s in wn.senses(word):
                    print(f"{curie(s)}  {wn.describe(s)}")
    except (KeyError, ValueError, llm.LLMUnavailable) as e:
        raise SystemExit(str(e)) from None


def routes(data_dir: Path, root_of: Callable[[str | None], Path]) -> Any:
    """GET /api/anchors, POST /api/anchors/{propose,approve,reject,revoke}."""
    from fastapi import APIRouter, HTTPException

    from . import wordnet

    r = APIRouter()

    def guard(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except KeyError as e:
            raise HTTPException(404, f"not found: {e.args[0]}") from None
        except (ValueError, llm.LLMUnavailable) as e:
            raise HTTPException(400, str(e)) from None

    @r.get("/api/anchors")
    def get_anchors(mirror: str | None = None, corpus: str | None = None, status: str | None = None) -> dict:
        root = root_of(mirror or corpus)
        book = load(root)
        return {"summary": summary(book), "proposals": [p for p in book["proposals"] if status in (None, p["status"])]}

    @r.post("/api/anchors/propose")
    def post_propose(a: AnchorsIn) -> dict:
        root = root_of(a.mirror or a.corpus)
        mirror = json.loads((root / "mirror.json").read_text(encoding="utf-8"))
        made = guard(lambda: propose(root, mirror, wordnet.load(data_dir, log=lambda *_: None), a.claude, a.limit,
                                     log=lambda *_: None))
        return {"proposed": len(made), "summary": summary(load(root))}

    @r.post("/api/anchors/approve")
    def post_approve(a: AnchorsIn) -> dict:
        rows = guard(lambda: approve(root_of(a.mirror or a.corpus), a.ids or ["all"], a.approver, a.min_confidence))
        return {"approved": len(rows)}

    @r.post("/api/anchors/reject")
    def post_reject(a: AnchorsIn) -> dict:
        return {"rejected": len(guard(lambda: reject(root_of(a.mirror or a.corpus), a.ids or ["none given"])))}

    @r.post("/api/anchors/revoke")
    def post_revoke(a: AnchorsIn) -> dict:
        return {"revoked": len(guard(lambda: revoke(root_of(a.mirror or a.corpus), a.ids)))}

    return r


__all__ = ["PREDICATES", "PREFIX", "approve", "approved", "load", "propose", "reject", "revoke", "set_anchor"]
