"""Transmutate: combine two Mirrors into one, then trim it to what is strong.

Three steps, and nothing is written into data/mirror/ until you approve:

1. Align. Which concepts of Mirror A are concepts of Mirror B? Pairs come from
   the same name (label or alias), from Claude if you ask for it, and from you.
   A pair may be "opposite": B's concept is the contrary pole of A's (B's
   "unrest" is A's "tranquillity", reversed), and then every sign that runs
   through it is flipped. Each concept is paired at most once, so combining
   never merges two concepts of the same Mirror (that is mutate's job).
2. Combine. Aligned concepts become one node (its id is A's, unprefixed);
   everything else keeps its own Mirror's name as a prefix ("fomc:inflation").
   The two subjects are both `self`: the result is one voice that says what
   both say. Passages keep their source as a prefix. Facts and judgments are
   pooled with all their evidence, and the causal model is fitted to the pooled
   judgments, so the combination is consistent by construction: where the two
   Mirrors disagree, the fit takes the better-supported side and reports the
   other as in tension. Disagreements that only appear once the two are put
   together (a cycle made of A's arrow and B's arrow, an evaluation A calls good
   and B calls bad) are listed separately, as are claims both Mirrors make.
3. Fitness. Would a reader of half the combined corpus still come away with
   this? Each passage is kept with probability `share`, the causal model is
   refitted on the kept passages, and a claim or denial survives a sample if
   the refitted model still says it (a path of the same sign; the denial still
   holds). A stated fact survives if any passage holding it is kept, which is
   1 - (1 - share)^support exactly. What survives less often than `keep` is
   proposed for trimming. A claim the model rejects is kept anyway, as a
   tension, if its own evidence is strong, so trimming never hides a
   disagreement. Trimmed items are archived in trimmed.json, never deleted.

The proposal lives in data/transmutations/<a>+<b>/; `approve` writes the
combined Mirror to data/mirror/<a>+<b>/, where the app shows it like any other.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .. import llm
from .causal import fit, holdout, public
from .codebook import norm
from .judgments import constraints as build_constraints
from .kg import KnowledgeBase, fact_key
from .schema import EVALUATIONS, HELD_MODALITIES, SELF

DEFAULTS = {"share": 0.5, "samples": 50, "keep": 0.6, "seed": 0}
SHARED = {SELF, *EVALUATIONS}


# ---------- loading ----------


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def combined_id(a: str, b: str) -> str:
    return f"{a}+{b}"


def proposal_dir(data_dir: Path, a: str, b: str) -> Path:
    return data_dir / "transmutations" / combined_id(a, b)


def load_source(data_dir: Path, name: str) -> dict[str, Any]:
    """A built Mirror as raw data: its mirror.json, passages, and a hash to tell if it changes."""
    root = data_dir / "mirror" / name
    path = root / "mirror.json"
    if not path.exists():
        raise FileNotFoundError(f"no Mirror built for {name!r} (looked for {path})")
    raw = path.read_bytes()
    mirror = json.loads(raw)
    if "judgments" not in mirror:
        raise RuntimeError(f"{name} was built by Prototype 2; run build again before combining it")
    passages = json.loads((root / "passages.json").read_text(encoding="utf-8"))
    return {"id": name, "mirror": mirror, "passages": passages, "hash": hashlib.sha256(raw).hexdigest()[:16]}


# ---------- 1. alignment ----------


def _names(node: dict[str, Any]) -> dict[str, int]:
    """Normalized names a concept goes by, with the polarity each carries."""
    out = {norm(node.get("label", "")): 1}
    for al in node.get("aliases", []):
        if al.endswith("(opposite)"):
            out.setdefault(norm(al[: -len("(opposite)")]), -1)
        else:
            out.setdefault(norm(al), 1)
    out.pop("", None)
    return out


def _concepts(mirror: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in mirror["nodes"].items() if k not in SHARED and not v.get("pole_of")}


def candidates(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Pairs of concepts that go by the same name in both Mirrors, one partner each, best first."""
    ca, cb_ = _concepts(a["mirror"]), _concepts(b["mirror"])
    index: dict[str, list[tuple[str, int, bool]]] = defaultdict(list)
    for aid, n in ca.items():
        for name, pol in _names(n).items():
            index[name].append((aid, pol, name == norm(n["label"])))
    found = []
    for bid, n in cb_.items():
        for name, pol in _names(n).items():
            for aid, apol, a_main in index.get(name, []):
                main = a_main and name == norm(n["label"])
                found.append({"a": aid, "b": bid, "polarity": pol * apol, "how": "same label" if main else "alias",
                              "why": f"both are called {name!r}",
                              "_rank": (main, ca[aid].get("support", 0) + n.get("support", 0))})
    return _one_to_one(sorted(found, key=lambda p: p["_rank"], reverse=True))


def _one_to_one(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used_a, used_b, out = set(), set(), []
    for p in pairs:
        if p["a"] in used_a or p["b"] in used_b:
            continue
        used_a.add(p["a"])
        used_b.add(p["b"])
        out.append({k: v for k, v in p.items() if not k.startswith("_")})
    return out


ALIGN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"pairs": {"type": "array", "items": {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "polarity": {"type": "integer", "enum": [1, -1]},
                       "why": {"type": "string"}},
        "required": ["a", "b", "polarity", "why"], "additionalProperties": False}}},
    "required": ["pairs"],
    "additionalProperties": False,
}

ALIGN_SYSTEM = """You are aligning two Android Mirrors, each a model of one subjective system built from a corpus.
Mirror A reflects {a_subject}; Mirror B reflects {b_subject}. Each lists its concepts as `id: label (kind) - gloss`.

Pair a concept of A with a concept of B only when they are the same idea, so that what one subject says about it is
a claim about the same thing the other means (polarity 1), or exact contraries, one the absence or reverse of the
other, such as calm and agitation (polarity -1). Words that merely sound alike across very different domains are not
the same idea: say so by leaving them out. Each concept appears in at most one pair. Use the ids exactly as given.
In `why`, say in one sentence what both mean. Fewer, sure pairs are better than many loose ones."""


def claude_pairs(a: dict[str, Any], b: dict[str, Any], limit: int = 150) -> list[dict[str, Any]]:
    """Pairs Claude proposes by meaning, from each Mirror's most-mentioned concepts."""
    def listing(m: dict[str, Any]) -> tuple[str, set[str]]:
        top = sorted(_concepts(m).items(), key=lambda kv: -kv[1].get("support", 0))[:limit]
        return "\n".join(f"{k}: {v['label']} ({v['kind']}) - {v.get('gloss', '')}" for k, v in top), {k for k, _ in top}

    la, ia = listing(a["mirror"])
    lb, ib = listing(b["mirror"])
    system = ALIGN_SYSTEM.replace("{a_subject}", a["mirror"]["meta"]["corpus"]["subject"]) \
        .replace("{b_subject}", b["mirror"]["meta"]["corpus"]["subject"])
    res = llm._json_call(system, f"Mirror A concepts:\n{la}\n\nMirror B concepts:\n{lb}", ALIGN_SCHEMA, effort="high")
    return [{"a": p["a"], "b": p["b"], "polarity": p["polarity"], "how": "claude", "why": p["why"]}
            for p in res["pairs"] if p["a"] in ia and p["b"] in ib]


def parse_pair(text: str, polarity: int = 1) -> dict[str, Any]:
    """'a_id=b_id' as a pair you made."""
    if "=" not in text:
        raise ValueError(f"a pair is A_CONCEPT=B_CONCEPT, got {text!r}")
    aid, bid = (s.strip() for s in text.split("=", 1))
    return {"a": aid, "b": bid, "polarity": polarity, "how": "you", "why": "paired by you"}


def align(a: dict[str, Any], b: dict[str, Any], use_claude: bool = False, add: list[dict[str, Any]] | None = None,
          drop: list[str] | None = None) -> dict[str, Any]:
    """The alignment to propose: your pairs first, then same-name pairs, then Claude's; one partner each."""
    ca, cb_ = _concepts(a["mirror"]), _concepts(b["mirror"])
    yours = []
    for p in add or []:
        if p["a"] not in ca or p["b"] not in cb_:
            raise KeyError(f"no concept {p['a']!r} in {a['id']} or {p['b']!r} in {b['id']}")
        yours.append(p)
    found = yours + candidates(a, b) + (claude_pairs(a, b) if use_claude else [])
    dropped = set(drop or [])
    pairs, set_aside = [], []
    for p in _one_to_one(found):
        (set_aside if p["a"] in dropped or p["b"] in dropped or f"{p['a']}={p['b']}" in dropped else pairs).append(p)
    for p in pairs:
        p["a_label"], p["b_label"] = ca[p["a"]]["label"], cb_[p["b"]]["label"]
        p["a_kind"], p["b_kind"] = ca[p["a"]]["kind"], cb_[p["b"]]["kind"]
    return {"pairs": pairs, "dropped": set_aside}


# ---------- 2. combining ----------


class _Ids:
    """How each source's node ids map into the combined Mirror."""

    def __init__(self, a: str, b: str, pairs: list[dict[str, Any]]) -> None:
        self.prefix = {"a": a, "b": b}
        self.b_to_a = {p["b"]: (p["a"], p["polarity"]) for p in pairs}
        self.a_aligned = {p["a"] for p in pairs}

    def concept(self, side: str, nid: str) -> tuple[str, int]:
        """A judgment or value endpoint: (combined concept id, sign flip)."""
        if nid in SHARED:
            return nid, 1
        if side == "a":
            return (nid if nid in self.a_aligned else f"{self.prefix['a']}:{nid}"), 1
        if nid in self.b_to_a:
            return self.b_to_a[nid]
        return f"{self.prefix['b']}:{nid}", 1

    def node(self, side: str, nid: str) -> str:
        """A fact endpoint or node id: the opposite pole is its own node, so nothing flips."""
        pole = nid.endswith("~")
        base = nid[:-1] if pole else nid
        cid, pol = self.concept(side, base)
        if cid in SHARED:
            return cid
        return cid + "~" if pole != (pol == -1) else cid

    def passage(self, side: str, pid: str) -> str:
        return f"{self.prefix[side]}:{pid}"


def combine(a: dict[str, Any], b: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The combined Mirror's data (not fitted, not saved) and its passages."""
    ids = _Ids(a["id"], b["id"], pairs)
    sides = (("a", a), ("b", b))
    passages, offset = [], 0
    for side, src in sides:
        for p in src["passages"]:
            passages.append({**p, "id": ids.passage(side, p["id"]), "locator": f"{src['id']} · {p['locator']}",
                             "group": f"{src['id']} · {p.get('group', '')}", "order": offset + p.get("order", 0),
                             "source": src["id"]})
        offset += len(src["passages"]) + 1
    loc = {p["id"]: p["locator"] for p in passages}

    def ev_map(side: str, e: dict[str, Any]) -> dict[str, Any]:
        pid = ids.passage(side, e["passage"])
        return {**e, "passage": pid, "locator": loc.get(pid, f"{ids.prefix[side]} · {e.get('locator', '')}"),
                "source": ids.prefix[side]}

    # nodes
    nodes: dict[str, dict[str, Any]] = {}
    for side, src in sides:
        for nid, n in src["mirror"]["nodes"].items():
            new = ids.node(side, nid)
            mentions = [ev_map(side, m) for m in n.get("mentions", [])]
            if new not in nodes:
                rec = {**n, "mentions": mentions, "sources": {src["id"]: nid}, "aliases": list(n.get("aliases", []))}
                if n.get("pole_of"):
                    rec["pole_of"] = ids.node(side, n["pole_of"])
                nodes[new] = rec
                continue
            rec = nodes[new]
            rec["sources"][src["id"]] = nid
            if nid in SHARED:
                continue
            for name in [n["label"]] + n.get("aliases", []):
                if name.lower() != rec["label"].lower() and name not in rec["aliases"]:
                    rec["aliases"].append(name)
            rec["mentions"] = rec.get("mentions", []) + mentions
            rec["support"] = rec.get("support", 0) + n.get("support", 0)
            rec["variable"] = bool(rec.get("variable")) or bool(n.get("variable"))
            if n.get("kind") != rec.get("kind"):
                rec.setdefault("kinds", {})[src["id"]] = n.get("kind")
    for nid, rec in nodes.items():  # B's concept paired as the opposite of A's becomes that concept's pole
        if nid.endswith("~") and not rec.get("pole_of"):
            rec["pole_of"] = nid[:-1]
            rec["variable"] = False

    # facts: pool the evidence of facts that are now the same fact
    facts: dict[str, dict[str, Any]] = {}
    for side, src in sides:
        for f in src["mirror"]["facts"]:
            s, o = ids.node(side, f["s"]), ids.node(side, f["o"])
            key = fact_key(s, f["r"], o, f["neg"])
            rec = facts.setdefault(key, {"s": s, "r": f["r"], "o": o, "neg": f["neg"], "evidence": [], "held": False,
                                         "qualifiers": [], "sources": []})
            rec["evidence"] += [ev_map(side, e) for e in f["evidence"]]
            rec["held"] = rec["held"] or f["held"]
            for q in f.get("qualifiers", []):
                if q not in rec["qualifiers"]:
                    rec["qualifiers"].append(q)
            for w in f.get("wordings", []):
                rec.setdefault("wordings", [])
                if w not in rec["wordings"]:
                    rec["wordings"].append(w)
            if f["held"] and src["id"] not in rec["sources"]:
                rec["sources"].append(src["id"])
    for f in facts.values():
        f["support"] = len({e["passage"] for e in f["evidence"]})
        f["modalities"] = sorted({e["modality"] for e in f["evidence"] if "modality" in e})

    # judgments, unresolved, values
    judgments, unresolved, value_records = [], [], []
    for side, src in sides:
        m = src["mirror"]
        for j in m["judgments"]:
            x, px = ids.concept(side, j["x"])
            y, py = ids.concept(side, j["y"])
            z, pz = ids.concept(side, j["z"]) if j["z"] else ("", 1)
            if len({x, y, z} - {""}) < (3 if j["z"] else 2):
                continue
            flip = py * pz if j["kind"] == "screens_off" else px * py
            judgments.append({**ev_map(side, j), "x": x, "y": y, "z": z, "sign": j["sign"] * flip})
        unresolved += [ev_map(side, u) for u in m.get("unresolved", [])]
        for v in m.get("values", []):
            cid, pol = ids.concept(side, v["concept"])
            more, less = (v["more"], v["less"]) if pol == 1 else (v["less"], v["more"])
            value_records.append({"concept": cid, "more": more, "less": less, "source": src["id"],
                                  "evidence": [{**ev_map(side, e), "concept": cid, "direction": e["direction"] * pol}
                                               for e in v["evidence"]]})
    values = _pool_values(value_records)

    ma, mb = a["mirror"]["meta"], b["mirror"]["meta"]
    corpus = {"id": combined_id(a["id"], b["id"]), "title": f"{ma['corpus']['title']} + {mb['corpus']['title']}",
              "subject": f"{ma['corpus']['subject']} and {mb['corpus']['subject']}", "splitter": "transmuted"}
    if SELF in nodes:
        nodes[SELF] = {**nodes[SELF], "label": corpus["subject"]}
    mirror = {
        "meta": {"corpus": corpus, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "prototype": 3,
                 "passages": len(passages), "coded": ma.get("coded", 0) + mb.get("coded", 0),
                 "read": ma.get("read", 0) + mb.get("read", 0), "model": f"{ma.get('model')} + {mb.get('model')}",
                 "sources": [a["id"], b["id"]]},
        "passage_ids": [p["id"] for p in passages],
        "nodes": nodes, "facts": list(facts.values()), "judgments": judgments, "unresolved": unresolved,
        "values": values, "causal": {"settings": {**a["mirror"].get("causal", {}).get("settings", {})}},
    }
    return {"mirror": mirror, "passages": passages}


def _pool_values(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, dict[str, Any]] = {}
    for r in records:
        rec = by.setdefault(r["concept"], {"concept": r["concept"], "more": 0, "less": 0, "evidence": [], "by_source": {}})
        rec["more"] += r["more"]
        rec["less"] += r["less"]
        rec["evidence"] = (rec["evidence"] + r["evidence"])[:8]
        rec["by_source"][r["source"]] = 1 if r["more"] > r["less"] else -1 if r["less"] > r["more"] else 0
    out = []
    for rec in by.values():
        rec["direction"] = 1 if rec["more"] > rec["less"] else -1 if rec["less"] > rec["more"] else 0
        out.append(rec)
    return sorted(out, key=lambda r: -(r["more"] + r["less"]))


# ---------- what combining agrees and disagrees on ----------


def _source_of(pid: str) -> str:
    return pid.split(":", 1)[0]


def _held_sources(c: dict[str, Any]) -> set[str]:
    return {_source_of(e["passage"]) for e in c["evidence"] if e["held"]}


def crossings(mirror: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Claims both Mirrors make, and the tensions that need both Mirrors to arise."""
    rows = {c["id"]: c for c in model["constraints"]}
    allc = {c["id"]: c for c in build_constraints(mirror["judgments"])}
    agreed, causal = [], []
    for cid, c in allc.items():
        if not c["held"]:
            continue
        mine = _held_sources(c)
        row = rows[cid]
        if len(mine) == 2:
            agreed.append({"id": cid, "type": c["type"], "weight": c["weight"], "status": row["status"],
                           "sign_conflict": c["sign_conflict"]})
        if c["sign_conflict"]:
            by = defaultdict(set)
            for e in c["evidence"]:
                if e["held"]:
                    by[e["sign"]].add(_source_of(e["passage"]))
            if by[1] and by[-1] and by[1] != by[-1]:
                causal.append({"id": cid, "kind": "sign", "x": c["x"], "y": c["y"],
                               "why": f"{', '.join(sorted(by[1]))} says it raises, {', '.join(sorted(by[-1]))} says it lowers"})
        if row["status"] == "rejected":
            against = [_held_sources(allc[k]) for k in row.get("conflicts", []) if k in allc]
            theirs = set().union(*against)
            # it comes from combining if neither Mirror on its own both makes the claim and holds all it runs into
            alone = any(s in mine and all(s in t for t in against) for s in mine | theirs)
            if against and not alone:
                causal.append({"id": cid, "kind": row["reason"], "x": c["x"], "y": c["y"], "z": c["z"],
                               "conflicts": row.get("conflicts", []),
                               "why": f"{'/'.join(sorted(mine))} says it; {'/'.join(sorted(theirs))} rules it out"})
    kb = KnowledgeBase(mirror["nodes"], mirror["facts"])
    kg = []
    for v in kb.violations:
        srcs = set()
        for k in v.get("facts", []):
            s, r, o, *neg = k.split("|")
            if neg:
                f = next((f for f in mirror["facts"] if fact_key(f["s"], f["r"], f["o"], True) == k), None)
                srcs |= {_source_of(e["passage"]) for e in (f or {}).get("evidence", [])}
            else:
                srcs |= {_source_of(p) for p in kb.stated_support((s, r, o))}
        if len(srcs) == 2:
            kg.append(v)
    values = [{"concept": v["concept"], "by_source": v["by_source"]} for v in mirror["values"]
              if len(set(v["by_source"].values()) - {0}) == 2]
    return {"agreed": sorted(agreed, key=lambda r: -r["weight"]), "causal": causal, "kg": kg, "values": values}


# ---------- 3. fitness ----------


def _held_passages(f: dict[str, Any]) -> set[str]:
    return {e["passage"] for e in f["evidence"] if e.get("attribution") == SELF and e.get("modality") in HELD_MODALITIES}


def fitness(mirror: dict[str, Any], model: dict[str, Any], settings: dict[str, Any] | None = None,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """How often each claim, denial and fact survives a reader who sees only part of the corpus."""
    cfg = {**DEFAULTS, **(settings or {})}
    share, rounds = cfg["share"], cfg["samples"]
    rng = random.Random(cfg["seed"])
    pids = sorted({j["passage"] for j in mirror["judgments"]})
    rows = [c for c in model["constraints"] if c["held"]]
    survived: Counter = Counter()
    for r in range(rounds):
        out = {p for p in pids if rng.random() >= share}
        g = fit(mirror["judgments"], model["settings"], exclude=out)
        gg, st = g["_graph"], {c["id"]: c["status"] for c in g["constraints"]}
        for c in rows:
            if c["type"] == "reaches":
                ok = gg.reaches(c["x"], c["y"]) and any(s == c["sign"] for _, s in gg.signed_paths(c["x"], c["y"]))
            else:
                ok = st.get(c["id"]) == "holds"
            survived[c["id"]] += ok
        if (r + 1) % 10 == 0:
            log(f"  fitness: {r + 1}/{rounds} samples")
    claims = {}
    for c in rows:
        own = 1 - (1 - share) ** c["weight"]
        claims[c["id"]] = {"model": round(survived[c["id"]] / rounds, 3), "evidence": round(own, 3)}
    facts = {}
    for f in mirror["facts"]:
        held = _held_passages(f)
        codebook = all(e["passage"].endswith(":codebook") for e in f["evidence"])
        facts[fact_key(f["s"], f["r"], f["o"], f["neg"])] = 1.0 if codebook else round(1 - (1 - share) ** len(held), 3) if held else 0.0
    return {"settings": cfg, "claims": claims, "facts": facts}


def trims(mirror: dict[str, Any], model: dict[str, Any], fit_: dict[str, Any]) -> list[dict[str, Any]]:
    """What the fitness says to cut: weak claims and denials, facts the subject does not hold firmly, orphaned concepts."""
    keep = fit_["settings"]["keep"]
    out = []
    for c in model["constraints"]:
        if not c["held"]:
            out.append({"id": f"c:{c['id']}", "what": "judgment", "why": "a view the subject reports or entertains, not their own",
                        "fitness": 0.0, "weight": 0})
            continue
        s = fit_["claims"][c["id"]]
        if s["model"] >= keep:
            continue
        if c["status"] == "rejected" and s["evidence"] >= keep:
            continue  # strongly said but outweighed: kept, so the trimmed Mirror still shows the tension
        what = "claim" if c["type"] == "reaches" else "denial"
        out.append({"id": f"c:{c['id']}", "what": what, "fitness": s["model"], "weight": c["weight"],
                    "why": f"survives {s['model']:.0%} of half-corpus readings (needs {keep:.0%})"
                           + (", and is in tension" if c["status"] == "rejected" else "")})
    for f in mirror["facts"]:
        k = fact_key(f["s"], f["r"], f["o"], f["neg"])
        s = fit_["facts"][k]
        if s < keep:
            why = f"held in {len(_held_passages(f))} passage(s): survives {s:.0%} of half-corpus readings" if f["held"] \
                else "a fact the subject reports or entertains, not their own"
            out.append({"id": f"f:{k}", "what": "fact", "fitness": s, "weight": f["support"], "why": why})
    cut = {t["id"] for t in out}
    used = set(SHARED)
    for f in mirror["facts"]:
        if f"f:{fact_key(f['s'], f['r'], f['o'], f['neg'])}" not in cut:
            used |= {f["s"], f["o"]}
    for j in _kept_judgments(mirror["judgments"], cut):
        used |= {j["x"], j["y"], j["z"]} - {""}
    used |= {v["concept"] for v in mirror["values"]}
    used |= {n["pole_of"] for nid, n in mirror["nodes"].items() if nid in used and n.get("pole_of")}
    for nid, n in mirror["nodes"].items():
        if nid not in used:
            out.append({"id": f"n:{nid}", "what": "concept", "fitness": 0.0, "weight": n.get("support", 0),
                        "why": "nothing strong is said about it once the weak items go"})
    return out


def _constraint_ids(j: dict[str, Any]) -> list[str]:
    if j["kind"] == "screens_off":
        return [f"reaches|{j['y']}|{j['z']}", f"only_through|{j['x']}|{j['y']}|{j['z']}"]
    if j["kind"] == "independent" or (j["kind"] == "intervention" and j["sign"] == 0):
        return [f"no_path|{j['x']}|{j['y']}"]
    return [f"reaches|{j['x']}|{j['y']}"]


def _kept_judgments(judgments: list[dict[str, Any]], cut: set[str]) -> list[dict[str, Any]]:
    """A judgment stays if any constraint it bears on stays."""
    return [j for j in judgments if any(f"c:{c}" not in cut for c in _constraint_ids(j))]


def apply_trims(mirror: dict[str, Any], cut: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The trimmed Mirror data, and the archive of what was trimmed (nothing is thrown away)."""
    kept_j = _kept_judgments(mirror["judgments"], cut)
    kept_ids = {id(j) for j in kept_j}
    archive = {"judgments": [j for j in mirror["judgments"] if id(j) not in kept_ids],
               "facts": [f for f in mirror["facts"] if f"f:{fact_key(f['s'], f['r'], f['o'], f['neg'])}" in cut],
               "nodes": {k: v for k, v in mirror["nodes"].items() if f"n:{k}" in cut}}
    out = {**mirror, "judgments": kept_j,
           "facts": [f for f in mirror["facts"] if f"f:{fact_key(f['s'], f['r'], f['o'], f['neg'])}" not in cut],
           "nodes": {k: v for k, v in mirror["nodes"].items() if f"n:{k}" not in cut}}
    return out, archive


# ---------- the proposal ----------


def propose(data_dir: Path, a_name: str, b_name: str, use_claude: bool = False, add: list[dict[str, Any]] | None = None,
            drop: list[str] | None = None, settings: dict[str, Any] | None = None,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """Align, combine and score two Mirrors, and save the proposal. Writes nothing into data/mirror/."""
    if a_name == b_name:
        raise ValueError("pick two different Mirrors")
    a, b = load_source(data_dir, a_name), load_source(data_dir, b_name)
    if use_claude and not llm.available():
        raise llm.LLMUnavailable("aligning with Claude needs ANTHROPIC_API_KEY in your local .env")
    alignment = align(a, b, use_claude, add, drop)
    log(f"aligned {len(alignment['pairs'])} concepts of {a_name} with {b_name}")
    combo = combine(a, b, alignment["pairs"])
    mirror = combo["mirror"]
    model = fit(mirror["judgments"], mirror["causal"]["settings"])
    log("scoring fitness")
    fit_ = fitness(mirror, model, settings, log)
    cut = trims(mirror, model, fit_)
    trimmed, _ = apply_trims(mirror, {t["id"] for t in cut})
    after = fit(trimmed["judgments"], mirror["causal"]["settings"])
    before_arrows = {(e["src"], e["dst"], e["sign"]) for e in model["edges"]}
    proposal = {
        "version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": combined_id(a_name, b_name),
        "sources": {s["id"]: {"hash": s["hash"], "title": s["mirror"]["meta"]["corpus"]["title"],
                              "subject": s["mirror"]["meta"]["corpus"]["subject"]} for s in (a, b)},
        "a": a_name, "b": b_name, "settings": fit_["settings"],
        "alignment": alignment["pairs"], "dropped": alignment["dropped"],
        "crossings": crossings(mirror, model),
        "trims": [{"n": i + 1, **t} for i, t in enumerate(sorted(cut, key=lambda t: ("cfn".index(t["id"][0]), -t["weight"], t["id"])))],
        "fitness": fit_["claims"],
        "stats": {
            "combined": {"concepts": len(mirror["nodes"]), "facts": len(mirror["facts"]),
                         "judgments": len(mirror["judgments"]), **{k: model["stats"][k] for k in ("arrows", "rejected")}},
            "trimmed": {"concepts": len(trimmed["nodes"]), "facts": len(trimmed["facts"]),
                        "judgments": len(trimmed["judgments"]), **{k: after["stats"][k] for k in ("arrows", "rejected")}},
            "arrows_kept": len(before_arrows & {(e["src"], e["dst"], e["sign"]) for e in after["edges"]}),
            "arrows_new": [f"{e['src']}|{e['dst']}" for e in after["edges"]
                           if (e["src"], e["dst"], e["sign"]) not in before_arrows],
        },
    }
    root = proposal_dir(data_dir, a_name, b_name)
    _write(root / "proposal.json", proposal)
    (root / "proposal.md").write_text(describe(proposal, mirror["nodes"]), encoding="utf-8")
    return proposal


def approve(data_dir: Path, a_name: str, b_name: str, keep: list[str] | None = None, approver: str = "you",
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """Write the combined, trimmed Mirror exactly as proposed, sparing the trims you name (by number or id)."""
    root = proposal_dir(data_dir, a_name, b_name)
    proposal = _read(root / "proposal.json")
    if proposal is None:
        raise FileNotFoundError(f"no proposal for {combined_id(a_name, b_name)}; run transmutate propose first")
    a, b = load_source(data_dir, a_name), load_source(data_dir, b_name)
    for s in (a, b):
        if proposal["sources"][s["id"]]["hash"] != s["hash"]:
            raise RuntimeError(f"{s['id']} has changed since the proposal was made; propose again")
    spare = set(keep or [])
    known = {str(t["n"]) for t in proposal["trims"]} | {t["id"] for t in proposal["trims"]}
    if spare - known:
        raise KeyError(f"not in the proposal's trims: {', '.join(sorted(spare - known))}")
    cut = {t["id"] for t in proposal["trims"] if str(t["n"]) not in spare and t["id"] not in spare}
    combo = combine(a, b, proposal["alignment"])
    mirror, archive = apply_trims(combo["mirror"], cut)
    # a concept a spared item still needs is spared with it
    need = set(SHARED)
    for f in mirror["facts"]:
        need |= {f["s"], f["o"]}
    for j in mirror["judgments"]:
        need |= {j["x"], j["y"], j["z"]} - {""}
    for nid in list(archive["nodes"]):
        if nid in need:
            mirror["nodes"][nid] = archive["nodes"].pop(nid)
    record = {"approved": time.strftime("%Y-%m-%dT%H:%M:%S"), "approver": approver, "spared": sorted(spare),
              "trimmed": len(cut), "alignment": proposal["alignment"], "settings": proposal["settings"],
              "sources": proposal["sources"]}
    mirror["meta"]["transmutation"] = record
    cfg = mirror["causal"]["settings"]
    model = fit(mirror["judgments"], cfg)
    mirror["causal"] = {**public(model), "holdout": holdout(mirror["judgments"], cfg)}
    out = data_dir / "mirror" / combined_id(a_name, b_name)
    if (out / "mirror.json").exists():  # an earlier approval is archived, never overwritten
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = out / "history" / stamp
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("mirror.json", "passages.json", "trimmed.json"):
            if (out / name).exists():
                (out / name).replace(backup / name)
    _write(out / "passages.json", combo["passages"])
    _write(out / "trimmed.json", archive)
    _write(out / "mirror.json", mirror)
    _write(root / "approved.json", {**record, "proposal_created": proposal["created"]})
    st = mirror["causal"]["stats"]
    log(f"wrote {out}: {len(mirror['nodes'])} concepts, {len(mirror['facts'])} facts, {st['arrows']} arrows, "
        f"{st['rejected']} in tension; {len(cut)} items archived in trimmed.json")
    return mirror


# ---------- in words ----------


def _label(nodes: dict[str, Any], nid: str) -> str:
    return nodes.get(nid, {}).get("label", nid) if nid else ""


def _say(nodes: dict[str, Any], cid: str) -> str:
    t, x, y, *z = cid.split("|")
    lx, ly = _label(nodes, x), _label(nodes, y)
    if t == "reaches":
        return f"{lx} → {ly}"
    if t == "no_path":
        return f"{lx} makes no difference to {ly}"
    return f"{lx} acts on {_label(nodes, z[0])} only through {ly}"


def describe(p: dict[str, Any], nodes: dict[str, Any]) -> str:
    """The proposal in words, for the terminal and proposal.md."""
    s, c = p["stats"], p["crossings"]
    lines = [f"# Transmutation proposal: {p['a']} + {p['b']}", "",
             (f"Combined: {s['combined']['concepts']} concepts, {s['combined']['facts']} facts, "
              f"{s['combined']['judgments']} judgments, {s['combined']['arrows']} arrows, "
              f"{s['combined']['rejected']} in tension."),
             (f"After trimming: {s['trimmed']['concepts']} concepts, {s['trimmed']['facts']} facts, "
              f"{s['trimmed']['judgments']} judgments, {s['trimmed']['arrows']} arrows "
              f"({s['arrows_kept']} of the combined arrows kept), {s['trimmed']['rejected']} in tension."), ""]
    if s["arrows_new"]:
        lines += ["Arrows that only appear after trimming (a claim that followed through a weak path now stands alone): "
                  + ", ".join(_say(nodes, f"reaches|{e}") for e in s["arrows_new"]), ""]
    lines += [f"## Aligned concepts ({len(p['alignment'])})", ""]
    for x in p["alignment"]:
        rel = "is" if x["polarity"] == 1 else "is the opposite of"
        lines.append(f"- {p['b']}'s {x['b_label']} ({x['b']}) {rel} {p['a']}'s {x['a_label']} ({x['a']}) [{x['how']}]"
                     + (f"; kinds differ: {x['a_kind']} / {x['b_kind']}" if x["a_kind"] != x["b_kind"] else ""))
    lines += ["", "To drop a pair: propose again with --drop B_CONCEPT. To add one: --pair A=B or --opposite A=B.", "",
              f"## Both Mirrors say ({len(c['agreed'])})", ""]
    lines += [f"- {_say(nodes, x['id'])} (weight {x['weight']}, {x['status']}"
              + ("; they disagree on the sign)" if x["sign_conflict"] else ")") for x in c["agreed"][:30]]
    lines += ["", f"## Tensions that come from combining ({len(c['causal']) + len(c['kg']) + len(c['values'])})", ""]
    lines += [f"- {_say(nodes, x['id'])}: {x['kind']}; {x['why']}" for x in c["causal"]]
    lines += [f"- {x['type'].replace('_', ' ')}: {', '.join(x.get('facts', []))}" for x in c["kg"]]
    lines += [f"- {_label(nodes, x['concept'])}: urged " + ", ".join(f"{'more' if d > 0 else 'less'} by {k}"
                                                                      for k, d in x["by_source"].items()) for x in c["values"]]
    lines += ["", f"## Proposed trims ({len(p['trims'])})", "",
              "Spare any of these at approval with --keep N (repeatable). Trimmed items are archived, not deleted.", ""]
    for t in p["trims"]:
        kind, _, rest = t["id"].partition(":")
        if kind == "f":
            fs, fr, fo, *neg = rest.split("|")
            text = f"{_label(nodes, fs)} {fr} {_label(nodes, fo)}" + (" (denied)" if neg else "")
        else:
            text = _say(nodes, rest) if kind == "c" else _label(nodes, rest)
        lines.append(f"{t['n']}. [{t['what']}] {text}: {t['why']}")
    return "\n".join(lines) + "\n"


# ---------- command line ----------


def cli(args: Any, data_dir: Path) -> None:
    if args.action == "propose":
        add = [parse_pair(x) for x in args.pair or []] + [parse_pair(x, -1) for x in args.opposite or []]
        settings = {k: v for k, v in (("share", args.share), ("samples", args.samples), ("keep", args.keep_at))
                    if v is not None}
        try:
            propose(data_dir, args.a, args.b, args.claude, add, args.drop, settings)
        except (FileNotFoundError, RuntimeError, KeyError, ValueError, llm.LLMUnavailable) as e:
            raise SystemExit(str(e)) from None
        root = proposal_dir(data_dir, args.a, args.b)
        text = (root / "proposal.md").read_text(encoding="utf-8")
        print(text if len(text) < 6000 else text[:6000] + f"\n... (the rest is in {root / 'proposal.md'})")
        print(f"\nNothing has been changed. To accept: python -m androidmirror transmutate approve {args.a} {args.b}")
    else:
        try:
            approve(data_dir, args.a, args.b, args.keep)
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            raise SystemExit(str(e)) from None
