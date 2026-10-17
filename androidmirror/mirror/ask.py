"""The Ask tab: find what the Mirror knows about a question, then let Claude answer with citations.

The Oracle gets its context marked so it can say how it knows: stated facts
(with a quote and locator), inferred facts (with the rule that derived them),
the subject's causal judgments (with how many passages make them), the arrows
of the subject's model they add up to (and the judgments in tension with it),
what the subject urges, and the passages most about the question. The graph is open-world: what it
lacks is "not found", never "the subject denies it".
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .. import llm
from .store import Mirror

STOP = set("a an the of to in on and or is are was were what why how who does do did his her he she it about "
           "that this with for from by as be has have their they them its which when would could should".split())

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}},
                   "nodes_used": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "citations", "nodes_used"],
    "additionalProperties": False,
}

SYSTEM = """You are the voice of the Android Mirror: a reflection of {subject}'s world built only from the context below, which comes from the corpus "{title}".

Answer the question from that context, in plain modern English.
- Cite passages by their locator in square brackets, e.g. [Book II, 1], for every claim you make.
- Say how you know: "states" for stated facts, "it follows that" for inferred ones (name the rule if it helps), "urges" for prescriptions.
- For causal questions, separate what the subject says outright from what follows from their model (a chain of their own judgments), and say where their judgments are in tension. A two-way link (x <-> y) means x and y go together because of a cause the subject never names: changing x by itself does not change y. What the subject urges is what they want, not how things work.
- If the context does not cover the question, say so. Missing is "not found", never "denied".
List the locators you cited and the node ids you relied on."""


def match_nodes(m: Mirror, question: str, limit: int = 6) -> list[str]:
    words = {w for w in re.findall(r"[a-z]+", question.lower()) if w not in STOP and len(w) > 2}
    scored = []
    for nid, n in m.nodes.items():
        if nid.endswith("~"):
            continue
        names = " ".join([n.get("label", "")] + n.get("aliases", []))
        toks = set(re.findall(r"[a-z]+", names.lower()))
        hits = sum(1 for w in words if any(t.startswith(w[:5]) for t in toks))
        if hits:
            scored.append((hits, n.get("support", 0), nid))
    return [nid for *_, nid in sorted(scored, reverse=True)[:limit]]


def context(m: Mirror, nodes: list[str], max_passages: int = 4) -> dict[str, Any]:
    lab = m.label
    lines: list[str] = []
    for nid in nodes:
        n = m.nodes[nid]
        ev = m.evaluation(nid)
        lines.append(f"[concept] {lab(nid)} ({n['kind']}): {n.get('gloss', '')}" + (f"; evaluated {ev}" if ev else ""))
        for f in m.facts:
            if nid in (f["s"], f["o"]) and f["evidence"]:
                e = f["evidence"][0]
                verb = "does not " if f["neg"] else ""
                how = "held" if f["held"] else f"reported ({e.get('attribution')}, {e.get('modality')})"
                lines.append(f"[stated, {how}, {f['support']} passage(s)] {lab(f['s'])} {verb}{f['r']} {lab(f['o'])}"
                             f" [{e['locator']}] \"{e['quote']}\"")
        for x in m.kb.query(s=nid)[:60] + m.kb.query(o=nid)[:60]:
            if x["status"] == "inferred":
                lines.append(f"[inferred: {x['rule']}] {lab(x['s'])} {x['r']} {lab(x['o'])}")
        v = m.value_of.get(nid)
        if v:
            lines.append(f"[urges, {v['more']} passage(s) for more, {v['less']} for less] {lab(nid)}")
    for c in m.causal.get("constraints", []):
        if c["held"] and (c["x"] in nodes or c["y"] in nodes or c["z"] in nodes) and c["evidence"]:
            e = next((x for x in c["evidence"] if x["held"]), c["evidence"][0])
            state = {"arrow": "in the model", "follows": "follows from the model", "holds": "in the model",
                     "rejected": "read as a shared cause the subject never names (it clashes with stronger judgments)"
                     if c.get("common_cause") else f"outweighed by stronger judgments ({c.get('reason')})"}.get(c["status"], c["status"])
            lines.append(f"[causal judgment, {c['weight']} passage(s), {state}] {m.say(c)} [{e['locator']}] \"{e['quote']}\"")
    for e in m.causal.get("edges", []):
        if e["src"] in nodes or e["dst"] in nodes:
            note = f"; may run only through {' -> '.join(lab(n) for n in e['open']['through'])}" if e["open"] else ""
            lines.append(f"[subject's model] {lab(e['src'])} {'raises' if e['sign'] > 0 else 'lowers'} {lab(e['dst'])}{note}")
    for b in m.causal.get("common_causes", []):
        if b["a"] in nodes or b["b"] in nodes:
            if b["via"]:
                lines.append(f"[subject's model, settled by the user] {lab(b['a'])} and {lab(b['b'])} go together because "
                             f"{lab(b['via'])} drives both; neither drives the other")
            else:
                lines.append(f"[subject's model] {lab(b['a'])} <-> {lab(b['b'])}: they go together, but neither drives the "
                             "other; something the subject never names drives both"
                             + (" (settled by the user)" if b["settled"] else " (their claims about the two clash)"))
    lines = list(dict.fromkeys(lines))
    counts: Counter = Counter()
    for nid in nodes:
        for mention in m.nodes[nid].get("mentions", []):
            counts[mention["passage"]] += 1
    passages = [m.passages[pid] for pid, _ in counts.most_common(max_passages) if pid in m.passages]
    return {"lines": lines[:150], "passages": [{"id": p["id"], "locator": p["locator"], "text": p["text"][:1200]}
                                               for p in passages]}


def shared_causes(m: Mirror, nodes: list[str]) -> list[dict[str, Any]]:
    """The two-way links the answer touches, each with the Question engine's question that can settle it."""
    from .question import _qid  # the question about a clashing claim is keyed by the claim's id

    out = []
    for b in m.causal.get("common_causes", []):
        if b["via"] or not ({b["a"], b["b"]} & set(nodes)):
            continue
        claims = [m.constraint[i] for i in b["claims"] if i in m.constraint]
        out.append({"id": b["id"], "a_label": m.label(b["a"]), "b_label": m.label(b["b"]), "settled": b["settled"],
                    "claims": [m.say(c) for c in claims],
                    "questions": [] if b["settled"] else [_qid("tension", c["id"]) for c in claims]})
    return out


def ask(m: Mirror, question: str) -> dict[str, Any]:
    nodes = match_nodes(m, question)
    ctx = context(m, nodes)
    out: dict[str, Any] = {"question": question, "nodes": nodes, "facts": ctx["lines"],
                           "passages": [{"id": p["id"], "locator": p["locator"]} for p in ctx["passages"]],
                           "answer": None, "citations": [], "common_causes": shared_causes(m, nodes)}
    if not ctx["lines"]:
        out["answer"] = "The Mirror has nothing on that yet."
        return out
    if llm.available():
        corpus = m.meta["corpus"]
        system = SYSTEM.replace("{subject}", corpus["subject"]).replace("{title}", corpus["title"])
        text = "\n".join(ctx["lines"]) + "\n\nPassages:\n" + "\n\n".join(
            f"[{p['locator']}] {p['text']}" for p in ctx["passages"])
        r = llm._json_call(system, f"Context:\n{text}\n\nQuestion: {question}", ANSWER_SCHEMA, effort="low")
        out.update(answer=r["answer"], citations=r["citations"], nodes=r["nodes_used"] or nodes)
    return out
