"""The Ask tab: find what the Mirror knows about a question, then let Claude answer with citations.

The Oracle gets four kinds of context, each marked so it can say how it knows:
stated facts (with a quote and locator), inferred facts (with the rule that
derived them), the subject's causal claims, and what the causal model says
about the same variables (compelled, reversible, or not linked in the data),
plus the passages most about the question. The graph is open-world: what it
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
- For causal questions, separate what the subject claims from what the corpus as data supports (compelled, reversible, or no link once other variables are accounted for).
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
        for c in m.claims:
            if nid in (c["cause"], c["effect"]):
                e = c["evidence"][0]
                verb = "raises" if c["sign"] > 0 else "lowers"
                lines.append(f"[causal claim, {c['support']} passage(s)] {lab(c['cause'])} {verb} {lab(c['effect'])}"
                             f" [{e['locator']}] \"{e['quote']}\"")
    for e in m.causal.get("edges", []):
        if e["a"] in nodes or e["b"] in nodes:
            arrow = f"{lab(e['a'])} -> {lab(e['b'])}" if e["directed"] else f"{lab(e['a'])} - {lab(e['b'])}"
            lines.append(f"[causal model: {e['status']}] {arrow}; adjusted effect {e['effect']}, "
                         f"stable in {e['stability'].get('adjacent', '?')} of resamples")
    for p in m.causal.get("pins", []):
        if p["status"] == "no link in the data" and (p["src"] in nodes or p["dst"] in nodes):
            lines.append(f"[causal model] the claim {lab(p['src'])} -> {lab(p['dst'])} has no link in the data")
    lines = list(dict.fromkeys(lines))
    counts: Counter = Counter()
    for nid in nodes:
        for mention in m.nodes[nid].get("mentions", []):
            counts[mention["passage"]] += 1
    passages = [m.passages[pid] for pid, _ in counts.most_common(max_passages) if pid in m.passages]
    return {"lines": lines[:150], "passages": [{"id": p["id"], "locator": p["locator"], "text": p["text"][:1200]}
                                               for p in passages]}


def ask(m: Mirror, question: str) -> dict[str, Any]:
    nodes = match_nodes(m, question)
    ctx = context(m, nodes)
    out: dict[str, Any] = {"question": question, "nodes": nodes, "facts": ctx["lines"],
                           "passages": [{"id": p["id"], "locator": p["locator"]} for p in ctx["passages"]],
                           "answer": None, "citations": []}
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
