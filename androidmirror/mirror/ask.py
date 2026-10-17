"""The Ask tab: find what the Mirror's structures say about a question, then let Claude answer from them alone.

Claude sees only the knowledge graph and the causal model, never the corpus text: stated facts (with their
relation, how many passages state them and whether the subject holds them), inferred facts (with the rule and
the facts they follow from), the subject's causal judgments (with how many passages make them), the arrows of
the subject's model, the chains of arrows between the ideas in the question, two-way links, and what the
subject wants more or less of. Each item has a reference ([K3] for the knowledge graph, [C2] for the causal
model) that the answer cites. The graph is open-world: what it lacks is "not found", never "the subject denies it".
"""

from __future__ import annotations

import re
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

SYSTEM = """You are the voice of the Android Mirror: a reflection of {subject}'s world built from the corpus "{title}". You do not see the corpus. You see only the Mirror's two structures: its knowledge graph (items marked K) and its causal model of how {subject} thinks things work (items marked C).

Answer the question from those items alone, in plain modern English. Do not quote or paraphrase the corpus, and do not add anything you know about {subject} from elsewhere.
- Cite the items you rely on by their reference in square brackets, e.g. [K3] or [C2], for every claim you make.
- Say how you know: "states" for stated facts, "it follows that" for inferred ones (name the rule and the facts it follows from if it helps), "wants more of" or "wants less of" for values.
- Use the relation types: "is a kind of", "is part of", "needs", "opposes" and the rest mean different things, so reason with them and say which you used.
- For cause and effect, separate what {subject} claims outright (a cause-and-effect claim) from what follows from a chain of their own links (a pathway), and say where their claims clash. A two-way link (x <-> y) means x and y go together because of a cause {subject} never names: changing x by itself does not change y. What {subject} wants is what they want, not how things work.
- If the items do not cover the question, say so. Missing is "not found", never "denied".
List the references you cited and the node ids you relied on."""

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


def context(m: Mirror, nodes: list[str], max_pathways: int = 8) -> list[str]:
    """Numbered items from the knowledge graph (K) and the causal model (C) about the matched ideas. No corpus text."""
    lab = m.label
    kg: list[str] = []
    cm: list[str] = []
    for nid in nodes:
        n = m.nodes[nid]
        ev = m.evaluation(nid)
        kg.append(f"idea: {lab(nid)} ({n['kind']}): {n.get('gloss', '')}" + (f"; evaluated {ev}" if ev else ""))
        for f in m.facts:
            if nid in (f["s"], f["o"]):
                verb = "does not " if f["neg"] else ""
                if f["held"]:
                    how = "stated as true"
                else:
                    e = f["evidence"][0] if f["evidence"] else {}
                    how = f"mentioned without endorsing ({e.get('attribution', 'other')}, {e.get('modality', 'unknown')})"
                kg.append(f"fact, {how}, {f['support']} passage(s): {lab(f['s'])} {verb}{f['r']} {lab(f['o'])}")
        for x in m.kb.query(s=nid)[:60] + m.kb.query(o=nid)[:60]:
            if x["status"] == "inferred":
                src = m.kb.closure[(x["s"], x["r"], x["o"])].get("from", [])
                because = "; ".join(f"{lab(a)} {r} {lab(b)}" for a, r, b in src)
                kg.append(f"follows by the rule '{x['rule']}': {lab(x['s'])} {x['r']} {lab(x['o'])}"
                          + (f" (from: {because})" if because else ""))
        v = m.value_of.get(nid)
        if v:
            kg.append(f"value: wants more of {lab(nid)} in {v['more']} passage(s), less in {v['less']}")
    for c in m.causal.get("constraints", []):
        if c["held"] and (c["x"] in nodes or c["y"] in nodes or c["z"] in nodes):
            state = {"arrow": "in the model", "follows": "already covered by a chain in the model", "holds": "in the model",
                     "rejected": "read as a shared cause the subject never names (it clashes with stronger claims)"
                     if c.get("common_cause") else f"outweighed by stronger claims ({c.get('reason')})"}.get(c["status"], c["status"])
            cm.append(f"cause-and-effect claim, {c['weight']} passage(s), {state}: {m.say(c)}")
    for e in m.causal.get("edges", []):
        if e["src"] in nodes or e["dst"] in nodes:
            note = f"; may work only by way of {' -> '.join(lab(n) for n in e['open']['through'])}" if e["open"] else ""
            cm.append(f"link in the model: {lab(e['src'])} {'raises' if e['sign'] > 0 else 'lowers'} {lab(e['dst'])}{note}")
    cm += pathways(m, nodes, max_pathways)
    for b in m.causal.get("common_causes", []):
        if b["a"] in nodes or b["b"] in nodes:
            if b["via"]:
                cm.append(f"two ideas with a shared cause, settled by the user: {lab(b['a'])} and {lab(b['b'])} go together "
                          f"because {lab(b['via'])} drives both; neither drives the other")
            else:
                cm.append(f"two-way link: {lab(b['a'])} <-> {lab(b['b'])}: they go together, but neither drives the other; "
                          "something the subject never names drives both"
                          + (" (settled by the user)" if b["settled"] else " (their claims about the two clash)"))
    kg, cm = list(dict.fromkeys(kg))[:100], list(dict.fromkeys(cm))[:80]
    return [f"[K{i}] {t}" for i, t in enumerate(kg, 1)] + [f"[C{i}] {t}" for i, t in enumerate(cm, 1)]


def pathways(m: Mirror, nodes: list[str], limit: int) -> list[str]:
    """Chains of two or more links in the subject's model: between the matched ideas, then onward from each."""
    lab = m.label
    vs = [n for n in nodes if n in m.variables]
    direct = {(e["src"], e["dst"]) for e in m.causal.get("edges", [])}
    pairs = [(x, y) for x in vs for y in vs if x != y]
    pairs += [(x, y) for x in vs for y in m.variables if y not in vs]
    out: list[str] = []
    for x, y in pairs:
        if (x, y) in direct:
            continue
        ex = m.explain(x, y)
        if not ex["paths"]:
            continue
        p = ex["paths"][0]
        more = f" ({len(ex['paths'])} chains in all)" if len(ex["paths"]) > 1 else ""
        out.append(f"pathway: {' -> '.join(p['labels'])}, so {lab(x)} {ex['verdict']} {lab(y)}{more}")
        if len(out) >= limit:
            break
    return out


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
    items = context(m, nodes)
    out: dict[str, Any] = {"question": question, "nodes": nodes, "facts": items, "answer": None, "citations": [],
                           "common_causes": shared_causes(m, nodes)}
    if not items:
        out["answer"] = "The Mirror has nothing on that yet."
        return out
    if llm.available():
        corpus = m.meta["corpus"]
        system = SYSTEM.replace("{subject}", corpus["subject"]).replace("{title}", corpus["title"])
        r = llm._json_call(system, "Items:\n" + "\n".join(items) + f"\n\nQuestion: {question}", ANSWER_SCHEMA, effort="low")
        out.update(answer=r["answer"], citations=r["citations"], nodes=r["nodes_used"] or nodes)
    return out
