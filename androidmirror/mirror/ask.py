"""The Ask tab: find what the Mirror's structures say about a question, then let Claude answer from them alone.

Claude sees only the knowledge graph and the causal model, never the corpus text: stated facts (with their
relation, how many passages state them and whether the subject holds them), inferred facts (with the rule and
the facts they follow from), the subject's causal judgments (with how many passages make them), the arrows of
the subject's model, the chains of arrows between the ideas in the question, two-way links, and what the
subject wants more or less of; and the OODA model built from the two (which stage of observe, orient, decide,
act each idea sits in and why, the links between stages, and what the shape of the loop shows). Each item has
a reference ([K3] for the knowledge graph, [C2] for the causal model, [O1] for the OODA model) that the answer
cites. The graph is open-world: what it lacks is "not found", never "the subject denies it".

A second step then reads the answer against the same items and hypothesizes a couple of ways to use it: which
idea to change (a lever), through which pathway of the subject's model, toward which outcome. These are
hypotheses from the subject's own model, not evidence that the world works that way.

"As the subject" mode (`ask_as_subject`) is the other way to ask: one call is given the whole of all three
structures, not just the items about the question, and is told to become the system they describe (a person, a
committee, a body of guidance) and answer the question as that system, without citing items. It still sees no
corpus text. If the structures are too big for one call, the least important items go first (facts that only
follow by logic, then views mentioned without endorsing, then the least supported stated facts) and the answer
says how many were left out.
"""

from __future__ import annotations

import os
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

SYSTEM = """You are the voice of the Android Mirror: a reflection of {subject}'s world built from the corpus "{title}". You do not see the corpus. You see only the Mirror's structures: its knowledge graph (items marked K), its causal model of how {subject} thinks things work (items marked C), and its OODA model (items marked O), which places {subject}'s ideas in the four stages of an observe-orient-decide-act loop and links the stages by {subject}'s own causal claims.

Answer the question from those items alone, in plain modern English. Do not quote or paraphrase the corpus, and do not add anything you know about {subject} from elsewhere.
- Cite the items you rely on by their reference in square brackets, e.g. [K3] or [C2], for every claim you make.
- Say how you know: "states" for stated facts, "it follows that" for inferred ones (name the rule and the facts it follows from if it helps), "wants more of" or "wants less of" for values.
- Use the relation types: "is a kind of", "is part of", "needs", "opposes" and the rest mean different things, so reason with them and say which you used.
- For cause and effect, separate what {subject} claims outright (a cause-and-effect claim) from what follows from a chain of their own links (a pathway), and say where their claims clash. A two-way link (x <-> y) means x and y go together because of a cause {subject} never names: changing x by itself does not change y. What {subject} wants is what they want, not how things work.
- The OODA stages are the Mirror's reading of the two graphs, not something {subject} said: say "the Mirror places X in Orient" and give the reason the item lists. A feedback link runs from Act back to an earlier stage and closes the loop; a link that skips a stage (Orient straight to Act) acts without a stated decision.
- If the items do not cover the question, say so. Missing is "not found", never "denied".
List the references you cited and the node ids you relied on."""

USES_SCHEMA = {
    "type": "object",
    "properties": {"uses": {"type": "array", "items": {
        "type": "object",
        "properties": {"lever": {"type": "string"}, "outcome": {"type": "string"}, "how": {"type": "string"},
                       "assumes": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}}},
        "required": ["lever", "outcome", "how", "assumes", "citations"],
        "additionalProperties": False}}},
    "required": ["uses"],
    "additionalProperties": False,
}

USES_SYSTEM = """You turn an answer from the Android Mirror into hypotheses for action. The Mirror reflects how {subject} sees the world, built from the corpus "{title}". You see only its knowledge graph (items marked K), its causal model of how {subject} thinks things work (items marked C), its OODA model placing {subject}'s ideas in the stages of an observe-orient-decide-act loop (items marked O), and an answer already written from them.

Propose two or three ways the answer could be used to drive real-world behavior or outcomes. For each:
- lever: the idea someone would change, nudge or act on (an idea that appears in the items).
- outcome: the result that should follow, and whether it should go up or down. Prefer outcomes {subject} wants more or less of, when the items say so.
- how: the route, step by step, through the causal model's links and pathways (and, where it helps, the OODA stages it passes through: what {subject} observes, how they orient, what they decide, how they act), using the knowledge graph's relations (kinds, parts, needs, opposes) to say what else the lever reaches. Be concrete about who would act and what they would do.
- assumes: what has to be true in the world for this to work, beginning with the fact that it rests on {subject}'s model. Name any clashing claims, two-way links (changing one side by itself does not move the other) or "makes no difference" claims that cut against it.
- citations: the item references it rests on.
Use only the items and the answer: no outside knowledge of {subject}, and no quoting the corpus. Do not use a two-way link as a lever, and do not route through a link the items mark as outweighed. If the items give no causal route, say so in "how" rather than inventing one."""

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


def context(m: Mirror, nodes: list[str], max_pathways: int = 12, question: str = "") -> list[str]:
    """Numbered items from the knowledge graph (K), the causal model (C) and the OODA model (O) about the matched
    ideas. No corpus text."""
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
    om = list(dict.fromkeys(ooda_items(m, nodes, question)))[:60]
    return ([f"[K{i}] {t}" for i, t in enumerate(kg, 1)] + [f"[C{i}] {t}" for i, t in enumerate(cm, 1)]
            + [f"[O{i}] {t}" for i, t in enumerate(om, 1)])


STAGE_WORDS = {"observe": ("observ", "notice", "perceiv"), "orient": ("orient", "interpret", "make sense"),
               "decide": ("decid", "decision", "choos", "choice"), "act": (" act ", " act?", " acts", "action", "acting")}
LOOP_WORDS = ("ooda", "loop", "stage", "tier")


def ooda_items(m: Mirror, nodes: list[str], question: str = "") -> list[str]:
    """Items from the OODA model: the stage of each matched idea and why, the links between stages that touch
    them, and, when the question is about the loop or a stage, that stage's ideas and the loop's shape."""
    from . import ooda

    md = ooda.model(m)
    lab = m.label
    q = f" {question.lower()} "
    asked = {t for t, ws in STAGE_WORDS.items() if any(w in q for w in ws)}
    whole = any(w in q for w in LOOP_WORDS)
    where = {c["id"]: c for t in md["tiers"] for c in t["concepts"]}
    out: list[str] = []
    for nid in nodes:
        c = where.get(nid)
        if not c:
            continue
        why = "; ".join(f"{ooda.LABEL[r['tier']]} +{r['points']}: {r['reason']}" for r in c["reasons"][:6])
        how = f"placed there by {c['by']}" if c["by"] else ("barely decided" if c["weak"] else "")
        out.append(f"stage: {lab(nid)} is placed in {ooda.LABEL[c['tier']]}" + (f" ({how})" if how else "")
                   + (f"; because {why}" if why else ""))
    for t in md["tiers"]:
        if whole or t["id"] in asked:
            names = ", ".join(c["label"] for c in t["concepts"][:12]) or "nothing yet"
            out.append(f"stage {t['label']} ({t['about']}): {names}")
    for l in md["links"]:
        if l["via"] == "fact":
            continue
        if l["src"] in nodes or l["dst"] in nodes or whole or l["from"] in asked or l["to"] in asked:
            kind = {"forward": "to the next stage", "within": "within a stage", "skip": "skipping a stage",
                    "feedback": "from Act back to an earlier stage, closing the loop", "back": "backward"}[l["kind"]]
            held = "; a claim that closes a cycle, so the causal model holds it aside" if l["via"] == "loop" else ""
            out.append(f"stage link, {l['weight']} passage(s): {l['text'].split(' (closes')[0]}, "
                       f"{ooda.LABEL[l['from']]} -> {ooda.LABEL[l['to']]}, {kind}{held}")
    if out or whole or asked:
        flows = [f"{ooda.LABEL[a]} -> {ooda.LABEL[b]} {n}" for a, row in md["flow"].items() for b, n in row.items() if n and a != b]
        if flows and (whole or asked):
            out.append("passages behind the links between stages: " + ", ".join(flows))
        out += [f"what the loop shows: {f}" for f in md["findings"]]
    return out


def pathways(m: Mirror, nodes: list[str], limit: int) -> list[str]:
    """Chains of two or more links in the subject's model: between the matched ideas, then onward from and into each."""
    lab = m.label
    vs = [n for n in nodes if n in m.variables]
    direct = {(e["src"], e["dst"]) for e in m.causal.get("edges", [])}
    pairs = [(x, y) for x in vs for y in vs if x != y]
    pairs += [(x, y) for x in vs for y in m.variables if y not in vs]  # what the matched ideas lead to
    pairs += [(y, x) for x in vs for y in m.variables if y not in vs]  # and what leads to them
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
    items = context(m, nodes, question=question)
    out: dict[str, Any] = {"question": question, "nodes": nodes, "facts": items, "answer": None, "citations": [],
                           "uses": [], "common_causes": shared_causes(m, nodes)}
    if not items:
        out["answer"] = "The Mirror has nothing on that yet."
        return out
    if llm.available():
        corpus = m.meta["corpus"]
        system = SYSTEM.replace("{subject}", corpus["subject"]).replace("{title}", corpus["title"])
        listed = "Items:\n" + "\n".join(items)
        r = llm._json_call(system, f"{listed}\n\nQuestion: {question}", ANSWER_SCHEMA, effort="low")
        out.update(answer=r["answer"], citations=r["citations"], nodes=r["nodes_used"] or nodes)
        out["uses"] = uses(m, items, question, r["answer"])
    return out


def uses(m: Mirror, items: list[str], question: str, answer: str) -> list[dict[str, Any]]:
    """The second step: a couple of hypotheses for acting on the answer, routed through the subject's model."""
    corpus = m.meta["corpus"]
    system = USES_SYSTEM.replace("{subject}", corpus["subject"]).replace("{title}", corpus["title"])
    user = "Items:\n" + "\n".join(items) + f"\n\nQuestion: {question}\n\nAnswer: {answer}"
    return llm._json_call(system, user, USES_SCHEMA, effort="low")["uses"][:3]


# ---------- as the subject ----------

SUBJECT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "ideas": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "ideas"],
    "additionalProperties": False,
}

SUBJECT_SYSTEM = """Below is the whole of the Android Mirror of {subject}, built from the corpus "{title}": its knowledge graph (what {subject} holds to be true, what follows from that, what they mention without endorsing, and what they want more or less of), its causal model (how {subject} thinks things work: their own cause-and-effect claims and the links fitted from them), and its OODA model (where {subject}'s ideas sit in a loop of observe, orient, decide and act, and how the stages connect).

Read all of it, then become the system it describes. {subject} may be a person, a group such as a committee, or a body of guidance; whatever it is, take on its beliefs, values, ways of reasoning about cause and effect, and the way it moves from noticing to acting, as one coherent whole. Then answer the question you are asked as {subject} would, in the first person, in plain modern English.

- Let the structures shape the answer from the inside. Do not list, number or cite them, and do not describe "the Mirror" or "the graph": you are {subject}, reasoning in your own way.
- Everything you believe comes from the structures below. Add nothing you know about {subject} from elsewhere, and do not quote or imitate the wording of the corpus, which you have not seen.
- Reason the way the structures do: follow your own cause-and-effect links, respect what you hold apart (a two-way link means two things go together because of something you never name, so changing one does not move the other), weigh what you want more or less of, and act through your own loop.
- Where the structures say nothing on a point, say you have not formed a view on it rather than inventing one. Where your own claims clash, let the clash show.

After the answer, list the ids of the ideas you drew on most (the ids in square brackets below).

{graphs}"""


def _limit() -> int:
    """Characters of structure to send in one call: about four characters a token, so the default is ~150k tokens."""
    try:
        return int(os.environ.get("ANDROIDMIRROR_SUBJECT_CHARS", "600000"))
    except ValueError:
        return 600000


def whole(m: Mirror, limit: int | None = None) -> tuple[str, dict[str, int]]:
    """The whole knowledge graph, causal model and OODA model as plain text, no corpus text. If it is longer than
    `limit` characters, the least important items are left out; returns the text and how many of each kind went."""
    from . import ooda

    limit = _limit() if limit is None else limit
    lab = m.label
    ideas = []
    for nid, n in sorted(m.nodes.items(), key=lambda kv: -kv[1].get("support", 0)):
        if nid.endswith("~"):
            continue
        ev = m.evaluation(nid)
        ideas.append(f"[{nid}] {lab(nid)} ({n.get('kind', '')}): {n.get('gloss', '')}" + (f"; evaluated {ev}" if ev else ""))
    held, unheld = [], []
    for f in sorted(m.facts, key=lambda f: -f["support"]):
        line = f"{lab(f['s'])} {'does not ' if f['neg'] else ''}{f['r'].replace('_', ' ')} {lab(f['o'])} ({f['support']} passage(s))"
        if f["held"]:
            held.append(line)
        else:
            e = f["evidence"][0] if f["evidence"] else {}
            unheld.append(f"{line}, {e.get('attribution', 'other')}, {e.get('modality', 'unknown')}")
    support = {nid: n.get("support", 0) for nid, n in m.nodes.items()}
    inferred = [f"{lab(s)} {r.replace('_', ' ')} {lab(o)} (by the rule '{p['rule']}')"
                for (s, r, o), p in sorted(m.kb.closure.items(), key=lambda kv: -(support.get(kv[0][0], 0) + support.get(kv[0][2], 0)))
                if p["rule"] != "stated"]
    values = [f"wants {'more' if v['more'] >= v['less'] else 'less'} of {lab(v['concept'])} "
              f"(more in {v['more']} passage(s), less in {v['less']})" for v in m.values]

    claims, links, two_way = [], [], []
    for c in sorted(m.causal.get("constraints", []), key=lambda c: -c["weight"]):
        if not c["held"]:
            continue
        state = {"arrow": "in the model", "follows": "covered by a chain in the model", "holds": "in the model",
                 "rejected": "clashes with stronger claims, read as a shared cause never named"
                 if c.get("common_cause") else f"outweighed by stronger claims ({c.get('reason')})"}.get(c["status"], c["status"])
        claims.append(f"{m.say(c)} ({c['weight']} passage(s), {state})")
    for e in m.causal.get("edges", []):
        note = f"; may work only by way of {' -> '.join(lab(n) for n in e['open']['through'])}" if e["open"] else ""
        links.append(f"{lab(e['src'])} {'raises' if e['sign'] > 0 else 'lowers'} {lab(e['dst'])}{note}")
    for b in m.causal.get("common_causes", []):
        if b["via"]:
            two_way.append(f"{lab(b['a'])} and {lab(b['b'])} go together because {lab(b['via'])} drives both")
        else:
            two_way.append(f"{lab(b['a'])} <-> {lab(b['b'])}: they go together, but neither drives the other")

    md = ooda.model(m)
    stages = []
    for t in md["tiers"]:
        names = ", ".join(c["label"] + (" (barely)" if c["weak"] else "") for c in t["concepts"]) or "nothing yet"
        stages.append(f"{t['label']} ({t['about']}): {names}")
    stage_links = [f"{l['text']}: {ooda.LABEL[l['from']]} -> {ooda.LABEL[l['to']]}, {l['kind']} ({l['weight']} passage(s))"
                   for l in md["links"] if l["via"] != "fact"]

    trimmable = {"inferred": inferred, "unendorsed": unheld, "stated": held}

    def render() -> str:
        part = lambda title, xs: f"## {title}\n" + ("\n".join(xs) if xs else "(none)")  # noqa: E731
        return "\n\n".join([
            "# KNOWLEDGE GRAPH", part("Ideas", ideas), part("Held as true", held),
            part("Follows from what is held", inferred), part("Mentioned without endorsing", unheld),
            part("Values", values),
            "# CAUSAL MODEL", part("Own cause-and-effect claims", claims), part("Links in the model", links),
            part("Two-way links (a cause never named)", two_way),
            "# OODA MODEL", part("Stages", stages), part("Links between stages", stage_links),
            part("What the loop shows", md["findings"])])

    text = render()
    dropped = {k: 0 for k in trimmable}
    for k, xs in trimmable.items():  # least important first; each list is sorted most important first
        while len(text) > limit and xs:
            over = len(text) - limit
            n = 0
            while xs and over > 0:
                over -= len(xs.pop()) + 1
                n += 1
            dropped[k] += n
            text = render()
    return text, dropped


def dropped_words(dropped: dict[str, int]) -> str:
    words = {"inferred": "facts that follow by logic", "unendorsed": "views mentioned without endorsing",
             "stated": "least supported facts held as true"}
    return ", ".join(f"{n} {words[k]}" for k, n in dropped.items() if n)


def ask_as_subject(m: Mirror, question: str) -> dict[str, Any]:
    """One call: become the system the three structures describe, then answer the question as it."""
    graphs, dropped = whole(m)
    out: dict[str, Any] = {"question": question, "mode": "subject", "answer": None, "nodes": [], "dropped": dropped,
                           "left_out": dropped_words(dropped), "size": len(graphs)}
    corpus = m.meta["corpus"]
    system = (SUBJECT_SYSTEM.replace("{subject}", corpus["subject"]).replace("{title}", corpus["title"])
              .replace("{graphs}", graphs))
    r = llm._json_call(system, question, SUBJECT_SCHEMA, effort="medium", cache_system=True)
    out.update(answer=r["answer"], nodes=[n for n in r["ideas"] if n in m.nodes])
    return out
