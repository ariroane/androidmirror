"""Claude's three reading jobs in building a Mirror, each one structured call.

1. `code_passage` (once per passage): open coding. Concept mentions with their
   level in the passage, and descriptive statements with modality and
   attribution. These build the knowledge graph.
2. `consolidate` (a few calls over the whole vocabulary): fold the labels from
   step 1 into one codebook, merging synonyms and folding antonyms into one
   concept with a polarity ("freedom from anger" = anger at level -1).
3. `judgments_passage` (once per passage): with the codebook fixed, the
   subject's causal judgments in the passage (what brings about what, the
   thought experiments, what makes no difference, what works only through
   what), and, apart from them, what the passage urges (the values layer).
   These build the causal layer (see judgments.py and causal/).

Each function has the same shape as its offline stand-in in the tests, so the
build pipeline can run without a key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .. import llm
from .schema import EVALUATIONS, KINDS, MODALITIES, RELATIONS, SELF

KIND_LIST = list(KINDS)
REL_LIST = list(RELATIONS)

_STR = {"type": "string"}
_LEVEL = {"type": "integer", "enum": [1, -1]}

CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "concepts": {"type": "array", "items": {
            "type": "object",
            "properties": {"label": _STR, "kind": {"type": "string", "enum": KIND_LIST}, "gloss": _STR,
                           "level": _LEVEL, "quote": _STR},
            "required": ["label", "kind", "gloss", "level", "quote"], "additionalProperties": False}},
        "statements": {"type": "array", "items": {
            "type": "object",
            "properties": {"subject": _STR, "relation": {"type": "string", "enum": REL_LIST}, "object": _STR,
                           "negated": {"type": "boolean"}, "modality": {"type": "string", "enum": list(MODALITIES)},
                           "attribution": _STR, "qualifier": _STR, "wording": _STR, "quote": _STR},
            "required": ["subject", "relation", "object", "negated", "modality", "attribution", "qualifier",
                         "wording", "quote"], "additionalProperties": False}},
    },
    "required": ["concepts", "statements"],
    "additionalProperties": False,
}

CODE_SYSTEM = f"""You are coding one passage of a corpus for the Android Mirror, a model of one subjective system's world (here: {{subject}}), built only from its own words. Code what the passage says, not what you know about the subject from elsewhere.

1. concepts: every idea, quality, feeling, practice, belief, state, faculty, person, place or class the passage engages.
   - label: short, in the corpus's own wording (singular, lower case, no article).
   - kind: one of {", ".join(KIND_LIST)}.
   - gloss: one line saying what the label means here.
   - level: 1 if the passage presents the concept as present, affirmed, high or to be had; -1 if as absent, negated, low or to be got rid of. ("Be free of anger" -> anger, -1.)
   - quote: the shortest verbatim span that shows it.
   The subject itself is the label "{SELF}" (kind agent); do not list it as a concept.

2. statements: descriptive facts, using only these relations:
{chr(10).join(f"   - {r}: {s['gloss']}" for r, s in RELATIONS.items())}
   - Evaluations are is_a with object "good", "bad" or "indifferent" ("pleasure is not a good" -> pleasure is_a good, negated).
   - Definitions and "X is a form of Y" are statements (is_a, part_of, requires).
   - subject and object are concept labels from your list, "{SELF}", or one of {", ".join(EVALUATIONS)}.
   - negated: true when the passage denies the fact.
   - modality: asserts (states it), prescribes (urges it, "thou must"), asks (rhetorical or open question), entertains (a hypothesis or one horn of a dilemma).
   - attribution: "{SELF}", or the name of the source the passage quotes or reports.
   - qualifier: extra detail a binary fact loses, e.g. for learned_from what was learned; else "".
   - wording: for related_to, the original verb phrase; else "".

Causal claims are read in a separate pass; here code only concepts and descriptive statements.

Quotes must be copied exactly from the passage. Prefer fewer, well-grounded items over speculative ones. An empty list is fine."""


def code_passage(text: str, locator: str, subject: str) -> dict[str, Any]:
    system = CODE_SYSTEM.replace("{subject}", subject)
    return llm._json_call(system, f"Passage ({locator}):\n{text}", CODE_SCHEMA, effort="medium")


CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"concepts": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": _STR, "label": _STR, "kind": {"type": "string", "enum": KIND_LIST}, "gloss": _STR,
                       "variable": {"type": "boolean"},
                       "members": {"type": "array", "items": {
                           "type": "object", "properties": {"label": _STR, "polarity": _LEVEL},
                           "required": ["label", "polarity"], "additionalProperties": False}}},
        "required": ["id", "label", "kind", "gloss", "variable", "members"], "additionalProperties": False}}},
    "required": ["concepts"],
    "additionalProperties": False,
}

CONSOLIDATE_SYSTEM = """You are building the codebook for the Android Mirror: one fixed list of concepts that a corpus about {subject} will be measured against.

You get concept labels found in the corpus, one per line: label | kind | number of passages | a gloss.
Group the labels that mean the same thing in this corpus into one concept:
- Merge synonyms and inflections. Do not merge on spelling alone ("reason" and "treason" differ).
- Fold opposites into one concept with polarity -1 for the opposite label ("freedom from anger" and "calm toward offenders" into anger at -1, if that is what they mean). Keep contraries that are distinct ideas apart (justice and injustice may be one concept; virtue and vice are two classes).
- Keep senses apart when one word has two meanings here (universal Nature versus a thing's own nature).
- id: snake_case of the label; label: the corpus's most common wording; gloss: one line.
- variable: true if the concept can be more or less present in a moment of the subject's life (a feeling, trait, practice, belief, state or use of a faculty), false for people, places, events, objects and abstract entities.
Every input label must appear in exactly one concept's members."""


def consolidate(lines: list[str], subject: str) -> list[dict[str, Any]]:
    system = CONSOLIDATE_SYSTEM.replace("{subject}", subject)
    return llm._json_call(system, "\n".join(lines), CONSOLIDATE_SCHEMA, effort="high")["concepts"]


KINDS_OF_JUDGMENT = ("causes", "intervention", "means_end", "screens_off", "independent")
REGISTERS = ("describes", "urges", "asks", "entertains")
HELD_REGISTERS = {"describes", "urges"}

JUDGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgments": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(KINDS_OF_JUDGMENT)},
                "x": _STR, "y": _STR, "z": _STR,
                "sign": {"type": "integer", "enum": [-1, 0, 1]},
                "scope": {"type": "string", "enum": ["direct", "overall", "unsaid"]},
                "register": {"type": "string", "enum": list(REGISTERS)},
                "attribution": _STR, "quote": _STR},
            "required": ["kind", "x", "y", "z", "sign", "scope", "register", "attribution", "quote"],
            "additionalProperties": False}},
        "values": {"type": "array", "items": {
            "type": "object",
            "properties": {"concept": _STR, "direction": _LEVEL, "quote": _STR},
            "required": ["concept", "direction", "quote"], "additionalProperties": False}},
    },
    "required": ["judgments", "values"],
    "additionalProperties": False,
}

JUDGMENT_SYSTEM = """You are reading one passage of a corpus by or about {subject} to recover {subject}'s own causal model: the judgments the passage makes about what brings about what. Record what the passage says, never what you know about the subject or about the world, and never a judgment the passage does not make.

Concepts (id: label - meaning). Use these ids for x, y and z; if no concept fits, give a short label in the corpus's own words:
{vocabulary}

1. judgments, each one of these kinds:
   - causes: x brings about, produces, increases or reduces y ("from this comes", "because of x, y", "x makes y"). sign 1 if more x gives more y, -1 if more x gives less y. scope: direct if the passage presents x acting on y immediately, overall if as the end of a chain, unsaid if it does not say.
   - intervention: a thought experiment. The passage imagines adding or removing x and says what then happens to y ("take away the opinion, and the complaint is taken away"; "if you remove x, y remains"). sign 1 if y moves with x, -1 if against it, 0 if y stays as it was.
   - means_end: x is a way to y ("do x so that y", "by x you will have y"). sign 1; -1 for "avoid x so that y" (x stands in the way of y).
   - screens_off: "it is not x but y that does z", "x affects z only through y". x and z, with y in between: y acts on z (sign is the sign of y on z), and x acts on z only through y or not at all.
   - independent: x makes no difference to y ("whether x or not, y", "x cannot harm y", "x does not touch y"). sign 0.
   Not judgments: definitions, classifications and evaluations ("anger is a vice", "death is a natural thing"), which are descriptive facts read elsewhere; and exhortations with no causal reason ("be free of anger"), which go under values.
   - register: describes (the passage states it), urges (it is part of an exhortation, e.g. "do x so that y"), asks (an open question; a rhetorical question that plainly asserts counts as describes), entertains (a hypothesis, one horn of a dilemma, or a view the passage tests and sets aside).
   - attribution: "{self}" for the subject's own view, or the name of the person, school or source the passage reports.
   - quote: the shortest verbatim span that shows the judgment.

2. values: what the passage urges, praises or holds up as the ideal, apart from any causal claim. concept: an id or label; direction 1 if it is to be had or increased, -1 if to be avoided or reduced; quote: verbatim.

Judge from this passage alone. Prefer fewer, well-grounded judgments. Empty lists are fine."""


def vocabulary(concepts: list[dict[str, str]]) -> str:
    return "\n".join(f"{c['id']}: {c['label']} - {c['gloss']}" for c in concepts)


def judgments_passage(text: str, locator: str, subject: str, concepts: list[dict[str, str]]) -> dict[str, Any]:
    """The causal judgments and values in one passage. The vocabulary is the same for every passage, so it goes in
    the (cached) system prompt."""
    system = JUDGMENT_SYSTEM.replace("{subject}", subject).replace("{self}", SELF).replace("{vocabulary}", vocabulary(concepts))
    return llm._json_call(system, f"Passage ({locator}):\n{text}", JUDGMENT_SCHEMA, effort="medium", cache_system=True)


def prompt_version() -> str:
    """A short hash of the coding prompts, stored with every build (a change re-codes every passage)."""
    blob = json.dumps([CODE_SYSTEM, CODE_SCHEMA, CONSOLIDATE_SYSTEM], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def judgment_version() -> str:
    """A short hash of the judgments prompt, so a change re-reads the judgments but not the coding."""
    return hashlib.sha256(json.dumps([JUDGMENT_SYSTEM, JUDGMENT_SCHEMA], sort_keys=True).encode()).hexdigest()[:12]
