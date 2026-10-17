"""Claude's three reading jobs in building a Mirror, each one structured call.

1. `code_passage` (once per passage): open coding. Concept mentions with their
   level in the passage, descriptive statements with modality and
   attribution, and causal claims with how their direction is grounded.
2. `consolidate` (a few calls over the whole vocabulary): fold the labels from
   step 1 into one codebook, merging synonyms and folding antonyms into one
   concept with a polarity ("freedom from anger" = anger at level -1).
3. `situations_passage` (once per passage): with the codebook fixed, find the
   situations the passage reasons about and the level (low, typical, high) of
   each variable the passage states for each, plus the places where the subject
   states its causal model outright. Situations, not passages, are the units of
   the subject's model (see causal/__init__.py).

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
        "causal_claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"cause": _STR, "effect": _STR, "sign": _LEVEL,
                           "direction_basis": {"type": "string", "enum": ["explicit", "temporal", "inferred"]},
                           "scope": {"type": "string", "enum": ["direct", "overall"]},
                           "strength_word": _STR, "modality": {"type": "string", "enum": list(MODALITIES)},
                           "attribution": _STR, "quote": _STR},
            "required": ["cause", "effect", "sign", "direction_basis", "scope", "strength_word", "modality",
                         "attribution", "quote"], "additionalProperties": False}},
    },
    "required": ["concepts", "statements", "causal_claims"],
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
   - Definitions and "X is a form of Y" are statements (is_a, part_of, requires), never causal claims.
   - subject and object are concept labels from your list, "{SELF}", or one of {", ".join(EVALUATIONS)}.
   - negated: true when the passage denies the fact.
   - modality: asserts (states it), prescribes (urges it, "thou must"), asks (rhetorical or open question), entertains (a hypothesis or one horn of a dilemma).
   - attribution: "{SELF}", or the name of the source the passage quotes or reports.
   - qualifier: extra detail a binary fact loses, e.g. for learned_from what was learned; else "".
   - wording: for related_to, the original verb phrase; else "".

3. causal_claims: what the passage says brings about, increases or reduces what. cause and effect are concept labels.
   - sign: 1 if more cause gives more effect, -1 if less.
   - direction_basis: explicit (a connective like because, so that, from this comes), temporal (one precedes the other), inferred (your reading).
   - scope: direct if the passage presents an immediate effect, overall if it is the end result of a chain.
   - strength_word: the quantifier the passage uses (always, necessarily, often, may), or "".

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


SITUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "situations": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "description": _STR,
                "kind": {"type": "string", "enum": ["actual", "hypothetical", "ideal", "general"]},
                "contrast_with": {"type": "integer"},
                "levels": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"variable": _STR, "level": {"type": "integer", "enum": [-1, 0, 1]}, "quote": _STR},
                    "required": ["variable", "level", "quote"], "additionalProperties": False}}},
            "required": ["description", "kind", "contrast_with", "levels"], "additionalProperties": False}},
        "model_claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["causes", "direct", "not_x_but_y", "whether_or_not", "so_that"]},
                           "x": _STR, "y": _STR, "z": _STR, "quote": _STR},
            "required": ["kind", "x", "y", "z", "quote"], "additionalProperties": False}},
    },
    "required": ["situations", "model_claims"],
    "additionalProperties": False,
}

SITUATION_SYSTEM = """You are reading one passage of a corpus about {subject} to recover the subject's own model of how things work: not what drives the writing, but the situations the subject thinks about and how things stand in them.

You get a fixed list of variables (id: label - meaning). Each can stand low (-1), typical (0) or high (1) in a situation.

1. situations: each concrete or imagined circumstance the passage reasons about (the morning with the meddlers, an insult, an illness, the hour of death). A passage may hold several; a purely general remark is one situation of kind "general".
   - description: one line, in neutral words, saying what the situation is. Do not state the variables' levels in it.
   - kind: actual (something that happened or happens), hypothetical (an "if" or "suppose"), ideal (the state the passage urges: "be free of anger" describes a situation where anger is low), general (how things stand in general).
   - contrast_with: a conditional or a contrast often describes two situations that differ in one respect ("take away the opinion, and the complaint is taken away": one situation with the opinion and the complaint, one without either). Give the index (0-based, within this passage) of the situation this one is contrasted with, or -1.
   - levels: only the variables whose level in this situation the passage states or plainly shows, each with the shortest verbatim quote that shows it. Leave out everything the passage leaves unsaid; most situations involve only a few variables. Never fill in a level from what you know about the subject or about the world.

2. model_claims: places where the subject states its model outright, using variable ids for x, y, z ("" when unused):
   - causes: x brings about or raises y ("from this comes").
   - direct: x acts on y directly, not through anything else.
   - not_x_but_y: "it is not x but y that does z" (x makes no difference to z once y is known).
   - whether_or_not: "whether x or not, y" (y does not depend on x).
   - so_that: "do x so that y" (x is a means to y).
   Quote each exactly. An empty list is fine.

Judge from this passage alone."""


def situations_passage(text: str, locator: str, subject: str, variables: list[dict[str, str]]) -> dict[str, Any]:
    system = SITUATION_SYSTEM.replace("{subject}", subject)
    listing = "\n".join(f"{v['id']}: {v['label']} - {v['gloss']}" for v in variables)
    user = f"Variables (id: label - meaning):\n{listing}\n\nPassage ({locator}):\n{text}"
    return llm._json_call(system, user, SITUATION_SCHEMA, effort="medium")


def prompt_version() -> str:
    """A short hash of the coding prompts, stored with every build (a change re-codes every passage)."""
    blob = json.dumps([CODE_SYSTEM, CODE_SCHEMA, CONSOLIDATE_SYSTEM], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def situation_version() -> str:
    """A short hash of the situations prompt, so a change re-reads the situations but not the coding."""
    return hashlib.sha256(json.dumps([SITUATION_SYSTEM, SITUATION_SCHEMA], sort_keys=True).encode()).hexdigest()[:12]
