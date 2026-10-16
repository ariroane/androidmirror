"""The Mirror's Claude agents.

Each agent is one Claude call with a fixed role and a JSON schema for its
answer (structured outputs), so what comes back can be turned straight into
patches or answers:

- Cartographer + Causal Modeler (one call): read a chunk, propose descriptive
  and causal ops grounded in a verbatim quote.
- Oracle: answer a question from the subgraph the query layer found.
- Storyteller: retell what an intervention on the causal DAG changes.

The key is read from the local .env (ANTHROPIC_API_KEY) and never leaves the
machine the app runs on. Without a key the app still works for everything that
does not need Claude: browsing, approving, analysis and interventions.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("ANDROIDMIRROR_MODEL", "claude-opus-5-5")


class LLMTruncated(RuntimeError):
    """The answer was cut off or malformed; a smaller request usually succeeds."""


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


NODE_TYPES = ["person", "virtue", "vice", "emotion", "belief", "practice", "value", "concept", "event", "place"]

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["descriptive", "causal"]},
                    "quote": {"type": "string"},
                    "rationale": {"type": "string"},
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "type": {"type": "string", "enum": NODE_TYPES},
                            },
                            "required": ["id", "label", "type"],
                            "additionalProperties": False,
                        },
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "src": {"type": "string"},
                                "rel": {"type": "string"},
                                "dst": {"type": "string"},
                            },
                            "required": ["src", "rel", "dst"],
                            "additionalProperties": False,
                        },
                    },
                    "causal_edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "src": {"type": "string"},
                                "dst": {"type": "string"},
                                "sign": {"type": "integer", "enum": [1, -1]},
                                "strength": {"type": "number"},
                                "confidence": {"type": "number"},
                                "mechanism": {"type": "string"},
                            },
                            "required": ["src", "dst", "sign", "strength", "confidence", "mechanism"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["kind", "quote", "rationale", "nodes", "relations", "causal_edges"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["patches"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """You are the Cartographer and Causal Modeler of the Android Mirror, a reflective model of one author's subjective world built from their own writing.

Read the passage and propose small patches to two linked ontologies:
- Descriptive (knowledge graph): the people, virtues, vices, emotions, beliefs, practices and values in the author's world, and how they relate (learned_from, values, rejects, part_of, exemplifies, opposes, ...). Relation names are short snake_case verbs.
- Causal (DAG): what the author believes drives what. sign 1 means "more of src leads to more of dst", -1 means "leads to less". strength and confidence are 0 to 1. mechanism is one short clause in the author's terms.

Rules:
- One patch per distinct idea, each grounded in a verbatim quote copied exactly from the passage (one sentence or less where possible).
- Reuse existing node ids from the list you are given whenever the meaning matches; only add new nodes when needed, and list every node a patch touches that does not exist yet in its "nodes".
- Node ids are snake_case slugs of the label.
- Model the author's view, not your own. Prefer fewer, well-grounded patches over many speculative ones."""


def _client():
    if not available():
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set; add it to your local .env")
    import anthropic

    return anthropic.Anthropic()


def _json_call(system: str, user: str, schema: dict[str, Any], effort: str = "medium",
               max_tokens: int = 16000) -> dict[str, Any]:
    client = _client()
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        # Server-side fallback: if a request is declined, the API retries it on
        # Anthropic's recommended fallback model instead of returning a refusal.
        extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
        extra_body={"fallbacks": "default"},
    )
    if response.stop_reason == "refusal":
        raise LLMUnavailable("Claude declined this request")
    if response.stop_reason == "max_tokens":
        raise LLMTruncated(f"Claude's answer hit the {max_tokens}-token limit before the JSON was complete")
    text = next(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMTruncated(f"Claude's answer was not complete JSON: {e}") from e


def extract(passage: str, locator: str, known_nodes: str) -> list[dict[str, Any]]:
    user = f"Known nodes (id: label (type)):\n{known_nodes}\n\nPassage ({locator}):\n{passage}"
    return _json_call(EXTRACT_SYSTEM, user, EXTRACT_SCHEMA)["patches"]


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "nodes_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "nodes_used"],
    "additionalProperties": False,
}


def answer(question: str, context: str) -> dict[str, Any]:
    system = (
        "You are the voice of the Android Mirror: a reflection of one author's subjective world, "
        "built only from the ontology facts you are given. Answer the question from those facts, "
        "in plain modern English, and say when the ontology does not cover something. "
        "List the node ids you relied on."
    )
    return _json_call(system, f"Ontology facts:\n{context}\n\nQuestion: {question}", ANSWER_SCHEMA, effort="low")


STORY_SCHEMA = {
    "type": "object",
    "properties": {"story": {"type": "string"}},
    "required": ["story"],
    "additionalProperties": False,
}


def retell(intervention: str, effects: str, context: str) -> str:
    system = (
        "You are the Storyteller of the Android Mirror. Given an intervention on the author's causal "
        "model and its computed effects, write a short first-person passage (under 200 words) in the "
        "author's voice describing how their inner life would change. Stay faithful to the effects given."
    )
    user = f"Ontology facts:\n{context}\n\nIntervention: {intervention}\n\nComputed effects:\n{effects}"
    return _json_call(system, user, STORY_SCHEMA, effort="low")["story"]
