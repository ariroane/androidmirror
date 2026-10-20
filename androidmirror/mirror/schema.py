"""The vocabulary the Mirror's descriptive ontology is written in.

The knowledge graph is built so that it can be reasoned over, not only browsed
(see CS520, "What are some inference algorithms?"): every node has a declared
kind, every relation has a declared domain, range and logical properties
(transitive, symmetric, inheritable down `is_a`), and the evaluation classes
`good`, `bad` and `indifferent` form a partition, so a thing the subject calls
both good and bad is a detectable inconsistency rather than two silent facts.

Nothing here is specific to one subject. The subject itself (a person, a group,
an institution, any subjective system) is the reserved node `self`.
"""

from __future__ import annotations

SELF = "self"

# What a thing is. Only some kinds can be causal variables: a variable must be
# something that can be more or less present in a passage (review finding #5).
KINDS = {
    "agent": "a person, group or other actor",
    "place": "a location",
    "event": "a dated happening",
    "object": "a physical thing",
    "disposition": "a character trait or habit of mind (virtues and vices are dispositions the subject evaluates)",
    "emotion": "a feeling or passion",
    "practice": "something the subject does or trains",
    "belief": "a judgment, opinion or impression",
    "state": "a condition the subject can be in (health, calm, distress)",
    "faculty": "a capacity of mind (reason, the ruling part)",
    "concept": "an abstract idea (nature, providence, death)",
    "class": "a category other things belong to (virtue, vice, passion)",
}
VARIABLE_KINDS = {"disposition", "emotion", "practice", "belief", "state", "faculty"}
# `concept` and `class` nodes may become variables only if the codebook marks them variable.

# The evaluation partition: built-in classes, pairwise disjoint.
EVALUATIONS = ("good", "bad", "indifferent")

MODALITIES = ("asserts", "prescribes", "asks", "entertains")
# Statements that count as the subject's own view, and so enter inference.
HELD_MODALITIES = {"asserts", "prescribes"}

# Relation inventory. domain/range: allowed kinds (None = any). family: the
# relation family the archetype matcher reads.
RELATIONS: dict[str, dict] = {
    "is_a": {"gloss": "is a kind of / is an instance of / is evaluated as (object may be good, bad, indifferent)",
             "transitive": True, "family": "PART"},
    "part_of": {"gloss": "is a part or component of", "transitive": True, "family": "PART"},
    "same_as": {"gloss": "is another name for", "symmetric": True, "transitive": True, "family": "PART"},
    "opposes": {"gloss": "is the contrary of / works against", "symmetric": True, "inheritable": True,
                "family": "OPPOSE"},
    "requires": {"gloss": "cannot exist or happen without", "transitive": True, "inheritable": True,
                 "family": "PART"},
    "accords_with": {"gloss": "is in accordance with", "inheritable": True, "family": "EMBODY"},
    "expresses": {"gloss": "shows itself as / is a sign of", "family": "EMBODY"},
    "exemplifies": {"gloss": "(an agent) displays or embodies", "domain": {"agent"}, "family": "EMBODY"},
    "practices": {"gloss": "(an agent) carries out", "domain": {"agent"}, "family": "EMBODY"},
    "learned_from": {"gloss": "(an agent) learned something from (an agent); qualifier: what",
                     "domain": {"agent"}, "family": "LINEAGE"},
    "values": {"gloss": "(an agent) holds dear, aims at", "domain": {"agent"}, "family": "COMMIT",
               "down_object": True},
    "rejects": {"gloss": "(an agent) repudiates, avoids", "domain": {"agent"}, "family": "OPPOSE",
                "down_object": True},
    "related_to": {"gloss": "escape hatch: any other link; keep the original wording", "family": "ASSOCIATE"},
}


def check_fact(rel: str, src_kind: str, dst_kind: str, dst: str) -> str | None:
    """A domain/range violation message, or None if the fact is well-formed."""
    spec = RELATIONS.get(rel)
    if spec is None:
        return f"unknown relation {rel!r}"
    if spec.get("domain") and src_kind not in spec["domain"]:
        return f"{rel} needs a {'/'.join(sorted(spec['domain']))} as subject, got {src_kind}"
    if spec.get("range") and dst_kind not in spec["range"]:
        return f"{rel} needs a {'/'.join(sorted(spec['range']))} as object, got {dst_kind}"
    if dst in EVALUATIONS and rel not in ("is_a", "values", "rejects"):
        return f"only is_a, values and rejects may point at the evaluation {dst!r}"
    return None


# Kind (+ evaluation) -> the Prototype 1 node type the archetype matcher reads.
def archetype_type(kind: str, evaluation: str | None) -> str:
    if kind == "agent":
        return "person"
    if kind in ("place", "event", "emotion", "practice", "belief"):
        return kind
    if kind == "disposition":
        return {"good": "virtue", "bad": "vice"}.get(evaluation or "", "concept")
    if evaluation == "good":
        return "value"
    if evaluation == "bad" and kind == "state":
        return "emotion"
    return "concept"
