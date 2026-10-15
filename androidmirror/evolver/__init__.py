"""The Evolver: searches for ontology designs that better mirror the subject.

It mutates sandbox copies of both graphs with the operations the other engines
propose (refactor, break, chain), scores each resulting design against the
subject (probes, grounding, coherence, parsimony, robustness), and keeps the
best few. The winners are offered to the Steward as ordinary patches of kind
"evolve"; nothing reaches the live ontology until a person approves one.

The Evolver never writes the store and never approves anything.
"""

from .fitness import DEFAULT_WEIGHTS, Fitness, Probe, default_probes, load_probes, score
from .operators import Operator, discover
from .search import Design, evolve, queue_designs, sandbox

__all__ = [
    "DEFAULT_WEIGHTS",
    "Design",
    "Fitness",
    "Operator",
    "Probe",
    "default_probes",
    "discover",
    "evolve",
    "load_probes",
    "queue_designs",
    "sandbox",
    "score",
]
