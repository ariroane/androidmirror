"""Where the Evolver gets its mutations: the refactor, break and chain engines.

Each engine exposes `propose(snapshot: GraphSnapshot) -> list[Patch]`, returning
candidate patches without queueing or applying them. The Evolver only ever calls
it on sandbox snapshots.
"""

from __future__ import annotations

import importlib
from typing import Callable

from ..ontology import GraphSnapshot
from ..patches import Patch

Operator = Callable[[GraphSnapshot], list[Patch]]

# operation family -> module that implements it
ENGINE_MODULES = {
    "refactor": "androidmirror.refactor",
    "break": "androidmirror.breaker",
    "chain": "androidmirror.chainer",
}


def discover() -> dict[str, Operator]:
    """The operation families whose engines are installed, by name."""
    found: dict[str, Operator] = {}
    for name, module in ENGINE_MODULES.items():
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        fn = getattr(mod, "propose", None)
        if callable(fn):
            found[name] = fn
    return found
