"""The Steward: governance for every change to the ontology.

Agents propose patches; the Steward checks them, holds them in a queue, and
applies one only when a person approves it. Every applied patch keeps its
approval record and its inverse, so it can be rolled back.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .ontology import Ontology, OntologyError, slugify
from .patches import Approval, Patch


class GovernanceError(ValueError):
    pass


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


class Steward:
    def __init__(self, ontology: Ontology, data_dir: Path, corpus_text: str | None = None) -> None:
        self.ontology = ontology
        self.data_dir = data_dir
        self.patches_path = data_dir / "patches.json"
        self.ontology_path = data_dir / "ontology.json"
        self.corpus_norm = normalize(corpus_text) if corpus_text else None
        self.patches: dict[str, Patch] = {}
        if self.patches_path.exists():
            for raw in json.loads(self.patches_path.read_text(encoding="utf-8")):
                p = Patch.model_validate(raw)
                self.patches[p.id] = p

    # ---------- queue ----------

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.patches_path.write_text(
            json.dumps([p.model_dump() for p in self.patches.values()], indent=2), encoding="utf-8"
        )
        self.ontology.save(self.ontology_path)

    def list(self, status: str | None = None) -> list[Patch]:
        return [p for p in self.patches.values() if status is None or p.status == status]

    def get(self, patch_id: str) -> Patch:
        if patch_id not in self.patches:
            raise GovernanceError(f"unknown patch: {patch_id}")
        return self.patches[patch_id]

    # ---------- propose ----------

    def propose(self, patch: Patch) -> Patch:
        """Dedupe against existing nodes, check the quote, and queue the patch."""
        self._dedupe(patch)
        patch.base_version = self.ontology.version if patch.base_version is None else patch.base_version
        quote = patch.source.quote.strip()
        if quote and self.corpus_norm is not None and normalize(quote) not in self.corpus_norm:
            patch.warnings.append("quote not found verbatim in the corpus")
        if not quote:
            patch.warnings.append("no source quote")
        try:
            self.ontology.check_ops(patch.ops)
        except OntologyError as e:
            patch.warnings.append(f"would not apply right now: {e}")
        patch.status = "proposed"
        self.patches[patch.id] = patch
        return patch

    def _dedupe(self, patch: Patch) -> None:
        """Point ops at existing nodes when an added node matches one by label."""
        renames: dict[str, str] = {}
        kept = []
        for op in patch.ops:
            if op.op == "add_node":
                nid = op.id or slugify(op.label or "")
                existing = self.ontology.find_node(op.label or nid)
                if existing:
                    renames[nid] = existing
                    continue  # node already exists: drop the add
                op.id = nid
            kept.append(op)
        for op in kept:
            if op.src in renames:
                op.src = renames[op.src]
            if op.dst in renames:
                op.dst = renames[op.dst]
            if op.id in renames:
                op.id = renames[op.id]
        if renames:
            patch.warnings.append("merged with existing nodes: " + ", ".join(f"{a}->{b}" for a, b in renames.items()))
        patch.ops = kept

    # ---------- decide ----------

    def approve(self, patch_id: str, approver: str, note: str = "") -> Patch:
        if not approver.strip():
            raise GovernanceError("an approval needs a named approver")
        patch = self.get(patch_id)
        if patch.status != "proposed":
            raise GovernanceError(f"patch {patch_id} is {patch.status}, not proposed")
        self._dedupe(patch)  # nodes another approved patch has added since this one was proposed
        if patch.extra.get("require_fresh_base"):
            self._check_fresh(patch)
        provenance = {"patch": patch.id, "locator": patch.source.locator, "quote": patch.source.quote}
        try:
            patch.inverse = self.ontology.apply_ops(patch.ops, patch_id=patch.id, provenance=provenance)
        except OntologyError as e:
            raise GovernanceError(f"cannot apply {patch_id}: {e}") from e
        patch.approval = Approval(approver=approver, note=note)
        patch.status = "approved"
        patch.applied_version = self.ontology.version
        self.save()
        return patch

    def _check_fresh(self, patch: Patch) -> None:
        """Refuse a patch if any node it touches changed after its base_version."""
        base = patch.base_version or 0
        for op in patch.ops:
            for nid in (op.id, op.src, op.dst):
                node = self.ontology.nodes.get(nid or "")
                if node and node.get("modified", 0) > base:
                    raise GovernanceError(
                        f"stale patch {patch.id}: {nid} changed in version {node['modified']}, after base {base}"
                    )

    def reject(self, patch_id: str, approver: str, note: str = "") -> Patch:
        patch = self.get(patch_id)
        if patch.status != "proposed":
            raise GovernanceError(f"patch {patch_id} is {patch.status}, not proposed")
        patch.status = "rejected"
        patch.approval = Approval(approver=approver, note=note or "rejected")
        self.save()
        return patch

    def rollback(self, patch_id: str, approver: str) -> Patch:
        patch = self.get(patch_id)
        if patch.status != "approved":
            raise GovernanceError(f"patch {patch_id} is {patch.status}, not approved")
        try:
            self.ontology.apply_ops(patch.inverse, patch_id=f"rollback:{patch.id}")
        except OntologyError as e:
            raise GovernanceError(f"cannot roll back {patch_id} (later patches depend on it): {e}") from e
        patch.status = "rolled_back"
        patch.extra["rolled_back_by"] = approver
        self.save()
        return patch
