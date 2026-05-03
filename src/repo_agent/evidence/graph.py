from __future__ import annotations

from enum import StrEnum

from .node import EvidenceNode


class EvidenceStatus(StrEnum):
    TOP_LEVEL = "top_level"
    ABSORBED = "absorbed"


class EvidenceGraph:
    def __init__(self) -> None:
        self.nodes: list[EvidenceNode] = []
        self.children: list[set[int]] = []

    def add(self, node: EvidenceNode) -> int:
        self._validate_based_on(node.based_on)

        evidence_id = len(self.nodes)
        self.nodes.append(node)
        self.children.append(set())

        for parent_id in node.based_on:
            self.children[parent_id].add(evidence_id)

        return evidence_id

    def get(self, evidence_id: int) -> EvidenceNode:
        return self.nodes[self._validate_id(evidence_id)]

    def get_many(self, ids: list[int]) -> list[EvidenceNode]:
        return [self.get(evidence_id) for evidence_id in ids]

    def get_top_level_ids(self) -> list[int]:
        return [
            evidence_id
            for evidence_id, child_ids in enumerate(self.children)
            if not child_ids
        ]

    def get_status(self, evidence_id: int) -> EvidenceStatus:
        validated_id = self._validate_id(evidence_id)
        if self.children[validated_id]:
            return EvidenceStatus.ABSORBED
        return EvidenceStatus.TOP_LEVEL

    def expand_chain(self, evidence_id: int) -> list[int]:
        validated_id = self._validate_id(evidence_id)
        expanded: set[int] = set()
        self._collect_ancestors(validated_id, expanded)
        expanded.add(validated_id)
        return sorted(expanded)

    def expand_many(self, ids: list[int]) -> list[int]:
        expanded: set[int] = set()
        for evidence_id in ids:
            expanded.update(self.expand_chain(evidence_id))
        return sorted(expanded)

    def _collect_ancestors(self, evidence_id: int, expanded: set[int]) -> None:
        for parent_id in self.nodes[evidence_id].based_on:
            if parent_id in expanded:
                continue
            expanded.add(parent_id)
            self._collect_ancestors(parent_id, expanded)

    def _validate_based_on(self, based_on: list[int]) -> None:
        if len(based_on) != len(set(based_on)):
            raise ValueError("based_on contains duplicate evidence ids")

        for evidence_id in based_on:
            self._validate_id(evidence_id)

    def _validate_id(self, evidence_id: int) -> int:
        if evidence_id < 0 or evidence_id >= len(self.nodes):
            raise ValueError(f"invalid evidence id: {evidence_id}")
        return evidence_id
