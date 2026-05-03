from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvidenceType = Literal[
    "investigate_result",
    "derived_claim",
    "final_claim",
]


class EvidenceNode(BaseModel):
    type: EvidenceType
    claim: str
    based_on: list[int] = Field(default_factory=list)
    confidence: float | None = None
    limitations: list[str] = Field(default_factory=list)
    source_report_ids: list[int] = Field(default_factory=list)
