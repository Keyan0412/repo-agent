from __future__ import annotations

from pydantic import BaseModel, Field

from .observation import Observation


class InvestigationReport(BaseModel):
    id: str
    task_id: str
    summary: str
    observations: list[Observation] = Field(default_factory=list)
    files_checked: list[str] = Field(default_factory=list)
    remaining_questions: list[str] = Field(default_factory=list)
