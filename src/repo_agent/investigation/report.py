from __future__ import annotations

from pydantic import BaseModel, Field

from .observation import Observation
from .subreport import SubInvestigationReport


class InvestigationReport(BaseModel):
    id: str
    task_id: str
    summary: str
    observations: list[Observation] = Field(default_factory=list)
    files_checked: list[str] = Field(default_factory=list)
    remaining_questions: list[str] = Field(default_factory=list)
    subreports: list[SubInvestigationReport] = Field(default_factory=list)
    profile_update_summary: str | None = None
