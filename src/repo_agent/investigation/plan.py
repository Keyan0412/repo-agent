from __future__ import annotations

from pydantic import BaseModel, Field

from .subtask import SubInvestigationTask


class AnalysisPlan(BaseModel):
    task_id: str
    goal: str
    subquestions: list[SubInvestigationTask] = Field(default_factory=list)
    synthesis_strategy: str
