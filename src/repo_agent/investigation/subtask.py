from __future__ import annotations

from pydantic import BaseModel, Field


class SubInvestigationTask(BaseModel):
    id: str
    parent_task_id: str
    question: str
    purpose: str
    expected_evidence: list[str] = Field(default_factory=list)
    known_information: str | None = None
    max_tool_calls: int = 4
    max_files: int = 3
