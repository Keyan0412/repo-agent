from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .observation import Observation


class SubInvestigationReport(BaseModel):
    id: str
    parent_task_id: str
    subtask_id: str
    question: str
    answer: str
    confidence: Literal["high", "medium", "low"]
    observations: list[Observation] = Field(default_factory=list)
    files_checked: list[str] = Field(default_factory=list)
    symbols_checked: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    profile_update_suggestion: str | None = None
    additional_tool_calls_needed: int = 0
    additional_file_reads_needed: int = 0
