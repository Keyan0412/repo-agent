from __future__ import annotations

from pydantic import BaseModel, Field


class InvestigationTask(BaseModel):
    id: str
    user_query: str
    task: str
    relevant_evidence_ids: list[int] = Field(default_factory=list)
    max_tool_calls: int = 6
