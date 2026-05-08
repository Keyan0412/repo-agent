from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class FileReaderAnswer(BaseModel):
    file_path: str
    question: str
    answer: str
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class InvestigatorSubreportPayload(BaseModel):
    model_config = ConfigDict(strict=True)

    class EvidenceSpan(BaseModel):
        model_config = ConfigDict(strict=True)

        file_path: str
        start_line: int
        end_line: int
        summary: str

    answer: str
    confidence: Literal["high", "medium", "low"]
    unresolved: list[str] = Field(default_factory=list)
    profile_update_suggestion: str | None = None
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    additional_tool_calls_needed: int = 0
    additional_file_reads_needed: int = 0


class AnalyzerPlanPayload(BaseModel):
    model_config = ConfigDict(strict=True)

    class Subquestion(BaseModel):
        model_config = ConfigDict(strict=True)

        question: str
        purpose: str
        expected_evidence: list[str] = Field(default_factory=list)
        known_information: str | None = None
        max_tool_calls: int = 4
        max_files: int = 3
        max_ask_file_calls: int = 12

    goal: str
    subquestions: list[Subquestion]
    synthesis_strategy: str


class AnalyzerReportPayload(BaseModel):
    model_config = ConfigDict(strict=True)

    summary: str
    remaining_questions: list[str] = Field(default_factory=list)
    profile_update_summary: str | None = None
