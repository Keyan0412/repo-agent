from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class FileReaderAnswer(BaseModel):
    file_path: str
    question: str
    answer: str
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class InvestigatorReportPayload(BaseModel):
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
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    additional_tool_calls_needed: int = 0
    additional_file_reads_needed: int = 0


class FileSummaryPayload(BaseModel):
    model_config = ConfigDict(strict=True)

    class EvidenceRegion(BaseModel):
        model_config = ConfigDict(strict=True)

        start_line: int
        end_line: int
        label: str
        summary: str

    path: str
    role: str
    key_points: list[str] = Field(default_factory=list)
    evidence_regions: list[EvidenceRegion] = Field(default_factory=list)


class FilesSummaryPayload(BaseModel):
    model_config = ConfigDict(strict=True)

    class FileSummary(BaseModel):
        model_config = ConfigDict(strict=True)

        path: str
        role: str
        key_points: list[str] = Field(default_factory=list)
        evidence_regions: list[FileSummaryPayload.EvidenceRegion] = Field(default_factory=list)

    class CrossFileFinding(BaseModel):
        model_config = ConfigDict(strict=True)

        summary: str
        files: list[str] = Field(default_factory=list)

    focus: str
    files: list[FileSummary] = Field(default_factory=list)
    cross_file_findings: list[CrossFileFinding] = Field(default_factory=list)
