from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from repo_agent.cache import ReportStore
from repo_agent.investigation import InvestigationReport, InvestigationTask
from repo_agent.runtime.session import AgentSession
from repo_agent.tools.base import BaseTool, ToolResult


class InvestigationProvider(Protocol):
    def investigate(self, task: InvestigationTask) -> InvestigationReport:
        ...


class RequestInvestigationArgs(BaseModel):
    task: str = Field(description="Focused investigation task for InvestigatorAgent.")
    missing_information: list[str] = Field(
        default_factory=list,
        description="Specific information gaps this investigation should close.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        description="Optional safety budget for InvestigatorAgent tool calls. Omit to use the configured default.",
    )
    max_file_reads: int | None = Field(
        default=None,
        description="Optional file-read budget for InvestigatorAgent. Omit to use the configured default.",
    )


class RequestInvestigationTool(BaseTool):
    name = "request_investigation"
    description = (
        "Ask InvestigatorAgent to inspect the repository and return evidence observations."
    )
    args_model = RequestInvestigationArgs

    def __init__(
        self,
        session: AgentSession,
        investigation_provider: InvestigationProvider,
        *,
        user_query: str,
        report_store: ReportStore | None = None,
        default_max_tool_calls: int = 30,
        default_max_file_reads: int = 15,
    ) -> None:
        self.session = session
        self.investigation_provider = investigation_provider
        self.user_query = user_query
        self.report_store = report_store
        self.default_max_tool_calls = default_max_tool_calls
        self.default_max_file_reads = default_max_file_reads

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        args = RequestInvestigationArgs.model_validate(arguments)
        max_tool_calls = (
            args.max_tool_calls
            if args.max_tool_calls is not None
            else self.default_max_tool_calls
        )
        max_file_reads = (
            args.max_file_reads
            if args.max_file_reads is not None
            else self.default_max_file_reads
        )

        task_text = args.task.strip()
        if args.missing_information:
            missing = "\n".join(f"- {item}" for item in args.missing_information)
            task_text = f"{task_text}\n\nMissing information to resolve:\n{missing}"

        task = InvestigationTask(
            id=self.session.next_task_id(),
            user_query=self.user_query,
            task=task_text,
            max_tool_calls=max_tool_calls,
            max_file_reads=max_file_reads,
        )
        report = self.investigation_provider.investigate(task)
        if report not in self.session.reports:
            self.session.reports.append(report)
        if self.report_store is not None:
            self.report_store.save(report)

        report_index = self.session.reports.index(report)
        return ToolResult(
            success=True,
            content=self._format_observations(report_index=report_index, report=report),
            metadata={
                "task_id": task.id,
                "report_id": report.id,
                "report_index": report_index,
            },
        )

    def get_openai_tool_schema(self) -> dict[str, Any]:
        schema = super().get_openai_tool_schema()
        properties = schema["function"]["parameters"].setdefault("properties", {})
        max_tool_calls = properties.setdefault("max_tool_calls", {})
        max_tool_calls["default"] = self.default_max_tool_calls
        max_file_reads = properties.setdefault("max_file_reads", {})
        max_file_reads["default"] = self.default_max_file_reads
        return schema

    @staticmethod
    def _format_observations(*, report_index: int, report: InvestigationReport) -> str:
        lines = [
            f"调查结果 [{report_index}] {report.id}:",
            f"总结: {report.summary}",
            "观察:",
        ]
        if report.observations:
            for observation in report.observations:
                location = ""
                if observation.file_path and observation.start_line is not None:
                    end_line = observation.end_line or observation.start_line
                    location = f" ({observation.file_path}:L{observation.start_line}-L{end_line})"
                lines.append(f"- O{observation.id}{location}: {observation.summary}")
        else:
            lines.append("- 无")
        return "\n".join(lines)
