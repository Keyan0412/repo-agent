from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from repo_agent.runtime.session import AgentSession
from repo_agent.tools.base import BaseTool, ToolResult


class FinalAnswerArgs(BaseModel):
    answer: str
    reports_used: list[int] = Field(default_factory=list)


class FinalAnswerTool(BaseTool):
    name = "final_answer"
    description = "Set the final user-facing answer."
    args_model = FinalAnswerArgs

    def __init__(self, session: AgentSession) -> None:
        self.session = session

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        args = FinalAnswerArgs.model_validate(arguments)
        if not args.answer.strip():
            return ToolResult(success=False, content="answer must not be empty")
        for report_id in args.reports_used:
            if report_id < 0 or report_id >= len(self.session.reports):
                raise ValueError(f"invalid report id: {report_id}")

        self.session.final_answer = args.answer.strip()
        return ToolResult(
            success=True,
            content="Final answer accepted.",
            metadata={"reports_used": args.reports_used},
        )
