from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from repo_agent.agents.read_file_agent import ReadFileAgent, ReadFileAgentAnswer, answer_to_json, read_numbered_file
from repo_agent.llm.client import LLMClient
from repo_agent.tools.base import BaseTool, ToolResult


class AskFileArgs(BaseModel):
    path: str
    question: str
    focus: str | None = Field(
        default=None,
        description="Optional narrow focus such as implementation_status, file_role, or local_behavior.",
    )


class AskFileTool(BaseTool):
    name = "ask_file"
    description = (
        "Ask a focused question about one file. Prefer this over read_file for file purpose, "
        "implementation status, local behavior, and whether a file is a stub/config/doc/test."
    )
    args_model = AskFileArgs

    def __init__(
        self,
        repo_root: str | Path,
        llm_client: LLMClient,
        *,
        max_chars: int = 50_000,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.agent = ReadFileAgent(
            llm_client,
            max_chars=max_chars,
        )
        self.max_chars = max_chars

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        args = AskFileArgs.model_validate(arguments)
        question = args.question
        if args.focus:
            question = f"{question}\nFocus: {args.focus}"

        try:
            path, line_count, truncated, numbered_content, line_map_content, valid_line_numbers = read_numbered_file(
                self.repo_root,
                args.path,
                max_chars=self.max_chars,
            )
            answer = self.agent.ask(
                path=path,
                question=question,
                line_map_content=line_map_content,
                line_count=line_count,
                truncated=truncated,
            )
            self._validate_observed_fact_lines(
                answer=answer,
                valid_line_numbers=valid_line_numbers,
            )
        except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
            return ToolResult(success=False, content=str(exc))

        return ToolResult(
            success=True,
            content=answer_to_json(answer),
            metadata={
                "path": path,
                "line_count": line_count,
                "truncated": truncated,
                "max_chars": self.max_chars,
                "implementation_status": answer.implementation_status,
                "needs_cross_file_check": answer.needs_cross_file_check,
                "numbered_content": numbered_content,
                "line_map_content": line_map_content,
                "valid_line_numbers": sorted(valid_line_numbers),
            },
        )

    @staticmethod
    def _validate_observed_fact_lines(
        *,
        answer: ReadFileAgentAnswer,
        valid_line_numbers: set[int],
    ) -> None:
        for fact in answer.observed_facts:
            if fact.line_start <= 0 or fact.line_end < fact.line_start:
                raise ValueError(
                    f"ReadFileAgent returned invalid observed_facts range: {fact.model_dump()}"
                )
            missing_lines = [
                line_no
                for line_no in range(fact.line_start, fact.line_end + 1)
                if line_no not in valid_line_numbers
            ]
            if missing_lines:
                raise ValueError(
                    "ReadFileAgent returned observed_facts lines outside available file content: "
                    f"{missing_lines}"
                )
