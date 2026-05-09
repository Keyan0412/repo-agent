from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from repo_agent.tools.base import BaseTool, ToolResult


class FileSummaryProvider(Protocol):
    def summarize_file(
        self,
        *,
        path: str,
        numbered_content: str,
        line_count: int,
        task: str | None = None,
    ) -> dict[str, Any]:
        ...

    def summarize_files(
        self,
        *,
        files: list[dict[str, Any]],
        task: str | None = None,
    ) -> dict[str, Any]:
        ...


class SummarizeFileArgs(BaseModel):
    path: str
    task: str | None = None


class SummarizeFilesArgs(BaseModel):
    paths: list[str]
    task: str | None = None


class SummarizeFileTool(BaseTool):
    name = "summarize_file"
    description = (
        "Summarize a repository file into structured memory when direct reading is too large "
        "or when only compressed file understanding is needed."
    )
    args_model = SummarizeFileArgs

    def __init__(
        self,
        repo_root: str | Path,
        summary_provider: FileSummaryProvider,
        *,
        max_input_chars: int = 120_000,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.summary_provider = summary_provider
        self.max_input_chars = max_input_chars

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        args = SummarizeFileArgs.model_validate(arguments)

        try:
            target = self._resolve_repo_path(args.path)
        except ValueError as exc:
            return ToolResult(success=False, content=str(exc))

        if not target.exists():
            return ToolResult(success=False, content=f"file does not exist: {args.path}")
        if not target.is_file():
            return ToolResult(success=False, content=f"path is not a file: {args.path}")

        text = target.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        numbered = "\n".join(
            f"{line_no} | {line}"
            for line_no, line in enumerate(all_lines, start=1)
        )
        truncated = False
        if len(numbered) > self.max_input_chars:
            numbered = numbered[: self.max_input_chars].rstrip()
            numbered += "\n... [truncated before summarization]"
            truncated = True

        path = target.relative_to(self.repo_root).as_posix()
        summary = self.summary_provider.summarize_file(
            path=path,
            numbered_content=numbered,
            line_count=len(all_lines),
            task=args.task,
        )
        return ToolResult(
            success=True,
            content=self._format_content(summary=summary, truncated=truncated),
            metadata={
                "path": path,
                "line_count": len(all_lines),
                "truncated_before_summary": truncated,
                "summary": summary,
            },
        )

    def _resolve_repo_path(self, raw_path: str) -> Path:
        candidate = (self.repo_root / raw_path).resolve()
        if self.repo_root != candidate and self.repo_root not in candidate.parents:
            raise ValueError(f"path escapes repository root: {raw_path}")
        return candidate

    @staticmethod
    def _format_content(*, summary: dict[str, Any], truncated: bool) -> str:
        lines = [
            f"<file_summary path=\"{summary.get('path', '')}\" trust=\"generated\">",
            f"role: {summary.get('role', '')}",
            "key_points:",
        ]
        for point in summary.get("key_points") or []:
            lines.append(f"- {point}")
        lines.append("evidence_regions:")
        for region in summary.get("evidence_regions") or []:
            lines.append(
                "- "
                f"L{region.get('start_line')}-L{region.get('end_line')} "
                f"{region.get('label')}: {region.get('summary')}"
            )
        if truncated:
            lines.append("note: file content was truncated before summarization")
        lines.append("</file_summary>")
        return "\n".join(lines)


class SummarizeFilesTool(BaseTool):
    name = "summarize_files"
    description = (
        "Jointly summarize multiple related repository files into structured memory. "
        "Prefer this for cross-file investigation after identifying candidate files."
    )
    args_model = SummarizeFilesArgs

    def __init__(
        self,
        repo_root: str | Path,
        summary_provider: FileSummaryProvider,
        *,
        max_total_input_chars: int = 180_000,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.summary_provider = summary_provider
        self.max_total_input_chars = max_total_input_chars

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        args = SummarizeFilesArgs.model_validate(arguments)
        if not args.paths:
            return ToolResult(success=False, content="paths must not be empty")

        files: list[dict[str, Any]] = []
        truncated_paths: list[str] = []
        seen_paths: set[str] = set()
        remaining_chars = self.max_total_input_chars
        for raw_path in args.paths:
            try:
                target = self._resolve_repo_path(raw_path)
            except ValueError as exc:
                return ToolResult(success=False, content=str(exc))

            if not target.exists():
                return ToolResult(success=False, content=f"file does not exist: {raw_path}")
            if not target.is_file():
                return ToolResult(success=False, content=f"path is not a file: {raw_path}")

            path = target.relative_to(self.repo_root).as_posix()
            if path in seen_paths:
                continue
            seen_paths.add(path)

            text = target.read_text(encoding="utf-8", errors="replace")
            all_lines = text.splitlines()
            numbered = "\n".join(
                f"{line_no} | {line}"
                for line_no, line in enumerate(all_lines, start=1)
            )
            truncated = False
            if len(numbered) > remaining_chars:
                numbered = numbered[: max(0, remaining_chars)].rstrip()
                numbered += "\n... [truncated before summarization]"
                truncated = True
            if truncated:
                truncated_paths.append(path)
            files.append(
                {
                    "path": path,
                    "numbered_content": numbered,
                    "line_count": len(all_lines),
                    "truncated_before_summary": truncated,
                }
            )
            remaining_chars -= min(len(numbered), max(remaining_chars, 0))

        summary = self.summary_provider.summarize_files(files=files, task=args.task)
        return ToolResult(
            success=True,
            content=self._format_content(summary=summary, truncated_paths=truncated_paths),
            metadata={
                "paths": [file["path"] for file in files],
                "file_count": len(files),
                "truncated_before_summary_paths": truncated_paths,
                "summary": summary,
            },
        )

    def _resolve_repo_path(self, raw_path: str) -> Path:
        candidate = (self.repo_root / raw_path).resolve()
        if self.repo_root != candidate and self.repo_root not in candidate.parents:
            raise ValueError(f"path escapes repository root: {raw_path}")
        return candidate

    @staticmethod
    def _format_content(*, summary: dict[str, Any], truncated_paths: list[str]) -> str:
        lines = [
            f"<files_summary focus=\"{summary.get('focus', '')}\" trust=\"generated\">",
            "files:",
        ]
        for file_summary in summary.get("files") or []:
            lines.append(f"- path: {file_summary.get('path', '')}")
            lines.append(f"  role: {file_summary.get('role', '')}")
            lines.append("  key_points:")
            for point in file_summary.get("key_points") or []:
                lines.append(f"  - {point}")
            lines.append("  evidence_regions:")
            for region in file_summary.get("evidence_regions") or []:
                lines.append(
                    "  - "
                    f"L{region.get('start_line')}-L{region.get('end_line')} "
                    f"{region.get('label')}: {region.get('summary')}"
                )
        lines.append("cross_file_findings:")
        for finding in summary.get("cross_file_findings") or []:
            files = ", ".join(finding.get("files") or [])
            lines.append(f"- {finding.get('summary', '')} [{files}]")
        if truncated_paths:
            lines.append(
                "note: these files were truncated before summarization: "
                + ", ".join(truncated_paths)
            )
        lines.append("</files_summary>")
        return "\n".join(lines)
