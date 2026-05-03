from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from repo_agent.tools.base import BaseTool, ToolResult


class ReadFileArgs(BaseModel):
    path: str


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a single file with line numbers."
    args_model = ReadFileArgs

    def __init__(self, repo_root: str | Path, *, max_chars: int = 50_000) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.max_chars = max_chars

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        args = ReadFileArgs.model_validate(arguments)

        # avoid accessing outside repository
        try:
            target = self._resolve_repo_path(args.path)
        except ValueError as exc:
            return ToolResult(success=False, content=str(exc))

        # ensure the file exist
        if not target.exists():
            return ToolResult(success=False, content=f"file does not exist: {args.path}")
        if not target.is_file():
            return ToolResult(success=False, content=f"path is not a file: {args.path}")

        # construct numbered text
        text = target.read_text(encoding="utf-8", errors="replace")
        numbered = "\n".join(
            f"{line_no} | {line}"
            for line_no, line in enumerate(text.splitlines(), start=1)
        )

        # truncated if too long
        truncated = False
        if len(numbered) > self.max_chars:
            numbered = numbered[: self.max_chars].rstrip()
            truncated = True
            numbered += "\n... [truncated]"

        return ToolResult(
            success=True,
            content=numbered,
            metadata={
                "path": target.relative_to(self.repo_root).as_posix(),
                "truncated": truncated,
                "max_chars": self.max_chars,
            },
        )

    def _resolve_repo_path(self, raw_path: str) -> Path:
        """Avoid model from accessing files outside repository"""
        candidate = (self.repo_root / raw_path).resolve()
        if self.repo_root != candidate and self.repo_root not in candidate.parents:
            raise ValueError(f"path escapes repository root: {raw_path}")
        return candidate
