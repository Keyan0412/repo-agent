from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from repo_agent.tools.base import BaseTool, ToolResult


class FindTextArgs(BaseModel):
    query: str
    path: str = "."
    max_results: int = Field(default=20, ge=1)
    case_sensitive: bool = False


class FindTextTool(BaseTool):
    name = "find_text"
    description = "Search repository files for matching text."
    args_model = FindTextArgs

    def __init__(
        self,
        repo_root: str | Path,
        *,
        ignored_names: set[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.ignored_names = ignored_names or set()

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        args = FindTextArgs.model_validate(arguments)
        if not args.query.strip():
            return ToolResult(success=False, content="query must not be empty")

        try:
            target = self._resolve_repo_path(args.path)
        except ValueError as exc:
            return ToolResult(success=False, content=str(exc))

        if not target.exists():
            return ToolResult(success=False, content=f"path does not exist: {args.path}")

        flags = 0 if args.case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(args.query, flags)
            used_literal_fallback = False
        except re.error:
            pattern = re.compile(re.escape(args.query), flags)
            used_literal_fallback = True
        matches: list[str] = []

        for file_path in self._iter_files(target):
            for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if pattern.search(line):
                    rel_path = file_path.relative_to(self.repo_root).as_posix()
                    matches.append(f"{rel_path}:{line_no}: {line}")
                    if len(matches) >= args.max_results:
                        return ToolResult(
                            success=True,
                            content="\n".join(matches),
                            metadata={
                                "truncated": True,
                                "match_count": len(matches),
                                "literal_fallback": used_literal_fallback,
                            },
                        )

        if not matches:
            return ToolResult(
                success=True,
                content="No matches found.",
                metadata={"match_count": 0, "literal_fallback": used_literal_fallback},
            )

        return ToolResult(
            success=True,
            content="\n".join(matches),
            metadata={
                "truncated": False,
                "match_count": len(matches),
                "literal_fallback": used_literal_fallback,
            },
        )

    def _iter_files(self, target: Path) -> list[Path]:
        if target.is_file():
            return [target]

        files: list[Path] = []
        for path in sorted(target.rglob("*")):
            if any(part in self.ignored_names for part in path.parts):
                continue
            if path.is_file():
                files.append(path)
        return files

    def _resolve_repo_path(self, raw_path: str) -> Path:
        candidate = (self.repo_root / raw_path).resolve()
        if self.repo_root != candidate and self.repo_root not in candidate.parents:
            raise ValueError(f"path escapes repository root: {raw_path}")
        return candidate
