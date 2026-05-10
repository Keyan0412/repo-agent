from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from repo_agent.tools.base import BaseTool, ToolResult


class FindFilesArgs(BaseModel):
    pattern: str = "*"
    path: str = "."
    max_results: int = Field(default=100, ge=1)
    case_sensitive: bool = False


class FindFilesTool(BaseTool):
    name = "find_files"
    description = "Find repository files by filename or glob-style path pattern."
    args_model = FindFilesArgs

    def __init__(
        self,
        repo_root: str | Path,
        *,
        ignored_names: set[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.ignored_names = ignored_names or set()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        args = FindFilesArgs.model_validate(arguments)
        pattern = args.pattern.strip() or "*"

        try:
            target = self._resolve_repo_path(args.path)
        except ValueError as exc:
            return ToolResult(success=False, content=str(exc))

        if not target.exists():
            return ToolResult(success=False, content=f"path does not exist: {args.path}")
        if target.is_file():
            candidates = [target]
        else:
            candidates = self._iter_files(target)

        matches = [
            self._display_path(path)
            for path in candidates
            if self._matches(path, pattern=pattern, case_sensitive=args.case_sensitive)
        ]
        visible_matches = matches[: args.max_results]
        truncated = len(matches) > len(visible_matches)

        if not visible_matches:
            content = "No files found."
        else:
            lines = [
                f'<file_matches path="{self._display_path(target)}" pattern="{pattern}" matches="{len(visible_matches)}" truncated="{str(truncated).lower()}">'
            ]
            lines.extend(visible_matches)
            if truncated:
                lines.append(f"... ({len(matches) - len(visible_matches)} more files)")
            lines.append("</file_matches>")
            content = "\n".join(lines)

        return ToolResult(
            success=True,
            content=content,
            metadata={
                "path": self._display_path(target),
                "pattern": pattern,
                "match_count": len(visible_matches),
                "total_match_count": len(matches),
                "truncated": truncated,
                "paths": visible_matches,
            },
        )

    def _iter_files(self, target: Path) -> list[Path]:
        return [
            path
            for path in sorted(target.rglob("*"))
            if path.is_file() and not self._is_ignored(path)
        ]

    def _matches(self, path: Path, *, pattern: str, case_sensitive: bool) -> bool:
        rel_path = self._display_path(path)
        name = path.name
        if not case_sensitive:
            pattern = pattern.lower()
            rel_path = rel_path.lower()
            name = name.lower()
        if "/" in pattern:
            return PurePosixPath(rel_path).match(pattern)
        return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern)

    def _is_ignored(self, path: Path) -> bool:
        rel_parts = path.relative_to(self.repo_root).parts
        return any(part in self.ignored_names for part in rel_parts)

    def _resolve_repo_path(self, raw_path: str) -> Path:
        candidate = (self.repo_root / raw_path).resolve()
        if self.repo_root != candidate and self.repo_root not in candidate.parents:
            raise ValueError(f"path escapes repository root: {raw_path}")
        return candidate

    def _display_path(self, path: Path) -> str:
        if path == self.repo_root:
            return "."
        return path.relative_to(self.repo_root).as_posix()
