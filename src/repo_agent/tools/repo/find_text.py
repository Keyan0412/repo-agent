from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from repo_agent.tools.base import BaseTool, ToolResult


class FindTextArgs(BaseModel):
    query: str = Field(description="Regex text query. Invalid regex is treated as literal text.")
    path: str = Field(default=".", description="Repository path to search within.")
    page: int = Field(default=1, ge=1, description="Result page to read. Each page contains at most 20 matches.")
    case_sensitive: bool = False


class FindTextTool(BaseTool):
    name = "find_text"
    description = "Search repository files for matching text."
    args_model = FindTextArgs
    page_size = 20

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
        start_index = (args.page - 1) * self.page_size
        stop_index = start_index + self.page_size
        matches: list[str] = []
        seen_matches = 0
        has_next_page = False

        for file_path in self._iter_files(target):
            for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if pattern.search(line):
                    if seen_matches < start_index:
                        seen_matches += 1
                        continue
                    if seen_matches >= stop_index:
                        has_next_page = True
                        return self._success_result(
                            matches=matches,
                            args=args,
                            literal_fallback=used_literal_fallback,
                            has_next_page=has_next_page,
                        )
                    rel_path = file_path.relative_to(self.repo_root).as_posix()
                    matches.append(f"{rel_path}:{line_no}: {line}")
                    seen_matches += 1

        if not matches:
            return ToolResult(
                success=True,
                content=self._no_matches_content(args.page),
                metadata={
                    "page": args.page,
                    "page_size": self.page_size,
                    "has_next_page": False,
                    "truncated": False,
                    "match_count": 0,
                    "literal_fallback": used_literal_fallback,
                },
            )

        return self._success_result(
            matches=matches,
            args=args,
            literal_fallback=used_literal_fallback,
            has_next_page=False,
        )

    def _success_result(
        self,
        *,
        matches: list[str],
        args: FindTextArgs,
        literal_fallback: bool,
        has_next_page: bool,
    ) -> ToolResult:
        footer = (
            f"还有更多结果，可以使用 page={args.page + 1} 继续读取。"
            if has_next_page
            else "以上为所有结果。"
        )
        return ToolResult(
            success=True,
            content="\n".join([*matches, footer]),
            metadata={
                "page": args.page,
                "page_size": self.page_size,
                "has_next_page": has_next_page,
                "next_page": args.page + 1 if has_next_page else None,
                "truncated": has_next_page,
                "match_count": len(matches),
                "literal_fallback": literal_fallback,
            },
        )

    @staticmethod
    def _no_matches_content(page: int) -> str:
        if page == 1:
            return "No matches found. 以上为所有结果。"
        return f"No matches found on page {page}. 以上为所有结果。"

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
