from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from repo_agent.tools.base import BaseTool, ToolResult


class ReadRepoTreeArgs(BaseModel):
    path: str = "."
    max_depth: int = 1
    read_all: bool = False


class ReadRepoTreeTool(BaseTool):
    name = "read_repo_tree"
    description = "Read a concise directory tree from the repository."
    args_model = ReadRepoTreeArgs

    def __init__(
        self,
        repo_root: str | Path,
        *,
        max_entries_per_dir: int = 20,
        ignored_names: set[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.max_entries_per_dir = max_entries_per_dir
        self.ignored_names = ignored_names or set()

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        args = ReadRepoTreeArgs.model_validate(arguments)
        if args.read_all and args.max_depth != 1:
            return ToolResult(
                success=False,
                content="read_all=True requires max_depth=1",
            )

        try:
            target = self._resolve_repo_path(args.path)
        except ValueError as exc:
            return ToolResult(success=False, content=str(exc))

        if not target.exists():
            return ToolResult(success=False, content=f"path does not exist: {args.path}")
        if not target.is_dir():
            return ToolResult(success=False, content=f"path is not a directory: {args.path}")

        lines = [f"{self._display_path(target)}/"]
        lines.extend(
            self._render_tree(
                directory=target,
                depth=0,
                max_depth=args.max_depth,
                read_all=args.read_all,
            )
        )
        return ToolResult(
            success=True,
            content="\n".join(lines),
            metadata={"path": self._display_path(target), "max_depth": args.max_depth},
        )

    def _render_tree(
        self,
        *,
        directory: Path,
        depth: int,
        max_depth: int,
        read_all: bool,
    ) -> list[str]:
        if depth >= max_depth:
            return []

        entries = [
            entry
            for entry in sorted(directory.iterdir(), key=lambda path: (not path.is_dir(), path.name))
            if entry.name not in self.ignored_names
        ]
        visible_entries = entries if read_all else entries[: self.max_entries_per_dir]
        lines: list[str] = []

        for entry in visible_entries:
            indent = "  " * (depth + 1)
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{indent}{entry.name}{suffix}")
            if entry.is_dir():
                lines.extend(
                    self._render_tree(
                        directory=entry,
                        depth=depth + 1,
                        max_depth=max_depth,
                        read_all=read_all,
                    )
                )

        if not read_all and len(entries) > len(visible_entries):
            indent = "  " * (depth + 1)
            remaining = len(entries) - len(visible_entries)
            lines.append(f"{indent}... ({remaining} more entries)")

        return lines

    def _resolve_repo_path(self, raw_path: str) -> Path:
        candidate = (self.repo_root / raw_path).resolve()
        if self.repo_root != candidate and self.repo_root not in candidate.parents:
            raise ValueError(f"path escapes repository root: {raw_path}")
        return candidate

    def _display_path(self, path: Path) -> str:
        """return relative path"""
        if path == self.repo_root:
            return "."
        return path.relative_to(self.repo_root).as_posix()
