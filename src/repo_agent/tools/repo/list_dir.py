from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from repo_agent.tools.base import BaseTool, ToolResult


class ListDirArgs(BaseModel):
    path: str = "."
    recursive: bool = False
    max_entries: int = Field(default=100, ge=1)


class ListDirTool(BaseTool):
    name = "list_dir"
    description = (
        "List files and directories under a repository directory with basic metadata."
    )
    args_model = ListDirArgs

    _MAX_LINE_COUNT_BYTES = 1_000_000

    def __init__(
        self,
        repo_root: str | Path,
        *,
        ignored_names: set[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.ignored_names = ignored_names or set()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        args = ListDirArgs.model_validate(arguments)

        try:
            target = self._resolve_repo_path(args.path)
        except ValueError as exc:
            return ToolResult(success=False, content=str(exc))

        if not target.exists():
            return ToolResult(success=False, content=f"path does not exist: {args.path}")
        if not target.is_dir():
            return ToolResult(success=False, content=f"path is not a directory: {args.path}")

        all_entries = self._iter_entries(target, recursive=args.recursive)
        visible_entries = all_entries[: args.max_entries]
        truncated = len(all_entries) > len(visible_entries)
        entries = [self._entry_metadata(entry) for entry in visible_entries]

        content = self._format_content(
            path=self._display_path(target),
            recursive=args.recursive,
            entries=entries,
            truncated=truncated,
            hidden_count=len(all_entries) - len(visible_entries),
        )
        return ToolResult(
            success=True,
            content=content,
            metadata={
                "path": self._display_path(target),
                "recursive": args.recursive,
                "entry_count": len(entries),
                "total_entry_count": len(all_entries),
                "truncated": truncated,
                "entries": entries,
            },
        )

    def _iter_entries(self, directory: Path, *, recursive: bool) -> list[Path]:
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        entries = [
            entry
            for entry in iterator
            if not self._is_ignored(entry)
        ]
        return sorted(
            entries,
            key=lambda path: (
                path.relative_to(self.repo_root).as_posix().count("/"),
                not path.is_dir(),
                path.relative_to(self.repo_root).as_posix(),
            ),
        )

    def _entry_metadata(self, path: Path) -> dict[str, Any]:
        if path.is_dir():
            return {
                "path": self._display_path(path),
                "type": "dir",
                "size_bytes": None,
                "line_count": None,
            }

        try:
            stat = path.stat()
        except OSError:
            return {
                "path": self._display_path(path),
                "type": "file",
                "size_bytes": None,
                "line_count": None,
            }
        return {
            "path": self._display_path(path),
            "type": "file",
            "size_bytes": stat.st_size,
            "line_count": self._line_count(path, size_bytes=stat.st_size),
        }

    def _line_count(self, path: Path, *, size_bytes: int) -> int | None:
        if size_bytes > self._MAX_LINE_COUNT_BYTES:
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if b"\0" in data[:4096]:
            return None
        return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)

    def _format_content(
        self,
        *,
        path: str,
        recursive: bool,
        entries: list[dict[str, Any]],
        truncated: bool,
        hidden_count: int,
    ) -> str:
        lines = [
            f'<directory_listing path="{path}" recursive="{str(recursive).lower()}" entries="{len(entries)}" truncated="{str(truncated).lower()}">'
        ]
        for entry in entries:
            entry_path = str(entry["path"])
            if entry["type"] == "dir":
                lines.append(f"dir  {entry_path}/")
                continue

            details = []
            if entry["size_bytes"] is not None:
                details.append(f"{entry['size_bytes']} bytes")
            if entry["line_count"] is not None:
                details.append(f"{entry['line_count']} lines")
            detail = f"  {', '.join(details)}" if details else ""
            lines.append(f"file {entry_path}{detail}")

        if truncated:
            lines.append(f"... ({hidden_count} more entries)")
        lines.append("</directory_listing>")
        return "\n".join(lines)

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
