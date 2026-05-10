from __future__ import annotations

from pathlib import Path

from repo_agent.tools.file import (
    FileSummaryProvider,
    ReadFilesTool,
)
from repo_agent.tools.registry import ToolRegistry
from repo_agent.tools.repo import FindFilesTool, FindTextTool, ListDirTool, TraceSymbolTool

INVESTIGATOR_TOOLS = [
    "list_dir",
    "find_files",
    "find_text",
    "trace_symbol",
    "read_files",
]


def build_investigator_tool_registry(
    repo_path: str | Path,
    *,
    max_file_chars: int = 50_000,
    require_summary_over_chars: int | None = None,
    summary_provider: FileSummaryProvider | None = None,
    ignored_names: set[str] | None = None,
) -> ToolRegistry:
    del require_summary_over_chars
    del summary_provider
    ignored = ignored_names or {
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        "dist",
        "build",
        ".cache",
    }
    tools = [
        ListDirTool(repo_path, ignored_names=ignored),
        FindFilesTool(repo_path, ignored_names=ignored),
        FindTextTool(repo_path, ignored_names=ignored),
        TraceSymbolTool(repo_path, ignored_names=ignored),
        ReadFilesTool(
            repo_path,
            max_chars=max_file_chars,
            require_summary_over_chars=None,
        ),
    ]
    return ToolRegistry(tools)
