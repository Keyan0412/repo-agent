from __future__ import annotations

from pathlib import Path

from repo_agent.tools.file import ReadFileTool
from repo_agent.tools.registry import ToolRegistry
from repo_agent.tools.repo import FindTextTool, ReadRepoTreeTool, TraceSymbolTool

INVESTIGATOR_TOOLS = [
    "read_repo_tree",
    "find_text",
    "trace_symbol",
    "read_file",
]


def build_investigator_tool_registry(
    repo_path: str | Path,
    *,
    max_file_chars: int = 50_000,
    ignored_names: set[str] | None = None,
) -> ToolRegistry:
    ignored = ignored_names or {
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        "dist",
        "build",
        ".cache",
    }
    return ToolRegistry(
        [
            ReadRepoTreeTool(repo_path, ignored_names=ignored),
            FindTextTool(repo_path, ignored_names=ignored),
            TraceSymbolTool(repo_path, ignored_names=ignored),
            ReadFileTool(repo_path, max_chars=max_file_chars),
        ]
    )
