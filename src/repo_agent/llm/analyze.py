from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repo_agent.cache.paths import CachePaths


def latest_run_summary(
    repo_path: str | Path,
    *,
    cache_dir: str = ".cache/repo-agent",
) -> dict[str, Any]:
    paths = CachePaths(Path(repo_path), cache_dir)
    if not paths.runs_dir.exists():
        raise FileNotFoundError(f"runs directory does not exist: {paths.runs_dir}")
    candidates = [
        path / "run_summary.json"
        for path in paths.runs_dir.iterdir()
        if path.is_dir() and (path / "run_summary.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"no run_summary.json files found in: {paths.runs_dir}")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return json.loads(latest.read_text(encoding="utf-8"))


def format_run_summary(summary: dict[str, Any]) -> str:
    total = summary.get("total_usage") if isinstance(summary.get("total_usage"), dict) else {}
    lines = [
        f"Run: {summary.get('run_id', 'unknown')}",
        f"Status: {summary.get('status', 'unknown')}",
        f"User Query: {summary.get('user_query', '')}",
        "",
        "Token Usage:",
        f"  prompt: {total.get('prompt_tokens', 0)}",
        f"  completion: {total.get('completion_tokens', 0)}",
        f"  total: {total.get('total_tokens', 0)}",
        "",
        "Agents:",
    ]
    agents = summary.get("agents") if isinstance(summary.get("agents"), dict) else {}
    for agent, usage in agents.items():
        if not isinstance(usage, dict):
            continue
        lines.append(
            "  "
            f"{agent}: calls={usage.get('call_count', 0)} "
            f"total={usage.get('total_tokens', 0)}"
        )

    lines.append("")
    lines.append("Investigations:")
    investigations = summary.get("investigations")
    if not isinstance(investigations, list) or not investigations:
        lines.append("  none")
    else:
        for investigation in investigations:
            if not isinstance(investigation, dict):
                continue
            usage = investigation.get("usage") if isinstance(investigation.get("usage"), dict) else {}
            tool_counts = investigation.get("tool_counts") if isinstance(investigation.get("tool_counts"), dict) else {}
            tools = ", ".join(f"{name} x{count}" for name, count in tool_counts.items()) or "-"
            lines.extend(
                [
                    f"  {investigation.get('id', 'unknown')}: {investigation.get('task', '')}",
                    f"    tokens: {usage.get('total_tokens', 0)}",
                    f"    tools: {tools}",
                    f"    read_files calls: {investigation.get('read_files_call_count', 0)}",
                ]
            )
            paths = investigation.get("read_files_paths")
            if isinstance(paths, list) and paths:
                lines.append(f"    read paths: {', '.join(str(path) for path in paths)}")
            issues = investigation.get("issues")
            if isinstance(issues, list) and issues:
                lines.append(f"    issues: {'; '.join(str(issue) for issue in issues)}")

    issues = summary.get("issues")
    if isinstance(issues, list) and issues:
        lines.append("")
        lines.append("Issues:")
        for issue in issues:
            lines.append(f"  - {issue}")
    return "\n".join(lines)
