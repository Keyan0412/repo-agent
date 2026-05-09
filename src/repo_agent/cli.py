from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from repo_agent.app import run
from repo_agent.runtime.config import AgentConfig


def _compact(text: str, *, max_chars: int = 160) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


class CliEventSink:
    def __init__(self, *, show_tool_results: bool = False) -> None:
        self.console = Console()
        self.show_tool_results = show_tool_results
        self._status: Any | None = None

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if event == "main.investigation":
            self._print_main_investigation(payload)
        elif event == "main.tool_error":
            self._print_main_tool_error(payload)
        elif event == "investigator.tool_call":
            self._print_tool_call(payload)
        elif event == "investigator.report":
            self._print_report(payload)

    def start_loading(self) -> None:
        self._status = self.console.status(
            "正在加载结果...",
            spinner="dots",
            spinner_style="green",
        )
        self._status.start()

    def stop_loading(self) -> None:
        if self._status is None:
            return
        self._status.stop()
        self._status = None

    def print_final_answer(self, answer: str) -> None:
        self.stop_loading()
        self.console.print()
        self.console.print(
            Panel(
                Markdown(answer.strip() or "无"),
                title="Final Answer",
                border_style="green",
                padding=(1, 2),
            )
        )

    def _print_main_investigation(self, payload: dict[str, Any]) -> None:
        missing = payload.get("missing_information") or []
        content = Text()
        content.append("Request investigation\n", style="bold")
        content.append("Task: ", style="bold cyan")
        content.append(_compact(str(payload.get("task") or "None"), max_chars=420))
        if missing:
            content.append("\n\nMissing:\n", style="bold cyan")
            for item in missing:
                content.append(f"  • {item}\n")
        self.console.print()
        self.console.print(
            Panel(
                content,
                title="MainAgent",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def _print_main_tool_error(self, payload: dict[str, Any]) -> None:
        content = Text()
        content.append("Tool failed\n", style="bold")
        content.append("Tool: ", style="bold red")
        content.append(str(payload.get("name") or "unknown"))
        content.append("\nError: ", style="bold red")
        content.append(_compact(str(payload.get("error") or "unknown"), max_chars=700))
        self.console.print()
        self.console.print(
            Panel(
                content,
                title="MainAgent",
                border_style="red",
                padding=(1, 2),
            )
        )

    def _print_tool_call(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("name") or "unknown")
        arguments = payload.get("arguments") or {}
        summary = str(payload.get("summary") or "")
        detail = str(payload.get("detail") or "")

        table = Table.grid(expand=True)
        table.add_column("Field", style="bold")
        table.add_column("Value", overflow="fold")
        table.add_row("Tool", name)
        table.add_row("Status", "ok" if payload.get("success") else "failed")
        table.add_row("Target", self._tool_target(name, arguments))
        table.add_row("Detail", detail or summary or "-")
        if self.show_tool_results and summary:
            table.add_row("Summary", _compact(summary, max_chars=300))

        self.console.print()
        self.console.print(
            Panel(
                table,
                title="Investigator Tool",
                border_style="magenta" if payload.get("success") else "red",
                padding=(1, 2),
            )
        )

    def _print_report(self, payload: dict[str, Any]) -> None:
        files = payload.get("files_checked") or []
        symbols = payload.get("symbols_checked") or []
        unresolved = payload.get("unresolved") or []
        confidence = str(payload.get("confidence") or "unknown")

        content = Text()
        content.append(f"{payload.get('id') or 'unknown'}\n", style="bold")
        content.append(f"\n{_compact(str(payload.get('answer') or 'None'), max_chars=700)}\n")
        if files:
            content.append("\nFiles:\n", style="bold cyan")
            for path in files:
                content.append(f"  • {path}\n")
        if symbols:
            content.append("\nSymbols:\n", style="bold cyan")
            for symbol in symbols:
                content.append(f"  • {symbol}\n")
        content.append(f"\nObservations: {payload.get('observations_count') or 0}")
        if unresolved:
            content.append("\n\nUnresolved:\n", style="bold yellow")
            for item in unresolved:
                content.append(f"  • {item}\n")

        self.console.print()
        self.console.print(
            Panel(
                content,
                title=f"Investigation Report  {confidence}",
                border_style="yellow" if confidence != "high" else "green",
                padding=(1, 2),
            )
        )

    @staticmethod
    def _tool_target(name: str, arguments: dict[str, Any]) -> str:
        if name in {"read_repo_tree", "read_file", "summarize_file"}:
            return str(arguments.get("path") or ".")
        if name == "summarize_files":
            paths = arguments.get("paths")
            if isinstance(paths, list):
                return ", ".join(str(path) for path in paths)
            return "."
        if name == "find_text":
            return str(arguments.get("query") or "")
        if name == "trace_symbol":
            return str(arguments.get("symbol_name") or "")
        return ", ".join(f"{key}={value!r}" for key, value in arguments.items()) or "-"

def main() -> None:
    parser = argparse.ArgumentParser(description="Run repo-agent against a repository question.")
    parser.add_argument("query", help="Question to answer about the repository.")
    parser.add_argument("--repo-root", default=".", help="Repository root to inspect.")
    parser.add_argument("--model", default=None, help="Override the complex model for this run.")
    parser.add_argument("--simple-model", default=None, help="Override the simple model for InvestigatorAgent.")
    parser.add_argument("--max-main-rounds", type=int, default=None)
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Show compact raw tool results in the Investigator tool panel.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = AgentConfig(
        repo_path=str(repo_root),
        complex_model=args.model,
        simple_model=args.simple_model,
        max_main_rounds=args.max_main_rounds,
    )
    event_sink = CliEventSink(show_tool_results=args.show_tool_results)
    event_sink.start_loading()
    try:
        answer = run(repo_root, args.query, config=config, event_sink=event_sink)
    finally:
        event_sink.stop_loading()
    event_sink.print_final_answer(answer)


if __name__ == "__main__":
    main()
