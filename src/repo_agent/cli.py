from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from repo_agent.app import run
from repo_agent.runtime.config import AgentConfig

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover - exercised only when optional CLI dependency is absent.
    Console = None  # type: ignore[assignment]
    Markdown = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]


def _compact(text: str, *, max_chars: int = 160) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _file_line_count(result: str) -> str | None:
    match = re.search(r'lines="(\d+)"', result)
    if match:
        return f"{match.group(1)} lines"
    return None


class CliEventSink:
    def __init__(self, *, show_tool_results: bool = False) -> None:
        self.console = Console() if Console is not None else None
        self.show_tool_results = show_tool_results
        self.tool_rows: list[dict[str, str]] = []
        self._status: Any | None = None

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if event == "main.investigation":
            self._print_main_investigation(payload)
        elif event == "investigator.tool_call":
            self._record_tool_call(payload)
        elif event == "investigator.report":
            self._flush_tool_calls()
            self._print_report(payload)

    def start_loading(self) -> None:
        if self.console is None:
            return
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
        if self.console is None:
            print(f"\n# Final Answer\n\n{answer.strip() or '无'}")
            return
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
        if self.console is None:
            print("\n[MainAgent] Request investigation")
            print(f"Task: {_compact(str(payload.get('task') or 'None'), max_chars=420)}")
            if missing:
                print("Missing:")
                for item in missing:
                    print(f"- {item}")
            return

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

    def _record_tool_call(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("name") or "unknown")
        arguments = payload.get("arguments") or {}
        result = str(payload.get("result") or "")
        self.tool_rows.append(
            {
                "tool": name,
                "status": "✓" if payload.get("success") else "✗",
                "target": self._tool_target(name, arguments),
                "detail": self._tool_detail(name, arguments, result),
            }
        )
        if self.show_tool_results and result:
            self.tool_rows[-1]["result"] = _compact(result, max_chars=900)

    def _flush_tool_calls(self) -> None:
        if not self.tool_rows:
            return

        if self.console is None:
            print("\n[Investigator]")
            for row in self.tool_rows:
                print(f"{row['tool']} {row['status']} {row['target']} {row['detail']}")
                if self.show_tool_results and row.get("result"):
                    print(f"result: {row['result']}")
            self.tool_rows.clear()
            return

        table = Table.grid(expand=True)
        table.add_column("Tool", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Target", overflow="fold")
        table.add_column("Detail", overflow="fold")
        for row in self.tool_rows:
            table.add_row(row["tool"], row["status"], row["target"], row["detail"])
            if self.show_tool_results and row.get("result"):
                table.add_row("", "", "result", row["result"])

        self.console.print()
        self.console.print(
            Panel(
                table,
                title="Investigator",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        self.tool_rows.clear()

    def _print_report(self, payload: dict[str, Any]) -> None:
        files = payload.get("files_checked") or []
        symbols = payload.get("symbols_checked") or []
        unresolved = payload.get("unresolved") or []
        confidence = str(payload.get("confidence") or "unknown")

        if self.console is None:
            print(f"\n[Investigation Report {confidence}]")
            print(payload.get("id") or "unknown")
            print(_compact(str(payload.get("answer") or "None"), max_chars=700))
            if files:
                print("Files:")
                for path in files:
                    print(f"- {path}")
            print(f"Observations: {payload.get('observations_count') or 0}")
            return

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
        if name in {"read_repo_tree", "read_file"}:
            return str(arguments.get("path") or ".")
        if name == "find_text":
            return str(arguments.get("query") or "")
        if name == "trace_symbol":
            return str(arguments.get("symbol_name") or "")
        return ", ".join(f"{key}={value!r}" for key, value in arguments.items()) or "-"

    @staticmethod
    def _tool_detail(name: str, arguments: dict[str, Any], result: str) -> str:
        if name == "read_repo_tree":
            depth = arguments.get("max_depth")
            entries = sum(1 for part in result.split() if part)
            parts = []
            if depth is not None:
                parts.append(f"depth={depth}")
            if entries:
                parts.append(f"{entries} entries")
            return "  ".join(parts) or "-"
        if name == "read_file":
            return _file_line_count(result) or "file read"
        if name == "find_text":
            matches = len([line for line in result.splitlines() if ":" in line])
            return f"{matches} matches" if matches else "search complete"
        if name == "trace_symbol":
            matches = len([line for line in result.splitlines() if ":" in line])
            return f"{matches} matches" if matches else "trace complete"
        return "-"


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
