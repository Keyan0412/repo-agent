from __future__ import annotations

import argparse
import re
import threading
import time
from pathlib import Path
from typing import Any

try:
    import readline as _readline  # noqa: F401
except ImportError:  # pragma: no cover - readline is platform-dependent.
    _readline = None

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from repo_agent.app import build_agent
from repo_agent.llm.analyze import format_run_summary, latest_run_summary
from repo_agent.runtime.config import AgentConfig
from repo_agent.runtime.text import strip_surrogates

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|O[@-~]|.)")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _compact(text: str, *, max_chars: int = 160) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _clean_query_input(text: str) -> str:
    text = strip_surrogates(text)
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


class CliEventSink:
    def __init__(self, *, show_tool_results: bool = False) -> None:
        self.console = Console()
        self.show_tool_results = show_tool_results
        self._status: Any | None = None
        self._timer_started_at: float | None = None
        self._timer_stop = threading.Event()
        self._timer_thread: threading.Thread | None = None

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
        self._timer_started_at = time.monotonic()
        self._timer_stop.clear()
        self._status = self.console.status(
            "0s",
            spinner="dots",
            spinner_style="green",
        )
        self._status.start()
        self._timer_thread = threading.Thread(target=self._refresh_timer, daemon=True)
        self._timer_thread.start()

    def stop_loading(self) -> None:
        self._timer_stop.set()
        if self._timer_thread is not None:
            self._timer_thread.join(timeout=0.2)
            self._timer_thread = None
        if self._status is None:
            return
        self._status.stop()
        self._status = None
        self._timer_started_at = None

    def _refresh_timer(self) -> None:
        while not self._timer_stop.wait(1):
            if self._status is None or self._timer_started_at is None:
                return
            elapsed = int(time.monotonic() - self._timer_started_at)
            self._status.update(f"{elapsed}s")

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
        success = bool(payload.get("success"))
        target = self._tool_target(name, arguments)
        status = "ok" if success else "failed"
        style = "dim magenta" if success else "red"

        line = Text()
        line.append("  tool ", style="dim")
        line.append(name, style="bold magenta" if success else "bold red")
        line.append(f" {status}", style=style)
        if target:
            line.append(" -> ", style="dim")
            line.append(_compact(target, max_chars=90), style="cyan")
        compact_detail = _compact(detail or summary or "-", max_chars=110)
        if compact_detail and compact_detail != "-":
            line.append("  ")
            line.append(compact_detail, style="dim")
        self.console.print(line)

        if self.show_tool_results and summary:
            self.console.print(Text(f"       {_compact(summary, max_chars=240)}", style="dim"))

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
        if name in {"list_dir", "read_file", "summarize_file"}:
            return str(arguments.get("path") or ".")
        if name == "find_files":
            pattern = str(arguments.get("pattern") or "*")
            path = str(arguments.get("path") or ".")
            return f"{path} :: {pattern}"
        if name == "read_files":
            files = arguments.get("files")
            if isinstance(files, list):
                paths = [
                    str(item.get("path") or "")
                    for item in files
                    if isinstance(item, dict)
                ]
                return ", ".join(path for path in paths if path)
            return "."
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
    parser.add_argument("--repo-root", default=".", help="Repository root to inspect.")
    parser.add_argument("--model", default=None, help="Override the complex model for this run.")
    parser.add_argument("--simple-model", default=None, help="Override the simple model for InvestigatorAgent.")
    parser.add_argument("--max-main-rounds", type=int, default=None)
    parser.add_argument(
        "--analyze-latest",
        action="store_true",
        help="Print a token and tool summary for the latest recorded run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --analyze-latest, print the raw run_summary.json payload.",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Show compact raw tool results in the Investigator tool panel.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if args.analyze_latest:
        summary = latest_run_summary(repo_root)
        console = Console()
        if args.json:
            console.print_json(data=summary)
        else:
            console.print(format_run_summary(summary))
        return

    config = AgentConfig(
        repo_path=str(repo_root),
        complex_model=args.model,
        simple_model=args.simple_model,
        max_main_rounds=args.max_main_rounds,
    )
    event_sink = CliEventSink(show_tool_results=args.show_tool_results)
    agent = build_agent(repo_path=repo_root, config=config, event_sink=event_sink)
    console = event_sink.console
    console.print("repo-agent interactive session. Type /exit or press Ctrl-D to quit.")

    while True:
        try:
            query = _clean_query_input(console.input("\n[bold cyan]repo-agent>[/] "))
        except EOFError:
            console.print()
            break

        if not query:
            continue
        if query in {"/exit", "/quit", "exit", "quit"}:
            break

        event_sink.start_loading()
        try:
            answer = agent.run(query)
        finally:
            event_sink.stop_loading()
        event_sink.print_final_answer(answer)


if __name__ == "__main__":
    main()
