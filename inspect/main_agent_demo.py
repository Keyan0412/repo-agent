from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repo_agent.app import build_agent
from repo_agent.runtime.config import AgentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect MainAgent end-to-end behavior through direct InvestigatorAgent calls.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="根据 development doc，这个项目的 MainAgent 主链路现在是如何工作的？",
        help="User query passed to MainAgent.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="Repository root used by the tools. Defaults to the current repo root.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the complex model for MainAgent.",
    )
    parser.add_argument(
        "--simple-model",
        default=None,
        help="Override the simple model for InvestigatorAgent.",
    )
    parser.add_argument(
        "--max-main-rounds",
        type=int,
        default=None,
        help="Maximum MainAgent tool-loop rounds. Unset means unlimited.",
    )
    return parser


def build_config(repo_root: Path, args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        repo_path=str(repo_root),
        complex_model=args.model,
        simple_model=args.simple_model,
        max_main_rounds=args.max_main_rounds,
    )


def print_session_summary(agent, repo_root: Path, answer: str) -> None:
    session = agent.session
    reports_dir = repo_root / ".cache/repo-agent/reports"
    llm_calls_path = repo_root / ".cache/repo-agent/llm_calls.jsonl"

    print("mode: main_agent")
    print(f"repo_root: {repo_root}")
    print(f"reports_dir: {reports_dir}")
    print(f"llm_calls_path: {llm_calls_path}")
    print()

    print("final_answer:")
    print(answer)
    print()

    print("session_reports:")
    if not session.reports:
        print("- None")
    for index, report in enumerate(session.reports):
        print(f"- [{index}] {report.id} task={report.task_id}")
        print(f"  summary: {report.summary}")
        if report.files_checked:
            print(f"  files_checked: {', '.join(report.files_checked)}")
        if report.remaining_questions:
            print(f"  remaining_questions: {', '.join(report.remaining_questions)}")


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    config = build_config(repo_root, args)

    try:
        agent = build_agent(repo_path=repo_root, config=config)
    except ValueError as exc:
        print(str(exc))
        print("Copy `.env.example` to `.env` and fill in DASHSCOPE_API_KEY before running this demo.")
        return 1

    try:
        answer = agent.run(args.query)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    print_session_summary(agent, repo_root, answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
