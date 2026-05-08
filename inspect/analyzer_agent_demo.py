from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repo_agent.agents.analyzer_agent import AnalyzerAgent
from repo_agent.agents.investigator_agent import InvestigatorAgent
from repo_agent.cache import RepoProfileStore, ReportStore
from repo_agent.investigation import InvestigationTask
from repo_agent.llm.client import LLMClient
from repo_agent.llm.debug import JsonlLLMCallDebugRecorder
from repo_agent.runtime.session import AgentSession
from repo_agent.tools.file import AskFileTool, ReadFileTool
from repo_agent.tools.registry import ToolRegistry
from repo_agent.tools.repo import FindTextTool, ReadRepoTreeTool, TraceSymbolTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect AnalyzerAgent end-to-end investigation behavior.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="Repository root used by the tools. Defaults to the current repo root.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name. Defaults to REPO_AGENT_MODEL or qwen-plus.",
    )
    parser.add_argument(
        "--task-id",
        default="INSPECT-T1",
        help="InvestigationTask id.",
    )
    parser.add_argument(
        "--user-query",
        default="当前项目已经开发到哪一个阶段了，下一步应该做什么？",
        help="Original user query passed to AnalyzerAgent.",
    )
    parser.add_argument(
        "--task",
        default="分析 repo-agent 当前实现阶段、缺口和下一步开发优先级。",
        help="Concrete investigation task passed to AnalyzerAgent.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=8,
        help="High-level InvestigationTask max_tool_calls field.",
    )
    parser.add_argument(
        "--recent-report-limit",
        type=int,
        default=3,
        help="Number of cached reports AnalyzerAgent may load as prior context.",
    )
    parser.add_argument(
        "--max-subtasks",
        type=int,
        default=4,
        help="Maximum number of subquestions AnalyzerAgent may create.",
    )
    parser.add_argument(
        "--repo-profile-max-ask-file-calls",
        type=int,
        default=40,
        help="ask_file budget used only when AnalyzerAgent must generate an initial RepoProfile.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full InvestigationReport JSON after the human-readable summary.",
    )
    return parser


def build_analyzer(repo_root: Path, model: str | None, args: argparse.Namespace) -> AnalyzerAgent:
    llm_client = LLMClient.from_env(
        model=model,
        env_path=ROOT / ".env",
        debug_recorder=JsonlLLMCallDebugRecorder.at_repo_cache(repo_root),
    )
    tool_registry = ToolRegistry(
        [
            ReadRepoTreeTool(repo_root),
            FindTextTool(repo_root),
            TraceSymbolTool(repo_root),
            AskFileTool(repo_root, llm_client),
            ReadFileTool(repo_root),
        ]
    )
    investigator = InvestigatorAgent(
        llm_client=llm_client,
        repo_path=repo_root,
        tool_registry=tool_registry,
    )
    return AnalyzerAgent(
        llm_client=llm_client,
        session=AgentSession(),
        repo_path=repo_root,
        investigator=investigator,
        profile_store=RepoProfileStore(repo_root),
        report_store=ReportStore(repo_root),
        recent_report_limit=args.recent_report_limit,
        max_subtasks=args.max_subtasks,
        repo_profile_max_ask_file_calls=args.repo_profile_max_ask_file_calls,
    )


def print_summary(analyzer: AnalyzerAgent, report_path_count_before: int, report) -> None:
    profile_path = analyzer.profile_store.paths.profile_path
    reports = analyzer.report_store.list_reports()
    new_reports = reports[report_path_count_before:]

    print("mode: analyzer")
    print(f"repo_root: {analyzer.repo_path}")
    print(f"profile_path: {profile_path}")
    print(f"profile_exists: {profile_path.exists()}")
    print(f"new_report_paths: {', '.join(str(path) for path in new_reports) or 'None'}")
    print()
    print(f"report_id: {report.id}")
    print(f"task_id: {report.task_id}")
    print(f"summary: {report.summary}")
    print()
    print("files_checked:")
    for path in report.files_checked or ["None"]:
        print(f"- {path}")
    print()
    print("remaining_questions:")
    for question in report.remaining_questions or ["None"]:
        print(f"- {question}")
    print()
    print("subreports:")
    if not report.subreports:
        print("- None")
    for subreport in report.subreports:
        print(f"- {subreport.subtask_id} [{subreport.confidence}] {subreport.question}")
        print(f"  answer: {subreport.answer}")
        if subreport.files_checked:
            print(f"  files: {', '.join(subreport.files_checked)}")
        if subreport.unresolved:
            print(f"  unresolved: {', '.join(subreport.unresolved)}")
    print()
    print(f"profile_update_summary: {report.profile_update_summary or 'None'}")


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()

    try:
        analyzer = build_analyzer(repo_root, args.model, args)
    except ValueError as exc:
        print(str(exc))
        print("Copy `.env.example` to `.env` and fill in DASHSCOPE_API_KEY before running this demo.")
        return 1

    task = InvestigationTask(
        id=args.task_id,
        user_query=args.user_query,
        task=args.task,
        max_tool_calls=args.max_tool_calls,
    )

    report_path_count_before = len(analyzer.report_store.list_reports())
    try:
        report = analyzer.investigate(task)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    print_summary(analyzer, report_path_count_before, report)
    if args.json:
        print()
        print("report_json:")
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
