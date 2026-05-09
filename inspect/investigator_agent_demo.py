from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repo_agent.agents.investigator_agent import InvestigatorAgent
from repo_agent.investigation import SubInvestigationTask
from repo_agent.llm.client import LLMClient
from repo_agent.llm.debug import JsonlLLMCallDebugRecorder
from repo_agent.tools.file import ReadFileTool
from repo_agent.tools.registry import ToolRegistry
from repo_agent.tools.repo import FindTextTool, ReadRepoTreeTool, TraceSymbolTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect InvestigatorAgent investigate_subtask() output.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="Repository root used by the tools. Defaults to the current repo root.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the simple model for InvestigatorAgent.",
    )

    parser.add_argument(
        "--question",
        default="`InvestigatorAgent` 如何在预算约束下组织 repo 调查？",
        help="Subtask question.",
    )
    parser.add_argument(
        "--purpose",
        default="理解 InvestigatorAgent 的调查执行方式",
        help="Why this subtask matters.",
    )
    parser.add_argument(
        "--expected-evidence",
        nargs="*",
        default=["关键函数调用", "预算控制逻辑", "文件和符号调查路径"],
        help="Expected evidence list.",
    )
    parser.add_argument(
        "--known-information",
        default=(
            "Known components include InvestigatorAgent, cache storage, and tool-based repository inspection. "
            "Search first around investigate_subtask and run_tool_calling_loop."
        ),
        help="Optional concise known information passed to investigate_subtask().",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=15,
        help="Max tool calls for the subtask.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Max files to read for the subtask.",
    )

    return parser


def build_investigator(repo_root: Path, model: str | None) -> InvestigatorAgent:
    llm_client = LLMClient.simple_from_env(
        model=model,
        env_path=ROOT / ".env",
        debug_recorder=JsonlLLMCallDebugRecorder.at_repo_cache(repo_root),
    )
    tool_registry = ToolRegistry(
        [
            ReadRepoTreeTool(repo_root),
            FindTextTool(repo_root),
            TraceSymbolTool(repo_root),
            ReadFileTool(repo_root),
        ]
    )
    return InvestigatorAgent(
        llm_client=llm_client,
        repo_path=repo_root,
        tool_registry=tool_registry,
    )

def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()

    try:
        investigator = build_investigator(repo_root, args.model)
    except ValueError as exc:
        print(str(exc))
        print("Copy `.env.example` to `.env` and fill in DASHSCOPE_API_KEY before running this demo.")
        return 1

    try:
        subtask = SubInvestigationTask(
            id="INSPECT-S1",
            parent_task_id="INSPECT-T1",
            question=args.question,
            purpose=args.purpose,
            expected_evidence=args.expected_evidence,
            known_information=args.known_information,
            max_tool_calls=args.max_tool_calls,
            max_files=args.max_files,
        )
        report = investigator.investigate_subtask(subtask=subtask)
        print("mode: investigator")
        print("report:")
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
