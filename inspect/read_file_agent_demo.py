from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repo_agent.agents.read_file_agent import ReadFileAgent, answer_to_json, read_numbered_file
from repo_agent.llm.client import LLMClient
from repo_agent.llm.debug import JsonlLLMCallDebugRecorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect ReadFileAgent line-map input and structured output.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="Repository root used to resolve the file path.",
    )
    parser.add_argument(
        "--path",
        default="src/repo_agent/runtime/session.py",
        help="Repository-relative file path to ask about.",
    )
    parser.add_argument(
        "--question",
        default="What is implemented in this file? Cite exact line ranges.",
        help="Question passed to ReadFileAgent.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name. Defaults to REPO_AGENT_MODEL or qwen-plus.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50_000,
        help="Approximate character budget for included line-map content.",
    )
    parser.add_argument(
        "--show-input",
        action="store_true",
        help="Print the full JSON line map sent to ReadFileAgent.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()

    try:
        llm_client = LLMClient.from_env(
            model=args.model,
            env_path=ROOT / ".env",
            debug_recorder=JsonlLLMCallDebugRecorder.at_repo_cache(repo_root),
        )
        path, line_count, truncated, numbered_content, line_map_content, valid_line_numbers = read_numbered_file(
            repo_root,
            args.path,
            max_chars=args.max_chars,
        )
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"error: {exc}")
        return 1

    print("mode: read_file_agent")
    print(f"repo_root: {repo_root}")
    print(f"path: {path}")
    print(f"line_count: {line_count}")
    print(f"included_line_numbers: {sorted(valid_line_numbers)}")
    print(f"truncated: {truncated}")
    print()

    if args.show_input:
        print("line_map_input:")
        print(line_map_content)
        print()

    agent = ReadFileAgent(llm_client, max_chars=args.max_chars)
    try:
        answer = agent.ask(
            path=path,
            question=args.question,
            line_map_content=line_map_content,
            line_count=line_count,
            truncated=truncated,
        )
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    print("answer:")
    print(answer_to_json(answer))
    print()
    print("numbered_content_for_excerpt:")
    print(numbered_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
