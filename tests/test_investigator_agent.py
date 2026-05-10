from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from repo_agent.agents.investigator_agent import InvestigatorAgent
from repo_agent.investigation import InvestigationTask
from repo_agent.llm.client import LLMClient
from repo_agent.tools.base import ToolResult
from repo_agent.tools.file import ReadFilesTool
from repo_agent.tools.registry import ToolRegistry
from repo_agent.tools.repo import FindFilesTool, FindTextTool, ListDirTool, TraceSymbolTool


class _FakeBackend:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No more fake LLM responses configured")
        return _FakeCompletion(self._responses.pop(0))


class _FakeCompletion:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.choices = [_FakeChoice(payload)]
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        message = {
            "content": self.choices[0].message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in (self.choices[0].message.tool_calls or [])
            ],
        }
        return {"choices": [{"message": message}]}


class _FakeChoice:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.message = _FakeMessage(payload)


class _FakeMessage:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = payload.get("content", "")
        tool_calls = payload.get("tool_calls")
        self.tool_calls = [_FakeToolCall(item) for item in tool_calls] if tool_calls else None


class _FakeToolCall:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.id = payload["id"]
        self.type = payload.get("type", "function")
        self.function = _FakeFunction(payload["function"])


class _FakeFunction:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.name = payload["name"]
        self.arguments = payload["arguments"]


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))


def _build_tool_registry(repo: Path) -> ToolRegistry:
    return ToolRegistry(
        [
            ListDirTool(repo),
            FindFilesTool(repo),
            FindTextTool(repo),
            TraceSymbolTool(repo),
            ReadFilesTool(repo),
        ]
    )


def test_investigate_uses_multi_round_tool_calling(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "worker.py").write_text(
        "def execute_task():\n"
        "    return 'done'\n"
        "\n"
        "result = execute_task()\n",
        encoding="utf-8",
    )

    backend = _FakeBackend(
        [
            {
                "tool_calls": [
                    {
                        "id": "call_trace",
                        "function": {
                            "name": "trace_symbol",
                            "arguments": '{"symbol_name": "execute_task", "max_results": 8}',
                        },
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "call_read",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"files": [{"path": "worker.py"}]}',
                        },
                    }
                ]
            },
            {
                "content": (
                    '{"answer":"`execute_task` is defined in worker.py and called later in the same file.",'
                    '"confidence":"high","unresolved":[],'
                    '"evidence_spans":[{"file_path":"worker.py","start_line":1,"end_line":4,"summary":"The file defines execute_task and calls it later."}],'
                    '"additional_tool_calls_needed":0,"additional_file_reads_needed":0}'
                )
            },
        ]
    )
    llm_client = LLMClient(model="qwen-plus", api_key="test-key", backend=backend)
    events = _RecordingEventSink()
    agent = InvestigatorAgent(
        llm_client=llm_client,
        repo_path=repo,
        tool_registry=_build_tool_registry(repo),
        event_sink=events,
    )
    task = InvestigationTask(
        id="T1",
        user_query="Where is `execute_task` defined and used?",
        task="Where is `execute_task` defined and used?",
        max_tool_calls=4,
        max_file_reads=2,
    )

    report = agent.investigate(task)

    assert report.summary
    assert "worker.py" in report.files_checked
    assert report.observations
    assert any(obs.file_path == "worker.py" and obs.start_line == 1 for obs in report.observations)
    assert report.observations[0].excerpt is not None
    assert "1 | def execute_task():" in report.observations[0].excerpt
    assert backend.calls[1]["messages"][-1]["role"] == "tool"
    assert backend.calls[2]["messages"][-1]["role"] == "tool"
    system_prompt = backend.calls[0]["messages"][0]["content"]
    first_user_message = backend.calls[0]["messages"][1]["content"]
    assert "Subtask ID:" not in first_user_message
    assert "Parent Task ID:" not in first_user_message
    assert "Search Hints:" not in first_user_message
    assert "已知信息:\n无" in first_user_message
    assert "最终输出契约:" in first_user_message
    assert "`confidence` 必须严格是小写 `high`、`medium` 或 `low`。" in first_user_message
    assert "evidence span 只能引用已用 `read_files` 检查过的文件。" in first_user_message
    assert "阶段 1：候选文件辨析" in system_prompt
    assert "阶段 2：集中精读并回答" in system_prompt
    assert "如果只需要读一个文件，也必须使用 `read_files`" in system_prompt
    assert "`read_files.files[].path` 只能使用你已经看见的真实文件路径。" in system_prompt
    assert "如果你只知道目录存在，但不知道目录下有哪些文件，必须先调用 `list_dir` 列出该目录" in system_prompt
    assert [event for event, _ in events.events] == [
        "investigator.tool_call",
        "investigator.tool_call",
        "investigator.report",
    ]
    assert events.events[0][1]["name"] == "trace_symbol"
    assert events.events[0][1]["arguments"]["symbol_name"] == "execute_task"
    assert events.events[0][1]["summary"] == "2 occurrences"
    assert events.events[0][1]["metadata"]["match_count"] == 2
    assert "occurrences" not in events.events[0][1]["metadata"]
    assert events.events[1][1]["name"] == "read_files"
    assert events.events[1][1]["summary"] == "read 1 files"
    assert events.events[1][1]["metadata"]["paths"] == ["worker.py"]
    assert "numbered_content" not in events.events[1][1]["metadata"]
    assert "result" not in events.events[1][1]
    assert events.events[2][1]["id"] == "R-T1"
    assert events.events[2][1]["task_id"] == "T1"
    assert events.events[2][1]["confidence"] == "high"
    assert events.events[2][1]["files_checked"] == ["worker.py"]


def test_investigator_collects_summary_regions_from_batch_summary(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    llm_client = LLMClient(model="qwen-plus", api_key="test-key", backend=object())
    agent = InvestigatorAgent(llm_client=llm_client, repo_path=repo, tool_registry=ToolRegistry([]))

    (
        files_checked,
        _symbols_checked,
        _file_contents,
        summary_regions,
    ) = agent._collect_execution_artifacts(
        executed_tools=[
            {
                "name": "summarize_files",
                "arguments": {"paths": ["a.py", "b.py"]},
                "result": ToolResult(
                    success=True,
                    content="summary",
                    metadata={
                        "paths": ["a.py", "b.py"],
                        "summary": {
                            "files": [
                                {
                                    "path": "a.py",
                                    "role": "entry point",
                                    "key_points": ["creates the app"],
                                    "evidence_regions": [
                                        {
                                            "start_line": 1,
                                            "end_line": 10,
                                            "label": "app factory",
                                            "summary": "Builds the app.",
                                        }
                                    ],
                                }
                            ],
                            "cross_file_findings": [
                                {
                                    "summary": "a.py imports b.py",
                                    "files": ["a.py", "b.py"],
                                }
                            ],
                        },
                    },
                ),
            }
        ],
        max_files=5,
    )

    assert files_checked == ["a.py", "b.py"]
    assert summary_regions["a.py"][0]["label"] == "app factory"


def test_investigator_collects_summary_regions_from_single_file_summary(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    llm_client = LLMClient(model="qwen-plus", api_key="test-key", backend=object())
    agent = InvestigatorAgent(llm_client=llm_client, repo_path=repo, tool_registry=ToolRegistry([]))

    (
        files_checked,
        _symbols_checked,
        _file_contents,
        summary_regions,
    ) = agent._collect_execution_artifacts(
        executed_tools=[
            {
                "name": "summarize_file",
                "arguments": {"path": "single.py"},
                "result": ToolResult(
                    success=True,
                    content="summary",
                    metadata={
                        "path": "single.py",
                        "summary": {
                            "path": "single.py",
                            "role": "single file utility",
                            "key_points": ["defines one helper"],
                            "evidence_regions": [
                                {
                                    "start_line": 3,
                                    "end_line": 8,
                                    "label": "helper",
                                    "summary": "Defines the helper.",
                                }
                            ],
                        },
                    },
                ),
            }
        ],
        max_files=5,
    )

    assert files_checked == ["single.py"]
    assert summary_regions["single.py"][0]["label"] == "helper"


def test_investigate_forces_output_when_file_budget_is_exhausted(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("target = 1\n", encoding="utf-8")
    (repo / "b.py").write_text("target = 2\n", encoding="utf-8")
    (repo / "c.py").write_text("target = 3\n", encoding="utf-8")

    backend = _FakeBackend(
        [
            {
                "tool_calls": [
                    {
                        "id": "call_find",
                        "function": {
                            "name": "find_text",
                            "arguments": '{"query": "target"}',
                        },
                    },
                    {
                        "id": "call_read_a",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"files": [{"path": "a.py"}]}',
                        },
                    },
                    {
                        "id": "call_read_b",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"files": [{"path": "b.py"}]}',
                        },
                    },
                ]
            },
            {
                "content": (
                    '{"answer":"The current answer is incomplete because the file read budget was exhausted after inspecting a.py.",'
                    '"confidence":"medium","unresolved":["Need to inspect additional files to confirm the full spread"],'
                    ''
                    '"evidence_spans":[{"file_path":"a.py","start_line":1,"end_line":1,"summary":"One occurrence in a.py."}],'
                    '"additional_tool_calls_needed":2,"additional_file_reads_needed":2}'
                )
            },
        ]
    )
    llm_client = LLMClient(model="qwen-plus", api_key="test-key", backend=backend)
    agent = InvestigatorAgent(llm_client=llm_client, repo_path=repo, tool_registry=_build_tool_registry(repo))
    task = InvestigationTask(
        id="T1",
        user_query="Where does target appear?",
        task="Where does target appear?",
        max_tool_calls=4,
        max_file_reads=1,
    )

    report = agent.investigate(task)

    assert "budget was exhausted" in report.summary
    assert report.files_checked == ["a.py"]
    assert backend.calls[1]["tools"]
    assert backend.calls[1]["tool_choice"] == "none"
    assert "文件访问预算已耗尽" in backend.calls[1]["messages"][-1]["content"]


def test_investigate_raises_on_invalid_payload_field_types(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "worker.py").write_text("def execute_task():\n    return 'done'\n", encoding="utf-8")

    backend = _FakeBackend(
        [
            {
                "tool_calls": [
                    {
                        "id": "call_read",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"files": [{"path": "worker.py"}]}',
                        },
                    }
                ]
            },
            {
                "content": (
                    '{"answer":"ok","confidence":"high","unresolved":"should-be-a-list",'
                    '"evidence_spans":[],"additional_tool_calls_needed":0,"additional_file_reads_needed":0}'
                )
            },
        ]
    )
    llm_client = LLMClient(model="qwen-plus", api_key="test-key", backend=backend)
    agent = InvestigatorAgent(llm_client=llm_client, repo_path=repo, tool_registry=_build_tool_registry(repo))
    task = InvestigationTask(
        id="T1",
        user_query="Inspect execute_task",
        task="Inspect execute_task",
        max_tool_calls=2,
        max_file_reads=1,
    )

    with pytest.raises(RuntimeError, match="invalid field types"):
        agent.investigate(task)


def test_investigate_repairs_fenced_json_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "worker.py").write_text("def execute_task():\n    return 'done'\n", encoding="utf-8")

    fenced_json = (
        '```json\n'
        '{"answer":"`execute_task` is implemented in worker.py.",'
        '"confidence":"high","unresolved":[],'
        '"evidence_spans":[{"file_path":"worker.py","start_line":1,"end_line":2,"summary":"execute_task returns done."}],'
        '"additional_tool_calls_needed":0,"additional_file_reads_needed":0}'
        '\n```'
    )
    repaired_json = (
        '{"answer":"`execute_task` is implemented in worker.py.",'
        '"confidence":"high","unresolved":[],'
        '"evidence_spans":[{"file_path":"worker.py","start_line":1,"end_line":2,"summary":"execute_task returns done."}],'
        '"additional_tool_calls_needed":0,"additional_file_reads_needed":0}'
    )
    backend = _FakeBackend(
        [
            {
                "tool_calls": [
                    {
                        "id": "call_read",
                        "function": {
                            "name": "read_files",
                            "arguments": '{"files": [{"path": "worker.py"}]}',
                        },
                    }
                ]
            },
            {"content": fenced_json},
            {"content": repaired_json, "tool_calls": []},
        ]
    )
    llm_client = LLMClient(model="qwen-plus", api_key="test-key", backend=backend)
    agent = InvestigatorAgent(llm_client=llm_client, repo_path=repo, tool_registry=_build_tool_registry(repo))
    task = InvestigationTask(
        id="T1",
        user_query="Inspect execute_task",
        task="Inspect execute_task",
        max_tool_calls=2,
        max_file_reads=1,
    )

    report = agent.investigate(task)

    assert report.summary == "`execute_task` is implemented in worker.py."
    assert report.observations[0].file_path == "worker.py"
    assert len(backend.calls) == 3
    repair_prompt = backend.calls[2]["messages"][0]["content"]
    assert "JsonRepairAgent" in repair_prompt
    assert "tool_choice" not in backend.calls[2]
    assert "tools" not in backend.calls[2]


def test_investigate_drops_unread_evidence_span_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "worker.py").write_text("def execute_task():\n    return 'done'\n", encoding="utf-8")

    backend = _FakeBackend(
        [
            {
                "tool_calls": [
                    {
                        "id": "call_trace",
                        "function": {
                            "name": "trace_symbol",
                            "arguments": '{"symbol_name": "execute_task", "max_results": 8}',
                        },
                    }
                ]
            },
            {
                "content": (
                    '{"answer":"ok","confidence":"high","unresolved":[],'
                    '"evidence_spans":[{"file_path":"other.py","start_line":1,"end_line":1,"summary":"Wrong file."}],'
                    '"additional_tool_calls_needed":0,"additional_file_reads_needed":0}'
                )
            },
        ]
    )
    llm_client = LLMClient(model="qwen-plus", api_key="test-key", backend=backend)
    agent = InvestigatorAgent(llm_client=llm_client, repo_path=repo, tool_registry=_build_tool_registry(repo))
    task = InvestigationTask(
        id="T1",
        user_query="Inspect execute_task",
        task="Inspect execute_task",
        max_tool_calls=2,
        max_file_reads=1,
    )

    report = agent.investigate(task)

    assert report.observations == []
    assert "Dropped evidence span for unchecked file: other.py" in report.remaining_questions
