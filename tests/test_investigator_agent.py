from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from repo_agent.agents.investigator_agent import InvestigatorAgent
from repo_agent.investigation import SubInvestigationTask
from repo_agent.llm.client import LLMClient
from repo_agent.tools.file import ReadFileTool
from repo_agent.tools.registry import ToolRegistry
from repo_agent.tools.repo import FindTextTool, ReadRepoTreeTool, TraceSymbolTool


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
            ReadRepoTreeTool(repo),
            FindTextTool(repo),
            TraceSymbolTool(repo),
            ReadFileTool(repo),
        ]
    )


def test_investigate_subtask_uses_multi_round_tool_calling(tmp_path: Path) -> None:
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
                            "name": "read_file",
                            "arguments": '{"path": "worker.py"}',
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
    subtask = SubInvestigationTask(
        id="S1",
        parent_task_id="T1",
        question="Where is `execute_task` defined and used?",
        purpose="Locate the local execution flow",
        expected_evidence=["definition and usage lines for execute_task"],
        known_information="Known Components: worker module. Search first for execute_task.",
        max_tool_calls=4,
        max_files=2,
    )

    report = agent.investigate_subtask(subtask)

    assert report.answer
    assert report.confidence == "high"
    assert "worker.py" in report.files_checked
    assert "execute_task" in report.symbols_checked
    assert report.observations
    assert any(obs.file_path == "worker.py" and obs.start_line == 1 for obs in report.observations)
    assert report.observations[0].excerpt is not None
    assert "1 | def execute_task():" in report.observations[0].excerpt
    assert report.additional_tool_calls_needed == 0
    assert report.additional_file_reads_needed == 0
    assert backend.calls[1]["messages"][-1]["role"] == "tool"
    assert backend.calls[2]["messages"][-1]["role"] == "tool"
    first_user_message = backend.calls[0]["messages"][1]["content"]
    assert "Subtask ID:" not in first_user_message
    assert "Parent Task ID:" not in first_user_message
    assert "Search Hints:" not in first_user_message
    assert "已知信息:\nKnown Components: worker module. Search first for execute_task." in first_user_message
    assert "最终输出契约:" in first_user_message
    assert "`confidence` 必须严格是小写 `high`、`medium` 或 `low`。" in first_user_message
    assert "evidence span 只能引用已用 `read_file` 检查过的文件。" in first_user_message
    assert [event for event, _ in events.events] == [
        "investigator.tool_call",
        "investigator.tool_call",
        "investigator.report",
    ]
    assert events.events[0][1]["name"] == "trace_symbol"
    assert events.events[0][1]["arguments"]["symbol_name"] == "execute_task"
    assert events.events[1][1]["name"] == "read_file"
    assert "worker.py" in events.events[1][1]["result"]
    assert events.events[2][1]["id"] == "S1-report"
    assert events.events[2][1]["confidence"] == "high"
    assert events.events[2][1]["files_checked"] == ["worker.py"]


def test_investigate_subtask_forces_output_when_file_budget_is_exhausted(tmp_path: Path) -> None:
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
                            "arguments": '{"query": "target", "max_results": 8}',
                        },
                    },
                    {
                        "id": "call_read_a",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.py"}',
                        },
                    },
                    {
                        "id": "call_read_b",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "b.py"}',
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
    subtask = SubInvestigationTask(
        id="S2",
        parent_task_id="T1",
        question="Where does target appear?",
        purpose="Inspect spread",
        expected_evidence=["target references"],
        known_information="Search first for target and confirm how many files contain it.",
        max_tool_calls=4,
        max_files=1,
    )

    report = agent.investigate_subtask(subtask)

    assert "budget was exhausted" in report.answer
    assert report.additional_tool_calls_needed == 2
    assert report.additional_file_reads_needed == 2
    assert report.files_checked == ["a.py"]
    assert backend.calls[1]["tools"]
    assert backend.calls[1]["tool_choice"] == "none"
    assert "文件读取预算已耗尽" in backend.calls[1]["messages"][-1]["content"]


def test_investigate_subtask_raises_on_invalid_payload_field_types(tmp_path: Path) -> None:
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
                            "name": "read_file",
                            "arguments": '{"path": "worker.py"}',
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
    subtask = SubInvestigationTask(
        id="S3",
        parent_task_id="T1",
        question="Inspect execute_task",
        purpose="Check strict payload typing",
        expected_evidence=["definition lines"],
        known_information="Search first for execute_task.",
        max_tool_calls=2,
        max_files=1,
    )

    with pytest.raises(RuntimeError, match="invalid field types"):
        agent.investigate_subtask(subtask)


def test_investigate_subtask_repairs_fenced_json_output(tmp_path: Path) -> None:
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
                            "name": "read_file",
                            "arguments": '{"path": "worker.py"}',
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
    subtask = SubInvestigationTask(
        id="S5",
        parent_task_id="T1",
        question="Inspect execute_task",
        purpose="Check fenced JSON repair",
        expected_evidence=["definition lines"],
        known_information="Read worker.py.",
        max_tool_calls=2,
        max_files=1,
    )

    report = agent.investigate_subtask(subtask)

    assert report.answer == "`execute_task` is implemented in worker.py."
    assert report.confidence == "high"
    assert report.observations[0].file_path == "worker.py"
    assert len(backend.calls) == 3
    repair_prompt = backend.calls[2]["messages"][0]["content"]
    assert "JsonRepairAgent" in repair_prompt
    assert "tool_choice" not in backend.calls[2]
    assert "tools" not in backend.calls[2]


def test_investigate_subtask_drops_unread_evidence_span_file(tmp_path: Path) -> None:
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
    subtask = SubInvestigationTask(
        id="S4",
        parent_task_id="T1",
        question="Inspect execute_task",
        purpose="Check span validation",
        expected_evidence=["definition lines"],
        known_information="Search first for execute_task.",
        max_tool_calls=2,
        max_files=1,
    )

    report = agent.investigate_subtask(subtask)

    assert report.observations == []
    assert report.confidence == "medium"
    assert "Dropped evidence span for unread file: other.py" in report.unresolved
