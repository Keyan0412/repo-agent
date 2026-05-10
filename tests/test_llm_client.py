import os
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from repo_agent.llm.client import DEFAULT_DASHSCOPE_BASE_URL, LLMClient
from repo_agent.llm.analyze import format_run_summary, latest_run_summary
from repo_agent.llm.debug import RunLLMCallDebugRecorder
from repo_agent.llm.schemas import LLMResponse
from repo_agent.tools.base import BaseTool, ToolResult
from repo_agent.tools.registry import ToolRegistry


class _FakeBackend:
    def __init__(self, responses: list[dict] | None = None) -> None:
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []
        self.responses = list(responses) if responses is not None else None

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        if self.responses is None:
            return _FakeCompletion()
        return _FakeCompletion(self.responses.pop(0))


class _ErrorBackend(_FakeBackend):
    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        raise RuntimeError("backend exploded")


class _FakeCompletion:
    def __init__(self, payload: dict | None = None) -> None:
        self.choices = [_FakeChoice(payload)]
        self.payload = payload
        self.usage = (payload or {}).get("usage")

    def model_dump(self) -> dict:
        if self.payload is not None:
            tool_calls = self.payload.get("tool_calls")
            dumped = {
                "choices": [
                    {
                        "message": {
                            "content": self.choices[0].message.content,
                            "tool_calls": tool_calls,
                        }
                    }
                ]
            }
            if "usage" in self.payload:
                dumped["usage"] = self.payload["usage"]
            return dumped
        return {
            "choices": [
                {
                    "message": {
                        "content": "hello",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{\"q\": \"x\"}"},
                            }
                        ],
                    }
                }
            ]
        }


class _FakeChoice:
    def __init__(self, payload: dict | None = None) -> None:
        self.message = _FakeMessage(payload)


class _FakeMessage:
    def __init__(self, payload: dict | None = None) -> None:
        payload = payload or {}
        self.content = payload.get("content", "hello")
        if "tool_calls" in payload:
            self.tool_calls = [_FakeToolCall(item) for item in payload["tool_calls"]]
        else:
            self.tool_calls = [_FakeToolCall()]


class _FakeToolCall:
    def __init__(self, payload: dict | None = None) -> None:
        payload = payload or {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"q\": \"x\"}"},
        }
        self.id = payload["id"]
        self.type = payload.get("type", "function")
        self.function = _FakeFunction(payload["function"])


class _FakeFunction:
    def __init__(self, payload: dict | None = None) -> None:
        payload = payload or {"name": "lookup", "arguments": "{\"q\": \"x\"}"}
        self.name = payload["name"]
        self.arguments = payload["arguments"]


class _EchoArgs(BaseModel):
    text: str


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echo text."
    args_model = _EchoArgs

    def execute(self, arguments: dict) -> ToolResult:
        args = _EchoArgs.model_validate(arguments)
        return ToolResult(success=True, content=f"echo:{args.text}", metadata={"text": args.text})


class _RecordingDebugRecorder:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def record_success(self, *, model: str, payload: dict, response) -> None:
        self.entries.append(
            {
                "status": "success",
                "model": model,
                "payload": payload,
                "response": response,
            }
        )

    def record_error(self, *, model: str, payload: dict, error: Exception) -> None:
        self.entries.append(
            {
                "status": "error",
                "model": model,
                "payload": payload,
                "error": error,
            }
        )


def test_llm_client_loads_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=sk-test\n"
        "DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1\n"
        "REPO_AGENT_MODEL=qwen-plus\n"
        "REPO_AGENT_ENABLE_THINKING=false\n",
        encoding="utf-8",
    )
    os.environ.pop("DASHSCOPE_API_KEY", None)
    os.environ.pop("DASHSCOPE_BASE_URL", None)
    os.environ.pop("REPO_AGENT_MODEL", None)
    os.environ.pop("REPO_AGENT_ENABLE_THINKING", None)

    client = LLMClient.from_env(env_path=env_file)

    assert client.api_key == "sk-test"
    assert client.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert client.model == "qwen-plus"
    assert client.enable_thinking is False


def test_llm_client_from_env_accepts_runtime_overrides(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CUSTOM_DASHSCOPE_KEY=sk-custom\n"
        "DASHSCOPE_BASE_URL=https://ignored.example/v1\n"
        "REPO_AGENT_ENABLE_THINKING=false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CUSTOM_DASHSCOPE_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.delenv("REPO_AGENT_ENABLE_THINKING", raising=False)

    client = LLMClient.from_env(
        env_path=env_file,
        api_key_env="CUSTOM_DASHSCOPE_KEY",
        base_url="https://configured.example/v1",
        enable_thinking=True,
    )

    assert client.api_key == "sk-custom"
    assert client.base_url == "https://configured.example/v1"
    assert client.enable_thinking is True


def test_llm_client_supports_complex_and_simple_env_models(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=sk-test\n"
        "REPO_AGENT_COMPLEX_MODEL=qwen-max\n"
        "REPO_AGENT_SIMPLE_MODEL=qwen-turbo\n"
        "REPO_AGENT_MODEL=qwen-plus\n",
        encoding="utf-8",
    )
    os.environ.pop("DASHSCOPE_API_KEY", None)
    os.environ.pop("REPO_AGENT_COMPLEX_MODEL", None)
    os.environ.pop("REPO_AGENT_SIMPLE_MODEL", None)
    os.environ.pop("REPO_AGENT_MODEL", None)

    complex_client = LLMClient.complex_from_env(env_path=env_file)
    simple_client = LLMClient.simple_from_env(env_path=env_file)

    assert complex_client.model == "qwen-max"
    assert simple_client.model == "qwen-turbo"


def test_llm_client_chat_maps_content_and_tool_calls() -> None:
    backend = _FakeBackend()
    recorder = _RecordingDebugRecorder()
    client = LLMClient(
        model="qwen-plus",
        api_key="sk-test",
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        backend=backend,
        debug_recorder=recorder,
    )

    response = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        tool_choice="auto",
        temperature=0,
    )

    assert response.content == "hello"
    assert response.tool_calls[0]["function"]["name"] == "lookup"
    assert backend.calls[0]["model"] == "qwen-plus"
    assert backend.calls[0]["extra_body"]["enable_thinking"] is False
    assert recorder.entries[0]["status"] == "success"
    assert recorder.entries[0]["payload"]["messages"][0]["content"] == "hello"
    assert recorder.entries[0]["response"].content == "hello"


def test_llm_client_strips_surrogates_before_backend_and_debug_recorder() -> None:
    backend = _FakeBackend(responses=[{"content": "done", "tool_calls": []}])
    recorder = _RecordingDebugRecorder()
    client = LLMClient(
        model="qwen-plus",
        api_key="sk-test",
        backend=backend,
        debug_recorder=recorder,
    )

    response = client.chat(
        messages=[{"role": "user", "content": "分析\udce8schemas"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "bad\udce8description",
                    "parameters": {},
                },
            }
        ],
    )

    assert response.content == "done"
    assert backend.calls[0]["messages"][0]["content"] == "分析schemas"
    assert backend.calls[0]["tools"][0]["function"]["description"] == "baddescription"
    assert recorder.entries[0]["payload"]["messages"][0]["content"] == "分析schemas"


def test_llm_client_chat_extracts_token_usage() -> None:
    backend = _FakeBackend(
        responses=[
            {
                "content": "done",
                "tool_calls": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                },
            }
        ]
    )
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=backend)

    response = client.chat(messages=[{"role": "user", "content": "hello"}])

    assert response.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }


def test_llm_client_chat_normalizes_dashscope_usage_names() -> None:
    backend = _FakeBackend(
        responses=[
            {
                "content": "done",
                "tool_calls": [],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 7,
                    "total_tokens": 27,
                },
            }
        ]
    )
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=backend)

    response = client.chat(messages=[{"role": "user", "content": "hello"}])

    assert response.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 7,
        "total_tokens": 27,
    }


def test_llm_client_omits_tool_choice_when_no_tools_are_supplied() -> None:
    backend = _FakeBackend(responses=[{"content": "{}", "tool_calls": []}])
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=backend)

    client.chat(
        messages=[{"role": "user", "content": "strict json"}],
        tool_choice="none",
        temperature=0,
    )

    assert "tools" not in backend.calls[0]
    assert "tool_choice" not in backend.calls[0]


def test_llm_client_chat_records_errors() -> None:
    backend = _ErrorBackend()
    recorder = _RecordingDebugRecorder()
    client = LLMClient(
        model="qwen-plus",
        api_key="sk-test",
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        backend=backend,
        debug_recorder=recorder,
    )

    try:
        client.chat(messages=[{"role": "user", "content": "hello"}])
    except RuntimeError as exc:
        assert str(exc) == "backend exploded"
    else:
        raise AssertionError("Expected RuntimeError")

    assert recorder.entries[0]["status"] == "error"
    assert recorder.entries[0]["payload"]["messages"][0]["content"] == "hello"
    assert str(recorder.entries[0]["error"]) == "backend exploded"


def test_run_llm_call_debug_recorder_writes_run_artifacts(tmp_path: Path) -> None:
    recorder = RunLLMCallDebugRecorder.at_repo_cache(tmp_path)
    response = LLMResponse(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "request_investigation",
                    "arguments": "{\"task\": \"Inspect tools\"}",
                },
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        raw={},
    )
    recorder.record_success(
        model="qwen-plus",
        payload={
            "messages": [
                {"role": "system", "content": "你是 MainAgent。你可以请求 InvestigatorAgent。"},
                {"role": "user", "content": "question"},
            ]
        },
        response=response,
    )
    recorder.record_success(
        model="qwen-plus",
        payload={
            "messages": [
                {"role": "system", "content": "你是 InvestigatorAgent。"},
                {"role": "user", "content": "inspect"},
            ]
        },
        response=LLMResponse(
            content="{}",
            tool_calls=[
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_files",
                        "arguments": "{\"files\": [{\"path\": \"a.py\"}]}",
                    },
                }
            ],
            usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            raw={},
        ),
    )

    recorder.finalize_run(
        user_query="How does it work?",
        final_answer="done",
        status="success",
    )

    run_dir = tmp_path / ".cache" / "repo-agent" / "runs" / recorder.run_id
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert (run_dir / "raw_llm_calls.jsonl").exists()
    assert (run_dir / "main_agent.json").exists()
    assert (run_dir / "investigations" / "T0001.json").exists()
    assert summary["run_id"] == recorder.run_id
    assert summary["total_usage"]["total_tokens"] == 35
    assert summary["agents"]["main"]["total_tokens"] == 12
    assert summary["investigations"][0]["read_files_paths"] == ["a.py"]

    latest = latest_run_summary(tmp_path)
    assert latest["run_id"] == recorder.run_id
    formatted = format_run_summary(latest)
    assert "How does it work?" in formatted
    assert "read_files calls: 1" in formatted


def test_run_llm_call_debug_recorder_strips_surrogates_from_artifacts(tmp_path: Path) -> None:
    recorder = RunLLMCallDebugRecorder.at_repo_cache(tmp_path)

    recorder.record_error(
        model="qwen-plus",
        payload={
            "messages": [
                {"role": "system", "content": "你是 MainAgent。"},
                {"role": "user", "content": "分析\udce8schemas"},
            ]
        },
        error=RuntimeError("backend\udce8exploded"),
    )
    recorder.finalize_run(
        user_query="分析\udce8schemas",
        final_answer=None,
        status="error",
        error="backend\udce8exploded",
    )

    run_dir = tmp_path / ".cache" / "repo-agent" / "runs" / recorder.run_id
    raw_text = (run_dir / "raw_llm_calls.jsonl").read_text(encoding="utf-8")
    summary_text = (run_dir / "run_summary.json").read_text(encoding="utf-8")

    assert "\udce8" not in raw_text
    assert "\udce8" not in summary_text
    assert "分析schemas" in summary_text
    assert "backendexploded" in raw_text


def test_llm_client_extract_json_object_raises_with_original_content() -> None:
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=_FakeBackend())

    with pytest.raises(RuntimeError, match="Failed to parse JSON object from LLM output: prefix"):
        client.extract_json_object('prefix {"answer":"ok"} suffix')


class _RepairAgent:
    def __init__(self, repaired: str) -> None:
        self.repaired = repaired
        self.calls: list[dict] = []

    def repair_json(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self.repaired


def test_llm_client_extract_json_object_uses_repair_agent_on_parse_failure() -> None:
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=_FakeBackend())
    repair_agent = _RepairAgent('{"answer":"ok"}')

    payload = client.extract_json_object(
        '```json\n{"answer":"ok"}\n```',
        repair_agent=repair_agent,
        target_name="DemoPayload",
        json_schema={"type": "object"},
    )

    assert payload == {"answer": "ok"}
    assert repair_agent.calls[0]["target_name"] == "DemoPayload"
    assert repair_agent.calls[0]["json_schema"] == {"type": "object"}
    assert "```json" in repair_agent.calls[0]["raw_content"]


def test_llm_client_run_tool_calling_loop_executes_and_reinjects_tools() -> None:
    backend = _FakeBackend(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"text\": \"hello\"}"},
                    }
                ]
            },
            {"content": "{\"answer\":\"done\"}", "tool_calls": []},
        ]
    )
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=backend)
    registry = ToolRegistry([_EchoTool()])
    observed_tools: list[dict] = []

    response, executed_tools = client.run_tool_calling_loop(
        system_prompt="test",
        user_content="say hi",
        tool_registry=registry,
        max_tool_calls=2,
        on_tool_result=observed_tools.append,
    )

    assert response.content == "{\"answer\":\"done\"}"
    assert executed_tools[0]["name"] == "echo"
    assert executed_tools[0]["result"].content == "echo:hello"
    assert observed_tools == executed_tools
    assert backend.calls[1]["messages"][-1]["role"] == "tool"


def test_llm_client_run_tool_calling_loop_forces_final_answer_after_budget_exhaustion() -> None:
    backend = _FakeBackend(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"b.py\"}"},
                    },
                ]
            },
            {"content": "{\"answer\":\"budget limited\"}", "tool_calls": []},
        ]
    )
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=backend)

    class _ReadFileArgs(BaseModel):
        path: str

    class _ReadFileTool(BaseTool):
        name = "read_file"
        description = "Read a file."
        args_model = _ReadFileArgs

        def execute(self, arguments: dict) -> ToolResult:
            args = _ReadFileArgs.model_validate(arguments)
            return ToolResult(success=True, content=f"1 | file:{args.path}", metadata={"path": args.path})

    registry = ToolRegistry([_ReadFileTool()])

    response, executed_tools = client.run_tool_calling_loop(
        system_prompt="test",
        user_content="inspect files",
        tool_registry=registry,
        max_tool_calls=4,
        max_files=1,
    )

    assert response.content == "{\"answer\":\"budget limited\"}"
    assert len(executed_tools) == 1
    assert executed_tools[0]["arguments"]["path"] == "a.py"
    assert backend.calls[1]["tools"]
    assert backend.calls[1]["tool_choice"] == "none"
    assert "文件访问预算已耗尽" in backend.calls[1]["messages"][-1]["content"]


def test_llm_client_counts_summarize_files_paths_against_file_budget() -> None:
    backend = _FakeBackend(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "summarize_files",
                            "arguments": "{\"paths\": [\"a.py\", \"b.py\"]}",
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"c.py\"}"},
                    },
                ]
            },
            {"content": "{\"answer\":\"budget limited\"}", "tool_calls": []},
        ]
    )
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=backend)

    class _SummarizeFilesArgs(BaseModel):
        paths: list[str]

    class _SummarizeFilesTool(BaseTool):
        name = "summarize_files"
        description = "Summarize files."
        args_model = _SummarizeFilesArgs

        def execute(self, arguments: dict) -> ToolResult:
            args = _SummarizeFilesArgs.model_validate(arguments)
            return ToolResult(
                success=True,
                content="summarized",
                metadata={"paths": args.paths, "file_count": len(args.paths)},
            )

    class _ReadFileArgs(BaseModel):
        path: str

    class _ReadFileTool(BaseTool):
        name = "read_file"
        description = "Read a file."
        args_model = _ReadFileArgs

        def execute(self, arguments: dict) -> ToolResult:
            args = _ReadFileArgs.model_validate(arguments)
            return ToolResult(success=True, content=f"1 | file:{args.path}", metadata={"path": args.path})

    registry = ToolRegistry([_SummarizeFilesTool(), _ReadFileTool()])

    response, executed_tools = client.run_tool_calling_loop(
        system_prompt="test",
        user_content="inspect files",
        tool_registry=registry,
        max_tool_calls=4,
        max_files=2,
    )

    assert response.content == "{\"answer\":\"budget limited\"}"
    assert len(executed_tools) == 1
    assert executed_tools[0]["name"] == "summarize_files"
    assert backend.calls[1]["tool_choice"] == "none"
    assert "文件访问预算已耗尽" in backend.calls[1]["messages"][-1]["content"]


def test_llm_client_recovers_when_model_requests_tools_after_budget_exhaustion() -> None:
    backend = _FakeBackend(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"text\": \"hello\"}"},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"text\": \"again\"}"},
                    }
                ]
            },
            {"content": "{\"answer\":\"budget limited\"}", "tool_calls": []},
        ]
    )
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=backend)
    registry = ToolRegistry([_EchoTool()])

    response, executed_tools = client.run_tool_calling_loop(
        system_prompt="test",
        user_content="say hi",
        tool_registry=registry,
        max_tool_calls=1,
    )

    assert response.content == "{\"answer\":\"budget limited\"}"
    assert len(executed_tools) == 1
    assert len(backend.calls) == 3
    assert backend.calls[1]["tool_choice"] == "none"
    assert backend.calls[2]["tool_choice"] == "none"
    assert "仍然请求了工具" in backend.calls[2]["messages"][-1]["content"]
