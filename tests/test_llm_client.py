import os
from pathlib import Path

from pydantic import BaseModel

from repo_agent.llm.client import DEFAULT_DASHSCOPE_BASE_URL, LLMClient
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


class _FakeCompletion:
    def __init__(self, payload: dict | None = None) -> None:
        self.choices = [_FakeChoice(payload)]
        self.payload = payload

    def model_dump(self) -> dict:
        if self.payload is not None:
            tool_calls = self.payload.get("tool_calls")
            return {
                "choices": [
                    {
                        "message": {
                            "content": self.choices[0].message.content,
                            "tool_calls": tool_calls,
                        }
                    }
                ]
            }
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


def test_llm_client_chat_maps_content_and_tool_calls() -> None:
    backend = _FakeBackend()
    client = LLMClient(
        model="qwen-plus",
        api_key="sk-test",
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        backend=backend,
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


def test_llm_client_extract_json_object_handles_wrapped_json() -> None:
    client = LLMClient(model="qwen-plus", api_key="sk-test", backend=_FakeBackend())

    payload = client.extract_json_object('prefix {"answer":"ok"} suffix')

    assert payload["answer"] == "ok"


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

    response, executed_tools = client.run_tool_calling_loop(
        system_prompt="test",
        user_content="say hi",
        tool_registry=registry,
        max_tool_calls=2,
    )

    assert response.content == "{\"answer\":\"done\"}"
    assert executed_tools[0]["name"] == "echo"
    assert executed_tools[0]["result"].content == "echo:hello"
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
    assert "tools" not in backend.calls[1]
    assert "File read budget exhausted" in backend.calls[1]["messages"][-1]["content"]
