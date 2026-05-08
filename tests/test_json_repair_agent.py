from __future__ import annotations

from typing import Any

from repo_agent.agents.json_repair_agent import JsonRepairAgent
from repo_agent.llm.schemas import LLMResponse


class _FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(content=self.response)


def test_json_repair_agent_requests_format_only_repair() -> None:
    client = _FakeLLMClient('{"answer":"ok"}')
    agent = JsonRepairAgent(client)  # type: ignore[arg-type]

    repaired = agent.repair_json(
        raw_content='```json\n{"answer":"ok"}\n```',
        target_name="DemoPayload",
        json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        error=RuntimeError("not json"),
    )

    assert repaired == '{"answer":"ok"}'
    assert client.calls[0]["tool_choice"] == "none"
    assert client.calls[0]["temperature"] == 0
    system_prompt = client.calls[0]["messages"][0]["content"]
    user_prompt = client.calls[0]["messages"][1]["content"]
    assert "format repair only" in system_prompt
    assert "must not add new facts" in system_prompt
    assert "Do not wrap the JSON in Markdown fences" in system_prompt
    assert "DemoPayload" in user_prompt
    assert "```json" in user_prompt
