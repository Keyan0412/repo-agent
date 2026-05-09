from __future__ import annotations

from typing import Any

from repo_agent.agents.main_agent import MainAgent
from repo_agent.investigation import (
    InvestigationReport,
    InvestigationTask,
    Observation,
)
from repo_agent.llm.schemas import LLMResponse
from repo_agent.runtime.session import AgentSession


class _FakeLLMClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No more fake LLM responses configured")
        return self.responses.pop(0)


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))


class _FakeInvestigationProvider:
    def __init__(self) -> None:
        self.tasks: list[InvestigationTask] = []

    def investigate(self, task: InvestigationTask) -> InvestigationReport:
        self.tasks.append(task)
        return InvestigationReport(
            id=f"R-{task.id}",
            task_id=task.id,
            summary="MainAgent is implemented as a tool-driven evidence loop.",
            observations=[
                Observation(
                    id=1,
                    summary="The main loop calls only main-agent control tools.",
                    file_path="src/repo_agent/agents/main_agent.py",
                    start_line=1,
                    end_line=10,
                )
            ],
            files_checked=["src/repo_agent/agents/main_agent.py"],
        )


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_main_agent_investigates_and_finalizes() -> None:
    llm_client = _FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    _tool_call(
                        "call_investigate",
                        "request_investigation",
                        (
                            '{"task":"Check MainAgent implementation",'
                            '"missing_information":["whether it uses control tools"],'
                            '"max_tool_calls":6}'
                        ),
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    _tool_call(
                        "call_final",
                        "final_answer",
                        '{"answer":"MainAgent is now wired as a direct investigation loop.","reports_used":[0]}',
                    )
                ]
            ),
        ]
    )
    session = AgentSession()
    investigator = _FakeInvestigationProvider()
    events = _RecordingEventSink()
    agent = MainAgent(
        llm_client=llm_client,  # type: ignore[arg-type]
        session=session,
        investigator=investigator,
        event_sink=events,
    )

    answer = agent.run("Is MainAgent implemented?")

    assert answer == "MainAgent is now wired as a direct investigation loop."
    assert len(investigator.tasks) == 1
    assert investigator.tasks[0].id == "T0001"
    assert len(session.reports) == 1
    exposed_tools = {
        tool["function"]["name"]
        for tool in llm_client.calls[0]["tools"]
    }
    assert exposed_tools == {"request_investigation", "final_answer"}
    assert [event for event, _ in events.events] == [
        "main.investigation",
        "main.final_answer",
    ]
    assert events.events[0][1]["task"] == "Check MainAgent implementation"
    all_message_text = "\n".join(
        str(message.get("content") or "")
        for call in llm_client.calls
        for message in call["messages"]
    )
    assert "调查结果 [0] R-T0001" in all_message_text
    assert "reports_used" in all_message_text
    assert "O1" in all_message_text


def test_main_agent_uses_configured_default_investigator_tool_budget() -> None:
    llm_client = _FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    _tool_call(
                        "call_investigate",
                        "request_investigation",
                        '{"task":"Check default budget","missing_information":[]}',
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    _tool_call(
                        "call_final",
                        "final_answer",
                        '{"answer":"done","reports_used":[0]}',
                    )
                ]
            ),
        ]
    )
    session = AgentSession()
    investigator = _FakeInvestigationProvider()
    agent = MainAgent(
        llm_client=llm_client,  # type: ignore[arg-type]
        session=session,
        investigator=investigator,
        max_investigator_tool_calls=11,
        max_investigator_file_reads=13,
    )

    assert agent.run("What is the default budget?") == "done"

    assert investigator.tasks[0].max_tool_calls == 11
    assert investigator.tasks[0].max_file_reads == 13
    request_schema = next(
        tool
        for tool in llm_client.calls[0]["tools"]
        if tool["function"]["name"] == "request_investigation"
    )
    assert request_schema["function"]["parameters"]["properties"]["max_tool_calls"]["default"] == 11
    assert request_schema["function"]["parameters"]["properties"]["max_file_reads"]["default"] == 13


def test_main_agent_emits_tool_error_event_when_tool_fails() -> None:
    llm_client = _FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    _tool_call(
                        "call_unknown",
                        "unknown_tool",
                        "{}",
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    _tool_call(
                        "call_final",
                        "final_answer",
                        '{"answer":"done","reports_used":[]}',
                    )
                ]
            ),
        ]
    )
    events = _RecordingEventSink()
    agent = MainAgent(
        llm_client=llm_client,  # type: ignore[arg-type]
        session=AgentSession(),
        investigator=_FakeInvestigationProvider(),
        event_sink=events,
    )

    assert agent.run("Trigger a tool failure.") == "done"

    assert events.events[0][0] == "main.tool_error"
    assert events.events[0][1]["name"] == "unknown_tool"
    assert "unknown tool" in events.events[0][1]["error"]


def test_main_agent_requires_final_answer_tool() -> None:
    llm_client = _FakeLLMClient([LLMResponse(content="I should use a tool.")])
    session = AgentSession()
    agent = MainAgent(
        llm_client=llm_client,  # type: ignore[arg-type]
        session=session,
        investigator=_FakeInvestigationProvider(),
        max_rounds=1,
    )

    try:
        agent.run("Can you answer directly?")
    except RuntimeError as exc:
        assert "max_main_rounds" in str(exc)
    else:
        raise AssertionError("expected MainAgent to require final_answer")
