from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repo_agent.cache import ReportStore
from repo_agent.llm.client import LLMClient
from repo_agent.runtime.events import EventSink, NullEventSink
from repo_agent.runtime.session import AgentSession
from repo_agent.toolsets.main_agent_toolset import build_main_agent_tool_registry
from repo_agent.tools.main import InvestigationProvider
from repo_agent.tools.registry import ToolRegistry


class MainAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        session: AgentSession,
        investigator: InvestigationProvider,
        *,
        prompt_path: Path | None = None,
        max_rounds: int | None = None,
        max_investigator_tool_calls: int = 30,
        max_investigator_file_reads: int = 15,
        report_store: ReportStore | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.session = session
        self.investigator = investigator
        self.max_rounds = max_rounds
        self.max_investigator_tool_calls = max_investigator_tool_calls
        self.max_investigator_file_reads = max_investigator_file_reads
        self.report_store = report_store
        self.event_sink = event_sink or NullEventSink()
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        self.prompt_path = prompt_path or prompts_dir / "main_agent.md"

    def run(self, user_query: str) -> str:
        tool_registry = build_main_agent_tool_registry(
            session=self.session,
            investigation_provider=self.investigator,
            user_query=user_query,
            report_store=self.report_store,
            default_max_tool_calls=self.max_investigator_tool_calls,
            default_max_file_reads=self.max_investigator_file_reads,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._round_context(user_query)},
        ]

        rounds = 0
        while self.max_rounds is None or rounds < self.max_rounds:
            rounds += 1
            response = self.llm_client.chat(
                messages=messages,
                tools=tool_registry.get_openai_tools(),
                tool_choice="auto",
                temperature=0,
            )
            if response.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": response.tool_calls,
                    }
                )
                self._execute_tool_calls(
                    tool_calls=response.tool_calls,
                    messages=messages,
                    tool_registry=tool_registry,
                )
                if self.session.final_answer is not None:
                    return self.session.final_answer
                continue

            messages.append({"role": "assistant", "content": response.content or ""})

        raise RuntimeError("MainAgent reached max_main_rounds without final_answer")

    def _execute_tool_calls(
        self,
        *,
        tool_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tool_registry: ToolRegistry,
    ) -> None:
        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            arguments_text = tool_call["function"]["arguments"] or "{}"
            try:
                arguments = json.loads(arguments_text)
                self._emit_tool_event(name=name, arguments=arguments)
                result = tool_registry.execute(name, arguments)
                content = result.content
            except Exception as exc:
                self.event_sink.emit(
                    "main.tool_error",
                    {
                        "name": name,
                        "error": str(exc),
                    },
                )
                content = f"Tool `{name}` failed: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": name,
                    "content": content,
                }
            )
            if self.session.final_answer is not None:
                break

    def _round_context(self, user_query: str) -> str:
        return f"用户问题:\n{user_query}\n\n当前调查报告:\n无"

    def _emit_tool_event(self, *, name: str, arguments: dict[str, Any]) -> None:
        if name == "request_investigation":
            self.event_sink.emit(
                "main.investigation",
                {
                    "task": arguments.get("task", ""),
                    "missing_information": arguments.get("missing_information", []),
                },
            )
        elif name == "final_answer":
            self.event_sink.emit(
                "main.final_answer",
                {
                    "answer": arguments.get("answer", ""),
                    "reports_used": arguments.get("reports_used", []),
                },
            )

    def _system_prompt(self) -> str:
        prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        if prompt and prompt != "# Main Agent Prompt":
            return prompt
        return (
            "你是 MainAgent。你的职责是围绕代码仓库问题进行调查调度和最终回答。\n\n"
            "允许使用的工具：\n"
            "- request_investigation：当信息不足时，请 InvestigatorAgent 直接检查仓库并返回调查报告。\n"
            "- final_answer：当已有调查报告足以回答用户时，输出最终回答。\n\n"
            "硬性规则：\n"
            "- 不要直接读取仓库文件或搜索代码；所有仓库检查都通过 request_investigation 进行。\n"
            "- 你直接通过 InvestigatorAgent 获取调查报告，不需要经过额外的分析或记忆层。\n"
            "- 如果调查结果不足以回答用户，可以继续发起更聚焦的 request_investigation。\n"
            "- final_answer 应综合已有调查报告，必要时在 reports_used 中列出使用的报告编号。\n"
            "- 每轮优先只调用一个工具，除非 final_answer 已经足够。"
        )
