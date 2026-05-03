from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from repo_agent.llm.schemas import LLMResponse
from repo_agent.tools.base import ToolResult
from repo_agent.tools.registry import ToolRegistry

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_DASHSCOPE_BASE_URL,
        timeout: float = 60.0,
        enable_thinking: bool = False,
        backend: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self._backend = backend

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        env_path: str | Path = ".env",
        timeout: float = 60.0,
    ) -> "LLMClient":
        cls._load_env_file(Path(env_path))

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

        resolved_model = model or os.getenv("REPO_AGENT_MODEL") or DEFAULT_MODEL
        base_url = os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL
        enable_thinking = cls._parse_bool_env(os.getenv("REPO_AGENT_ENABLE_THINKING"), default=False)
        return cls(
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            enable_thinking=enable_thinking,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        client = self._get_backend()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        merged_extra_body = {"enable_thinking": self.enable_thinking}
        if extra_body is not None:
            merged_extra_body.update(extra_body)
        payload["extra_body"] = merged_extra_body

        completion = client.chat.completions.create(**payload)
        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[dict[str, Any]] = []
        if getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                function = getattr(tool_call, "function", None)
                tool_calls.append(
                    {
                        "id": getattr(tool_call, "id", None),
                        "type": getattr(tool_call, "type", None),
                        "function": {
                            "name": getattr(function, "name", None),
                            "arguments": getattr(function, "arguments", None),
                        },
                    }
                )

        content = message.content or ""
        raw = completion.model_dump() if hasattr(completion, "model_dump") else json.loads(completion.json())
        return LLMResponse(content=content, tool_calls=tool_calls, raw=raw)

    def run_tool_calling_loop(
        self,
        *,
        system_prompt: str,
        user_content: str,
        tool_registry: ToolRegistry,
        max_tool_calls: int,
        max_files: int | None = None,
        temperature: float | None = 0,
    ) -> tuple[LLMResponse, list[dict[str, Any]]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = tool_registry.get_openai_tools()
        executed_tools: list[dict[str, Any]] = []
        remaining_calls = max_tool_calls
        files_read = 0
        budget_exhausted_reason: str | None = None

        while True:
            response = self.chat(
                messages=messages,
                tools=tools if budget_exhausted_reason is None else None,
                tool_choice="auto" if budget_exhausted_reason is None else "none",
                temperature=temperature,
            )
            if not response.tool_calls:
                return response, executed_tools

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                }
            )

            if remaining_calls <= 0:
                budget_exhausted_reason = (
                    f"Tool call budget exhausted. You have already used {max_tool_calls} tool calls. "
                    "Do not request more tools. Produce your best final answer from the evidence already collected. "
                    "Explicitly note any information gaps caused by the limited budget."
                )
                self._append_budget_exhausted_messages(
                    messages=messages,
                    tool_calls=response.tool_calls,
                    reason=budget_exhausted_reason,
                )
                continue

            for index, tool_call in enumerate(response.tool_calls):
                if remaining_calls <= 0:
                    budget_exhausted_reason = (
                        f"Tool call budget exhausted. You have already used {max_tool_calls} tool calls. "
                        "Do not request more tools. Produce your best final answer from the evidence already collected. "
                        "Explicitly note any information gaps caused by the limited budget."
                    )
                    self._append_budget_exhausted_messages(
                        messages=messages,
                        tool_calls=response.tool_calls[index:],
                        reason=budget_exhausted_reason,
                    )
                    break
                name = tool_call["function"]["name"]
                arguments_text = tool_call["function"]["arguments"] or "{}"
                try:
                    arguments = json.loads(arguments_text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"LLM generated invalid tool arguments for `{name}`: {arguments_text}"
                    ) from exc

                if name == "read_file" and max_files is not None:
                    if files_read >= max_files:
                        budget_exhausted_reason = (
                            f"File read budget exhausted. You have already read {max_files} files. "
                            "Do not request more read_file calls. Produce your best final answer from the evidence already collected. "
                            "Explicitly note any information gaps caused by the limited budget."
                        )
                        self._append_budget_exhausted_messages(
                            messages=messages,
                            tool_calls=response.tool_calls[index:],
                            reason=budget_exhausted_reason,
                            blocked_tool_call_id=tool_call["id"],
                        )
                        break
                    files_read += 1

                result = tool_registry.execute(name, arguments)
                executed_tools.append(
                    {
                        "id": tool_call["id"],
                        "name": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": result.content,
                    }
                )
                remaining_calls -= 1
            if budget_exhausted_reason is not None:
                continue

    def extract_json_object(self, content: str) -> dict[str, Any]:
        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                raise
            return json.loads(text[start : end + 1])

    def _get_backend(self) -> Any:
        if self._backend is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The `openai` package is required for LLMClient. Install project dependencies first."
                ) from exc

            self._backend = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._backend

    @staticmethod
    def _load_env_file(env_path: Path) -> None:
        if not env_path.exists():
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)

    @staticmethod
    def _parse_bool_env(value: str | None, *, default: bool) -> bool:
        if value is None:
            return default
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _append_budget_exhausted_messages(
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        reason: str,
        blocked_tool_call_id: str | None = None,
    ) -> None:
        for tool_call in tool_calls:
            content = reason
            if blocked_tool_call_id is not None and tool_call["id"] != blocked_tool_call_id:
                content = "Not executed because the current turn was stopped after budget exhaustion."
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "content": content,
                }
            )
