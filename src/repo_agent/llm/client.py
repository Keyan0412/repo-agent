from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from repo_agent.llm.debug import LLMCallDebugRecorder
from repo_agent.llm.schemas import LLMResponse
from repo_agent.tools.registry import ToolRegistry

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall
except ImportError as exc:
    raise RuntimeError(
        "The `openai` package is required for LLMClient. Install project dependencies first."
    ) from exc

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
        debug_recorder: LLMCallDebugRecorder | None = None,
    ) -> None:
        if backend is None:
            backend = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self._backend = backend
        self.debug_recorder = debug_recorder

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        env_path: str | Path = ".env",
        timeout: float = 60.0,
        debug_recorder: LLMCallDebugRecorder | None = None,
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
            debug_recorder=debug_recorder,
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

        try:
            completion = self._backend.chat.completions.create(**payload)
            if not isinstance(completion, ChatCompletion):
                print("here should not use streaming mode")
                exit(1)

            message = completion.choices[0].message

            # construct tool_call information
            tool_calls = []
            for tool_call in message.tool_calls or []:
                if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    raise RuntimeError(f"Unexpected tool call type: {type(tool_call).__name__}")

                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )

            response = LLMResponse(content=message.content or "", tool_calls=tool_calls, raw=completion.model_dump())
        except Exception as unknown_exc:
            if self.debug_recorder is not None:
                self.debug_recorder.record_error(model=self.model, payload=payload, error=unknown_exc)
            raise

        if self.debug_recorder is not None:
            self.debug_recorder.record_success(model=self.model, payload=payload, response=response)
        return response

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
                budget_exhausted_reason: str = (
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
                    budget_exhausted_reason: str = (
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

                # get function name and argument
                name = tool_call["function"]["name"]
                arguments_text = tool_call["function"]["arguments"] or "{}"
                try:
                    arguments = json.loads(arguments_text)
                except json.JSONDecodeError as decode_error:
                    raise RuntimeError(
                        f"LLM generated invalid tool arguments for `{name}`: {arguments_text}"
                    ) from decode_error

                if name == "read_file" and max_files is not None:
                    if files_read >= max_files:
                        budget_exhausted_reason: str = (
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

        raise RuntimeError("run_tool_calling_loop exited unexpectedly")

    @staticmethod
    def extract_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as decode_error:
            raise RuntimeError(f"Failed to parse JSON object from LLM output: {text}") from decode_error

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
