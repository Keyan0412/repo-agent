from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from repo_agent.llm.debug import LLMCallDebugRecorder
from repo_agent.llm.schemas import LLMResponse
from repo_agent.runtime.text import strip_surrogates_from_json
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
DEFAULT_COMPLEX_MODEL = DEFAULT_MODEL
DEFAULT_SIMPLE_MODEL = "qwen-turbo"


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
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str | None = None,
        enable_thinking: bool | None = None,
        timeout: float = 60.0,
        debug_recorder: LLMCallDebugRecorder | None = None,
    ) -> "LLMClient":
        cls._load_env_file(Path(env_path))

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is not set")

        resolved_model = model or os.getenv("REPO_AGENT_MODEL") or DEFAULT_MODEL
        resolved_base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL
        resolved_enable_thinking = (
            enable_thinking
            if enable_thinking is not None
            else cls._parse_bool_env(os.getenv("REPO_AGENT_ENABLE_THINKING"), default=False)
        )
        return cls(
            model=resolved_model,
            api_key=api_key,
            base_url=resolved_base_url,
            timeout=timeout,
            enable_thinking=resolved_enable_thinking,
            debug_recorder=debug_recorder,
        )

    @classmethod
    def complex_from_env(
        cls,
        *,
        model: str | None = None,
        env_path: str | Path = ".env",
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str | None = None,
        enable_thinking: bool | None = None,
        timeout: float = 60.0,
        debug_recorder: LLMCallDebugRecorder | None = None,
    ) -> "LLMClient":
        cls._load_env_file(Path(env_path))
        resolved_model = (
            model
            or os.getenv("REPO_AGENT_COMPLEX_MODEL")
            or os.getenv("REPO_AGENT_MODEL")
            or DEFAULT_COMPLEX_MODEL
        )
        return cls.from_env(
            model=resolved_model,
            env_path=env_path,
            api_key_env=api_key_env,
            base_url=base_url,
            enable_thinking=enable_thinking,
            timeout=timeout,
            debug_recorder=debug_recorder,
        )

    @classmethod
    def simple_from_env(
        cls,
        *,
        model: str | None = None,
        env_path: str | Path = ".env",
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str | None = None,
        enable_thinking: bool | None = None,
        timeout: float = 60.0,
        debug_recorder: LLMCallDebugRecorder | None = None,
    ) -> "LLMClient":
        cls._load_env_file(Path(env_path))
        resolved_model = (
            model
            or os.getenv("REPO_AGENT_SIMPLE_MODEL")
            or os.getenv("REPO_AGENT_MODEL")
            or DEFAULT_SIMPLE_MODEL
        )
        return cls.from_env(
            model=resolved_model,
            env_path=env_path,
            api_key_env=api_key_env,
            base_url=base_url,
            enable_thinking=enable_thinking,
            timeout=timeout,
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
        tools_supplied = bool(tools)
        if tools_supplied:
            payload["tools"] = tools
        if tool_choice is not None and tools_supplied:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        merged_extra_body = {"enable_thinking": self.enable_thinking}
        if extra_body is not None:
            merged_extra_body.update(extra_body)
        payload["extra_body"] = merged_extra_body
        payload = strip_surrogates_from_json(payload)

        try:
            completion = self._backend.chat.completions.create(**payload)
            message = completion.choices[0].message

            # construct tool_call information
            tool_calls = []
            for tool_call in message.tool_calls or []:
                function = getattr(tool_call, "function", None)
                if function is None:
                    raise RuntimeError(f"Unexpected tool call type: {type(tool_call).__name__}")

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

            raw = completion.model_dump() if hasattr(completion, "model_dump") else json.loads(completion.json())
            response = LLMResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                usage=self._extract_usage(completion=completion, raw=raw),
                raw=raw,
            )
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
        on_tool_result: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[LLMResponse, list[dict[str, Any]]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = tool_registry.get_openai_tools()
        executed_tools: list[dict[str, Any]] = []
        remaining_calls = max_tool_calls
        files_read = 0
        disabled_tool_names: set[str] = set()
        budget_exhausted_reason: str | None = None
        post_budget_tool_call_retries = 0

        while True:
            tools_allowed = budget_exhausted_reason is None
            active_tools = self._filter_tools(tools, disabled_tool_names)
            response = self.chat(
                messages=messages,
                tools=active_tools,
                tool_choice="auto" if tools_allowed and active_tools else "none",
                temperature=temperature,
            )
            if not tools_allowed and response.tool_calls:
                post_budget_tool_call_retries += 1
                if post_budget_tool_call_retries > 3:
                    return LLMResponse(
                        content=budget_exhausted_reason
                        or "预算已耗尽。请基于已收集的证据生成当前最佳最终回答。",
                        tool_calls=[],
                        usage=response.usage,
                        raw=response.raw,
                    ), executed_tools
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": response.tool_calls,
                    }
                )
                retry_reason = (
                    f"{budget_exhausted_reason or '预算已耗尽。'}"
                    "你刚才仍然请求了工具，但工具预算已经耗尽，这些工具不会被执行。"
                    "现在必须直接返回最终答案，不要再调用任何工具。"
                )
                self._append_budget_exhausted_messages(
                    messages=messages,
                    tool_calls=response.tool_calls,
                    reason=retry_reason,
                )
                messages.append({"role": "user", "content": retry_reason})
                continue
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
                    f"工具调用预算已耗尽。你已经使用了 {max_tool_calls} 次工具调用。"
                    "不要再请求更多工具。请基于已收集的证据生成当前最佳最终回答。"
                    "需要明确说明由预算限制造成的信息缺口。"
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
                        f"工具调用预算已耗尽。你已经使用了 {max_tool_calls} 次工具调用。"
                        "不要再请求更多工具。请基于已收集的证据生成当前最佳最终回答。"
                        "需要明确说明由预算限制造成的信息缺口。"
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

                if name in disabled_tool_names:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": name,
                            "content": f"`{name}` 已禁用，因为它的预算已经耗尽。请使用其它可用工具，或基于已有证据回答。",
                        }
                    )
                    continue

                file_access_cost = self._file_access_cost(name=name, arguments=arguments)
                if file_access_cost and max_files is not None:
                    if files_read + file_access_cost > max_files:
                        budget_exhausted_reason: str = (
                            f"文件访问预算已耗尽。你最多可以读取或总结 {max_files} 个文件，"
                            f"当前请求会超过预算。"
                            "不要再请求 read_files。"
                            "请基于已收集的证据生成当前最佳最终回答。"
                            "需要明确说明由预算限制造成的信息缺口。"
                        )
                        self._append_budget_exhausted_messages(
                            messages=messages,
                            tool_calls=response.tool_calls[index:],
                            reason=budget_exhausted_reason,
                            blocked_tool_call_id=tool_call["id"],
                        )
                        break
                    files_read += file_access_cost

                result = tool_registry.execute(name, arguments)
                executed_tools.append(
                    {
                        "id": tool_call["id"],
                        "name": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )
                if on_tool_result is not None:
                    on_tool_result(executed_tools[-1])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": result.content,
                    }
                )
                remaining_calls -= 1
                if remaining_calls <= 0:
                    budget_exhausted_reason = (
                        f"工具调用预算已耗尽。你已经使用了 {max_tool_calls} 次工具调用。"
                        "不要再请求更多工具。请基于已收集的证据生成当前最佳最终回答。"
                        "需要明确说明由预算限制造成的信息缺口。"
                    )
                    self._append_budget_exhausted_messages(
                        messages=messages,
                        tool_calls=response.tool_calls[index + 1 :],
                        reason=budget_exhausted_reason,
                    )
                    messages.append({"role": "user", "content": budget_exhausted_reason})
                    break
                if file_access_cost and max_files is not None and files_read >= max_files:
                    budget_exhausted_reason = (
                        f"文件访问预算已耗尽。你已经读取或总结了 {max_files} 个文件。"
                        "不要再请求 read_files。"
                        "请基于已收集的证据生成当前最佳最终回答。"
                        "需要明确说明由预算限制造成的信息缺口。"
                    )
                    self._append_budget_exhausted_messages(
                        messages=messages,
                        tool_calls=response.tool_calls[index + 1 :],
                        reason=budget_exhausted_reason,
                    )
                    messages.append({"role": "user", "content": budget_exhausted_reason})
                    break

            if budget_exhausted_reason is not None:
                continue

        raise RuntimeError("run_tool_calling_loop exited unexpectedly")

    def extract_json_object(
        self,
        content: str,
        *,
        repair_agent: Any | None = None,
        target_name: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as decode_error:
            if repair_agent is None:
                raise RuntimeError(f"Failed to parse JSON object from LLM output: {text}") from decode_error
            if target_name is None or json_schema is None:
                raise RuntimeError(
                    "JSON repair requires target_name and json_schema"
                ) from decode_error

            repaired_content = repair_agent.repair_json(
                raw_content=content,
                target_name=target_name,
                json_schema=json_schema,
                error=decode_error,
            )
            repaired_text = repaired_content.strip()
            try:
                return json.loads(repaired_text)
            except json.JSONDecodeError as repair_decode_error:
                raise RuntimeError(
                    "Failed to parse JSON object from LLM output after repair: "
                    f"{repaired_text}"
                ) from repair_decode_error

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
    def _extract_usage(*, completion: Any, raw: dict[str, Any]) -> dict[str, int]:
        usage = raw.get("usage")
        if usage is None:
            usage = getattr(completion, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            elif usage is not None and not isinstance(usage, dict):
                usage = getattr(usage, "__dict__", None)
        if not isinstance(usage, dict):
            return {}

        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        total_tokens = usage.get("total_tokens")
        normalized: dict[str, int] = {}
        if isinstance(prompt_tokens, int):
            normalized["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            normalized["completion_tokens"] = completion_tokens
        if isinstance(total_tokens, int):
            normalized["total_tokens"] = total_tokens
        return normalized

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
                content = "未执行，因为当前轮次在预算耗尽后已停止。"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "content": content,
                }
            )

    @staticmethod
    def _filter_tools(
        tools: list[dict[str, Any]],
        disabled_tool_names: set[str],
    ) -> list[dict[str, Any]]:
        if not disabled_tool_names:
            return tools
        return [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") not in disabled_tool_names
        ]

    @staticmethod
    def _file_access_cost(*, name: str, arguments: dict[str, Any]) -> int:
        if name in {"read_file", "summarize_file"}:
            return 1
        if name == "read_files":
            files = arguments.get("files")
            if isinstance(files, list):
                paths = [
                    str(item.get("path") or "")
                    for item in files
                    if isinstance(item, dict)
                ]
                return len({path for path in paths if path})
            return 1
        if name == "summarize_files":
            paths = arguments.get("paths")
            if isinstance(paths, list):
                return len({str(path) for path in paths if str(path)})
            return 1
        return 0
