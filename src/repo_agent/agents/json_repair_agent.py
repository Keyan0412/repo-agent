from __future__ import annotations

import json
from typing import Any

from repo_agent.llm.client import LLMClient


class JsonRepairAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def repair_json(
        self,
        *,
        raw_content: str,
        target_name: str,
        json_schema: dict[str, Any],
        error: Exception,
    ) -> str:
        response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_content(
                        raw_content=raw_content,
                        target_name=target_name,
                        json_schema=json_schema,
                        error=error,
                    ),
                },
            ],
            tool_choice="none",
            temperature=0,
        )
        repaired = response.content.strip()
        if not repaired:
            raise RuntimeError("JsonRepairAgent returned empty content")
        return repaired

    @staticmethod
    def _system_prompt() -> str:
        return """
你是 JsonRepairAgent。你负责修复格式不正确的结构化 JSON 输出。

你的职责只限于格式修复。
必须保留原始含义和证据。
不得添加新的事实、文件、行号、结论或证据。
除非某些事实无法用目标 schema 表示，否则不得删除事实。

允许的修复：
- 移除 Markdown code fence
- 移除 JSON object 前后的说明性文字
- 把类 JSON 文本转换为合法 JSON
- 将 object key 规范化到目标 schema
- 当 schema 明确允许时，为缺失的可选字段补 null、[] 或 0
- 将明显的枚举大小写转换为 schema 值，例如 Medium 转成 medium
- 只有当值显然是数字时，才把数字字符串转换为整数

只返回一个严格 JSON object。
不要把 JSON 包在 Markdown fence 中。
不要包含解释、注释或额外文本。
""".strip()

    @staticmethod
    def _user_content(
        *,
        raw_content: str,
        target_name: str,
        json_schema: dict[str, Any],
        error: Exception,
    ) -> str:
        return f"""
目标输出类型:
{target_name}

目标 JSON schema:
{json.dumps(json_schema, ensure_ascii=False, indent=2)}

原始解析错误:
{type(error).__name__}: {error}

需要修复的原始模型输出:
<raw_output>
{raw_content}
</raw_output>

请把原始输出修复为一个符合目标 schema 的 JSON object。
保留原始内容含义。不要添加证据或结论。
""".strip()
