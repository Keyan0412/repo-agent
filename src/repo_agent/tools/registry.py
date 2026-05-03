from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolResult


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.get_openai_tool_schema() for tool in self.tools.values()]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name not in self.tools:
            raise KeyError(f"unknown tool: {tool_name}")
        return self.tools[tool_name].execute(arguments)
