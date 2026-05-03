from __future__ import annotations

from pydantic import BaseModel


class AgentConfig(BaseModel):
    model: str
    repo_path: str
    dashscope_api_key_env: str = "DASHSCOPE_API_KEY"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    enable_thinking: bool = False
    max_main_rounds: int = 10
    max_investigator_tool_calls: int = 8
    max_file_chars: int = 50_000
