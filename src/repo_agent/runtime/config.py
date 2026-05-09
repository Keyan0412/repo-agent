from __future__ import annotations

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    repo_path: str
    complex_model: str | None = None
    simple_model: str | None = None
    dashscope_api_key_env: str = "DASHSCOPE_API_KEY"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    enable_thinking: bool = False
    max_main_rounds: int | None = None
    max_investigator_tool_calls: int = 8
    max_file_chars: int = 50_000
    cache_enabled: bool = True
    cache_dir: str = ".cache/repo-agent"
    ignored_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            ".venv",
            "node_modules",
            "dist",
            "build",
            ".cache",
        ]
    )
