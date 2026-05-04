from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from repo_agent.cache.paths import CachePaths
from repo_agent.llm.schemas import LLMResponse


class LLMCallDebugRecorder(Protocol):
    def record_success(self, *, model: str, payload: dict[str, Any], response: LLMResponse) -> None: ...

    def record_error(self, *, model: str, payload: dict[str, Any], error: Exception) -> None: ...


class JsonlLLMCallDebugRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def at_repo_cache(
        cls,
        repo_path: str | Path,
        *,
        cache_dir: str = ".cache/repo-agent",
    ) -> "JsonlLLMCallDebugRecorder":
        paths = CachePaths(Path(repo_path), cache_dir)
        return cls(paths.llm_calls_path)

    def record_success(self, *, model: str, payload: dict[str, Any], response: LLMResponse) -> None:
        self._append_record(
            {
                "timestamp": _utc_timestamp(),
                "status": "success",
                "model": model,
                "request": payload,
                "response": {
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                    "raw": response.raw,
                },
            }
        )

    def record_error(self, *, model: str, payload: dict[str, Any], error: Exception) -> None:
        self._append_record(
            {
                "timestamp": _utc_timestamp(),
                "status": "error",
                "model": model,
                "request": payload,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                },
            }
        )

    def _append_record(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            fh.write("\n")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return repr(value)
