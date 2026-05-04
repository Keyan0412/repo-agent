from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMCallRecord:
    id: int
    timestamp: str
    status: str
    model: str
    summary: str
    payload: dict[str, Any]


class LLMCallStore:
    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)

    def list_calls(self) -> list[LLMCallRecord]:
        records = [self._to_record(index, payload) for index, payload in self._iter_payloads()]
        return sorted(records, key=lambda record: (self._sort_key(record.timestamp), record.id), reverse=True)

    def get_call(self, call_id: int) -> LLMCallRecord | None:
        for index, payload in self._iter_payloads():
            if index == call_id:
                return self._to_record(index, payload)
        return None

    def _iter_payloads(self) -> list[tuple[int, dict[str, Any]]]:
        if not self.jsonl_path.exists():
            return []

        items: list[tuple[int, dict[str, Any]]] = []
        for index, raw_line in enumerate(self.jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid LLM call record at line {index}: expected JSON object")
            items.append((index, payload))
        return items

    def _to_record(self, call_id: int, payload: dict[str, Any]) -> LLMCallRecord:
        timestamp = str(payload.get("timestamp", ""))
        status = str(payload.get("status", "unknown"))
        model = str(payload.get("model", ""))
        return LLMCallRecord(
            id=call_id,
            timestamp=timestamp,
            status=status,
            model=model,
            summary=self._build_summary(payload),
            payload=payload,
        )

    @staticmethod
    def _build_summary(payload: dict[str, Any]) -> str:
        request = payload.get("request")
        if not isinstance(request, dict):
            return ""

        messages = request.get("messages")
        if not isinstance(messages, list):
            return ""

        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return _truncate(content.strip())
        return ""

    @staticmethod
    def _sort_key(timestamp: str) -> datetime:
        if not timestamp:
            return datetime.min
        normalized = timestamp.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return datetime.min


def _truncate(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
