from __future__ import annotations

from typing import Any, Protocol


class EventSink(Protocol):
    def emit(self, event: str, payload: dict[str, Any]) -> None:
        ...


class NullEventSink:
    def emit(self, event: str, payload: dict[str, Any]) -> None:
        return None
