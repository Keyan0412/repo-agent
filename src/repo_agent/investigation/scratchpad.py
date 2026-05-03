from __future__ import annotations


class InvestigationScratchpad:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def append(self, message: dict) -> None:
        self.messages.append(message)

    def to_messages(self) -> list[dict]:
        return list(self.messages)
