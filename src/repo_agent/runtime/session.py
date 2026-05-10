from __future__ import annotations

from dataclasses import dataclass

from repo_agent.investigation import InvestigationReport
from repo_agent.runtime.text import strip_surrogates


@dataclass
class ConversationMessage:
    role: str
    content: str


class AgentSession:
    def __init__(self) -> None:
        self.reports: list[InvestigationReport] = []
        self.final_answer: str | None = None
        self.conversation: list[ConversationMessage] = []
        self._task_counter = 1

    @property
    def investigation_reports(self) -> list[InvestigationReport]:
        return self.reports

    @property
    def conversation_messages(self) -> list[ConversationMessage]:
        return self.conversation

    def begin_user_turn(self, user_query: str) -> None:
        self.final_answer = None
        self.conversation.append(ConversationMessage(role="user", content=strip_surrogates(user_query).strip()))

    def record_assistant_answer(self, answer: str) -> None:
        self.conversation.append(ConversationMessage(role="assistant", content=strip_surrogates(answer).strip()))

    def next_task_id(self) -> str:
        task_id = f"T{self._task_counter:04d}"
        self._task_counter += 1
        return task_id
