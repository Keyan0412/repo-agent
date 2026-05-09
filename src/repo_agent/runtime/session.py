from __future__ import annotations

from repo_agent.investigation import InvestigationReport


class AgentSession:
    def __init__(self) -> None:
        self.reports: list[InvestigationReport] = []
        self.final_answer: str | None = None
        self._task_counter = 1

    @property
    def investigation_reports(self) -> list[InvestigationReport]:
        return self.reports

    def next_task_id(self) -> str:
        task_id = f"T{self._task_counter:04d}"
        self._task_counter += 1
        return task_id
