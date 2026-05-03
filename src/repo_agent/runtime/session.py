from __future__ import annotations

from repo_agent.evidence import EvidenceGraph
from repo_agent.investigation import InvestigationReport


class AgentSession:
    def __init__(self) -> None:
        self.evidence_graph = EvidenceGraph()
        self.reports: list[InvestigationReport] = []
        self.repo_profile: str | None = None
