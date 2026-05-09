from __future__ import annotations

from repo_agent.cache import ReportStore
from repo_agent.runtime.session import AgentSession
from repo_agent.tools.main import (
    FinalAnswerTool,
    InvestigationProvider,
    RequestInvestigationTool,
)
from repo_agent.tools.registry import ToolRegistry

MAIN_AGENT_TOOLS = [
    "request_investigation",
    "final_answer",
]


def build_main_agent_tool_registry(
    *,
    session: AgentSession,
    investigation_provider: InvestigationProvider,
    user_query: str,
    report_store: ReportStore | None = None,
) -> ToolRegistry:
    return ToolRegistry(
        [
            RequestInvestigationTool(
                session=session,
                investigation_provider=investigation_provider,
                user_query=user_query,
                report_store=report_store,
            ),
            FinalAnswerTool(session),
        ]
    )
