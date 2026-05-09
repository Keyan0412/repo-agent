from __future__ import annotations

from pathlib import Path

from repo_agent.agents.investigator_agent import InvestigatorAgent
from repo_agent.agents.main_agent import MainAgent
from repo_agent.cache import ReportStore
from repo_agent.llm.client import LLMClient
from repo_agent.llm.debug import JsonlLLMCallDebugRecorder
from repo_agent.runtime.config import AgentConfig
from repo_agent.runtime.events import EventSink
from repo_agent.runtime.session import AgentSession
from repo_agent.toolsets.investigator_toolset import build_investigator_tool_registry


def build_agent(
    *,
    repo_path: str | Path,
    config: AgentConfig | None = None,
    llm_client: LLMClient | None = None,
    simple_llm_client: LLMClient | None = None,
    event_sink: EventSink | None = None,
) -> MainAgent:
    repo = Path(repo_path).resolve()
    cfg = config or AgentConfig(repo_path=str(repo))
    session = AgentSession()
    report_store = ReportStore(repo, cache_dir=cfg.cache_dir)

    complex_client = llm_client or LLMClient.complex_from_env(
        model=cfg.complex_model,
        debug_recorder=JsonlLLMCallDebugRecorder.at_repo_cache(repo, cache_dir=cfg.cache_dir),
    )
    simple_client = simple_llm_client or LLMClient.simple_from_env(
        model=cfg.simple_model,
        debug_recorder=JsonlLLMCallDebugRecorder.at_repo_cache(repo, cache_dir=cfg.cache_dir),
    )
    investigator_registry = build_investigator_tool_registry(
        repo,
        max_file_chars=cfg.max_file_chars,
        ignored_names=set(cfg.ignored_dirs),
    )
    investigator = InvestigatorAgent(
        llm_client=simple_client,
        repo_path=repo,
        tool_registry=investigator_registry,
        event_sink=event_sink,
    )
    return MainAgent(
        llm_client=complex_client,
        session=session,
        investigator=investigator,
        max_rounds=cfg.max_main_rounds,
        report_store=report_store if cfg.cache_enabled else None,
        event_sink=event_sink,
    )


def run(
    repo_path: str | Path,
    user_query: str,
    *,
    config: AgentConfig | None = None,
    event_sink: EventSink | None = None,
) -> str:
    agent = build_agent(repo_path=repo_path, config=config, event_sink=event_sink)
    return agent.run(user_query)
