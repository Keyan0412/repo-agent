from __future__ import annotations

from pathlib import Path

from repo_agent.app import build_agent
from repo_agent.llm.client import LLMClient
from repo_agent.runtime.config import AgentConfig


def test_build_agent_uses_complex_client_for_main_and_simple_client_for_investigator(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    complex_client = LLMClient(model="complex-model", api_key="test-key", backend=object())
    simple_client = LLMClient(model="simple-model", api_key="test-key", backend=object())

    agent = build_agent(
        repo_path=repo,
        config=AgentConfig(repo_path=str(repo)),
        llm_client=complex_client,
        simple_llm_client=simple_client,
    )

    assert agent.llm_client is complex_client
    investigator = agent.investigator
    assert investigator.llm_client is simple_client
    assert set(investigator.tool_registry.tools) == {
        "read_repo_tree",
        "find_text",
        "trace_symbol",
        "read_file",
    }
