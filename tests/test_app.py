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
    assert agent.max_investigator_tool_calls == 30
    assert agent.max_investigator_file_reads == 15
    investigator = agent.investigator
    assert investigator.llm_client is simple_client
    assert set(investigator.tool_registry.tools) == {
        "list_dir",
        "find_files",
        "find_text",
        "trace_symbol",
        "read_files",
    }


def test_build_agent_passes_configured_llm_options_to_env_factories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    complex_client = LLMClient(model="complex-model", api_key="test-key", backend=object())
    simple_client = LLMClient(model="simple-model", api_key="test-key", backend=object())
    calls: dict[str, dict] = {}

    def complex_from_env(**kwargs):
        calls["complex"] = kwargs
        return complex_client

    def simple_from_env(**kwargs):
        calls["simple"] = kwargs
        return simple_client

    monkeypatch.setattr(LLMClient, "complex_from_env", staticmethod(complex_from_env))
    monkeypatch.setattr(LLMClient, "simple_from_env", staticmethod(simple_from_env))

    build_agent(
        repo_path=repo,
        config=AgentConfig(
            repo_path=str(repo),
            complex_model="qwen-max",
            simple_model="qwen-turbo",
            dashscope_api_key_env="CUSTOM_DASHSCOPE_KEY",
            dashscope_base_url="https://example.test/v1",
            enable_thinking=True,
        ),
    )

    assert calls["complex"]["model"] == "qwen-max"
    assert calls["simple"]["model"] == "qwen-turbo"
    for call in calls.values():
        assert call["api_key_env"] == "CUSTOM_DASHSCOPE_KEY"
        assert call["base_url"] == "https://example.test/v1"
        assert call["enable_thinking"] is True
