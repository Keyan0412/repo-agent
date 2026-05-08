from __future__ import annotations

import json
from pathlib import Path

from repo_agent.llm.client import LLMClient
from repo_agent.tools.file import AskFileTool


class _FakeBackend:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        return _FakeCompletion(self.response)


class _FakeCompletion:
    def __init__(self, payload: dict) -> None:
        self.choices = [_FakeChoice(payload)]
        self._payload = payload

    def model_dump(self) -> dict:
        return {"choices": [{"message": self._payload}]}


class _FakeChoice:
    def __init__(self, payload: dict) -> None:
        self.message = _FakeMessage(payload)


class _FakeMessage:
    def __init__(self, payload: dict) -> None:
        self.content = payload.get("content", "")
        self.tool_calls = None


def test_ask_file_returns_structured_file_answer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main_agent.py").write_text(
        '"""MainAgent coordinates investigations and evidence updates."""\n',
        encoding="utf-8",
    )
    backend = _FakeBackend(
        {
            "content": json.dumps(
                {
                    "path": "main_agent.py",
                    "question": "What is implemented in this file?",
                    "answer": "The file is only a module docstring and has no implementation.",
                    "confidence": "high",
                    "implementation_status": "stub_or_placeholder",
                    "file_role": "Declares intended MainAgent responsibility.",
                    "observed_facts": [
                        {
                            "line_start": 1,
                            "line_end": 1,
                            "fact": "The file contains a module docstring.",
                        }
                    ],
                    "not_evidence": ["The docstring is not runtime behavior."],
                    "needs_cross_file_check": False,
                    "suggested_followups": [],
                }
            )
        }
    )
    client = LLMClient(model="qwen-plus", api_key="test-key", backend=backend)
    tool = AskFileTool(repo, client)

    result = tool.execute(
        {
            "path": "main_agent.py",
            "question": "What is implemented in this file?",
            "focus": "implementation_status",
        }
    )

    assert result.success is True
    payload = json.loads(result.content)
    assert payload["implementation_status"] == "stub_or_placeholder"
    assert payload["observed_facts"][0]["line_start"] == 1
    assert result.metadata["path"] == "main_agent.py"
    assert result.metadata["line_count"] == 1
    assert result.metadata["implementation_status"] == "stub_or_placeholder"
    assert "numbered_content" in result.metadata
    assert "line_map_content" in result.metadata
    assert result.metadata["valid_line_numbers"] == [1]
    assert backend.calls[0]["tool_choice"] == "none"
    assert "max_tokens" not in backend.calls[0]
    assert "<file_content path=\"main_agent.py\" trust=\"untrusted\" lines=\"1\">" in backend.calls[0]["messages"][1]["content"]
    assert '"lines": {' in backend.calls[0]["messages"][1]["content"]
    assert '"1": "\\"\\\"\\"MainAgent coordinates investigations and evidence updates.\\"\\\"\\""' in backend.calls[0]["messages"][1]["content"]


def test_ask_file_rejects_observed_fact_lines_outside_available_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "session.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    backend = _FakeBackend(
        {
            "content": json.dumps(
                {
                    "path": "session.py",
                    "question": "What is implemented?",
                    "answer": "Two assignments exist.",
                    "confidence": "high",
                    "implementation_status": "implemented",
                    "file_role": "Small module.",
                    "observed_facts": [
                        {
                            "line_start": 2,
                            "line_end": 3,
                            "fact": "The second assignment exists.",
                        }
                    ],
                    "not_evidence": [],
                    "needs_cross_file_check": False,
                    "suggested_followups": [],
                }
            )
        }
    )
    client = LLMClient(model="qwen-plus", api_key="test-key", backend=backend)
    tool = AskFileTool(repo, client)

    result = tool.execute({"path": "session.py", "question": "What is implemented?"})

    assert result.success is False
    assert "observed_facts lines outside available file content" in result.content


def test_ask_file_rejects_paths_outside_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    backend = _FakeBackend({"content": "{}"})
    client = LLMClient(model="qwen-plus", api_key="test-key", backend=backend)
    tool = AskFileTool(repo, client)

    result = tool.execute({"path": "../secret.txt", "question": "What is this?"})

    assert result.success is False
    assert "path escapes repository root" in result.content
    assert backend.calls == []
