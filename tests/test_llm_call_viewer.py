from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repo_agent_viewer.app import create_app
from repo_agent_viewer.store import LLMCallStore


def test_llm_call_store_lists_calls_in_reverse_timestamp_order(tmp_path: Path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-04T12:00:00+00:00",
                        "status": "success",
                        "model": "qwen-plus",
                        "request": {"messages": [{"role": "user", "content": "older request"}]},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-04T13:00:00+00:00",
                        "status": "error",
                        "model": "qwen-max",
                        "request": {"messages": [{"role": "user", "content": "newer request"}]},
                        "error": {"message": "boom"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    store = LLMCallStore(path)
    calls = store.list_calls()

    assert [call.id for call in calls] == [2, 1]
    assert calls[0].summary == "newer request"
    assert calls[1].summary == "older request"


def test_llm_call_viewer_api_returns_summary_and_detail(tmp_path: Path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    payload = {
        "timestamp": "2026-05-04T13:00:00+00:00",
        "status": "success",
        "model": "qwen-plus",
        "request": {"messages": [{"role": "user", "content": "inspect repo"}]},
        "response": {"content": "done"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    client = TestClient(create_app(jsonl_path=path))

    summary_response = client.get("/api/calls")
    assert summary_response.status_code == 200
    assert summary_response.json()[0]["summary"] == "inspect repo"

    detail_response = client.get("/api/calls/1")
    assert detail_response.status_code == 200
    assert detail_response.json()["payload"]["response"]["content"] == "done"


def test_llm_call_viewer_index_renders_call_list(tmp_path: Path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-04T13:00:00+00:00",
                "status": "success",
                "model": "qwen-plus",
                "request": {"messages": [{"role": "user", "content": "inspect repo"}]},
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(jsonl_path=path))
    response = client.get("/")

    assert response.status_code == 200
    assert "LLM Call Viewer" in response.text
    assert "inspect repo" in response.text
