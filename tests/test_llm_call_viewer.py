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
from repo_agent_viewer.store import RunStore


def _write_run(runs_dir: Path, run_id: str = "run-1") -> Path:
    run_dir = runs_dir / run_id
    (run_dir / "investigations").mkdir(parents=True)
    summary = {
        "run_id": run_id,
        "status": "success",
        "user_query": "inspect repo",
        "started_at": "2026-05-04T12:00:00+00:00",
        "ended_at": "2026-05-04T12:01:00+00:00",
        "total_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "agents": {"main": {"call_count": 2, "total_tokens": 8}},
        "investigations": [
            {
                "id": "T0001",
                "task": "Inspect tools",
                "usage": {"total_tokens": 7},
                "calls": [
                    {
                        "call_index": 2,
                        "agent": "investigator",
                        "timestamp": "2026-05-04T12:00:30+00:00",
                        "usage": {"total_tokens": 7},
                        "tool_calls": [{"name": "read_files", "arguments": {"files": [{"path": "a.py"}]}}],
                        "tool_result_previews": [],
                        "content_preview": "{}",
                    }
                ],
                "tool_counts": {"read_files": 1},
                "read_files_call_count": 1,
                "summarize_files_call_count": 0,
                "read_files_paths": ["a.py"],
                "issues": [],
            }
        ],
        "issues": [],
        "final_answer": "done",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "investigations" / "T0001.json").write_text(
        json.dumps(summary["investigations"][0]),
        encoding="utf-8",
    )
    raw_records = [
        {
            "timestamp": "2026-05-04T12:00:00+00:00",
            "run_id": run_id,
            "call_index": 1,
            "status": "success",
            "agent": "main",
            "model": "qwen-plus",
            "request": {
                "messages": [
                    {"role": "system", "content": "MainAgent"},
                    {"role": "user", "content": "inspect repo"},
                ]
            },
            "response": {
                "usage": {"total_tokens": 8},
                "tool_calls": [
                    {
                        "id": "call_main_investigation",
                        "function": {
                            "name": "request_investigation",
                            "arguments": "{\"task\":\"Inspect tools\"}",
                        }
                    }
                ],
            },
        },
        {
            "timestamp": "2026-05-04T12:01:00+00:00",
            "run_id": run_id,
            "call_index": 2,
            "status": "success",
            "agent": "investigator",
            "model": "qwen-plus",
            "request": {
                "messages": [
                    {"role": "system", "content": "你是 InvestigatorAgent。"},
                    {"role": "user", "content": "Inspect tools"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_read",
                                "function": {
                                    "name": "read_files",
                                    "arguments": "{\"files\":[{\"path\":\"a.py\"}]}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read",
                        "name": "read_files",
                        "content": "file content for a.py",
                    },
                ]
            },
            "response": {
                "usage": {"total_tokens": 7},
                "content": "{\"answer\":\"done\"}",
                "tool_calls": [],
            },
        },
        {
            "timestamp": "2026-05-04T12:02:00+00:00",
            "run_id": run_id,
            "call_index": 3,
            "status": "success",
            "agent": "main",
            "model": "qwen-plus",
            "request": {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_main_investigation",
                                "function": {
                                    "name": "request_investigation",
                                    "arguments": "{\"task\":\"Inspect tools\"}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_main_investigation",
                        "name": "request_investigation",
                        "content": "调查结果 [0] R-T0001:\n总结: done",
                    }
                ]
            },
            "response": {"usage": {"total_tokens": 7}, "content": "final answer", "tool_calls": []},
        },
    ]
    (run_dir / "raw_llm_calls.jsonl").write_text(
        "\n".join(json.dumps(record) for record in raw_records),
        encoding="utf-8",
    )
    return run_dir


def test_run_store_lists_runs_in_reverse_end_time_order(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "older")
    newer_dir = _write_run(runs_dir, "newer")
    payload = json.loads((newer_dir / "run_summary.json").read_text(encoding="utf-8"))
    payload["ended_at"] = "2026-05-04T13:00:00+00:00"
    (newer_dir / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    runs = RunStore(runs_dir).list_runs()

    assert [run.run_id for run in runs] == ["newer", "older"]
    assert runs[0].user_query == "inspect repo"


def test_run_viewer_index_run_and_investigation_pages(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir)
    client = TestClient(create_app(runs_dir=runs_dir))

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert "repo-agent Runs" in index_response.text
    assert "inspect repo" in index_response.text

    run_response = client.get("/runs/run-1")
    assert run_response.status_code == 200
    assert "MainAgent Messages" in run_response.text
    assert "Reports Received By MainAgent" in run_response.text
    assert "R-T0001" in run_response.text
    assert "assistant" in run_response.text
    assert "request_investigation" in run_response.text
    assert "task:" in run_response.text
    assert "Tool result" in run_response.text
    assert '"function"' not in run_response.text

    investigation_response = client.get("/runs/run-1/investigations/T0001")
    assert investigation_response.status_code == 200
    assert "Investigator Messages" in investigation_response.text
    assert "a.py" in investigation_response.text
    assert "file content for a.py" in investigation_response.text
    assert "read_files" in investigation_response.text
    assert "Tool result" in investigation_response.text
    assert '"function"' not in investigation_response.text


def test_run_viewer_api_returns_run_and_investigation(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir)
    client = TestClient(create_app(runs_dir=runs_dir))

    runs_response = client.get("/api/runs")
    assert runs_response.status_code == 200
    assert runs_response.json()[0]["run_id"] == "run-1"

    run_response = client.get("/api/runs/run-1")
    assert run_response.status_code == 200
    assert run_response.json()["reports"][0]["investigation_id"] == "T0001"
    assert run_response.json()["main_messages"][-1]["role"] == "assistant"

    investigation_response = client.get("/api/runs/run-1/investigations/T0001")
    assert investigation_response.status_code == 200
    payload = investigation_response.json()
    assert payload["investigation"]["read_files_paths"] == ["a.py"]
    tool_call = payload["messages"][2]["tool_calls"][0]
    assert tool_call["name"] == "read_files"
    assert tool_call["arguments"] == {"files": [{"path": "a.py"}]}
    assert tool_call["has_result"] is True
    assert tool_call["result_content"] == "file content for a.py"


def test_run_store_skips_invalid_summary_and_recovers_raw_tool_calls(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    bad_dir = runs_dir / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "run_summary.json").write_text("", encoding="utf-8")

    raw_only_dir = runs_dir / "raw-only"
    raw_only_dir.mkdir(parents=True)
    raw_records = [
        {
            "timestamp": "2026-05-04T12:00:00+00:00",
            "run_id": "raw-only",
            "call_index": 1,
            "status": "success",
            "agent": "main",
            "model": "qwen-plus",
            "request": {
                "messages": [
                    {"role": "system", "content": "你是 MainAgent。"},
                    {"role": "user", "content": "当前用户问题:\ninspect repo\n\n当前调查报告:\n无"},
                ]
            },
            "response": {
                "usage": {"total_tokens": 3},
                "content": "",
                "tool_calls": [],
                "raw": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_raw",
                                        "type": "function",
                                        "function": {
                                            "name": "request_investigation",
                                            "arguments": "```json\n{\"task\":\"Inspect tools\",}\n```",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            },
        },
        {
            "timestamp": "2026-05-04T12:01:00+00:00",
            "run_id": "raw-only",
            "call_index": 2,
            "status": "success",
            "agent": "investigator",
            "model": "qwen-plus",
            "request": {
                "messages": [
                    {"role": "system", "content": "你是 InvestigatorAgent。"},
                    {"role": "user", "content": "Inspect tools"},
                ]
            },
            "response": {
                "usage": {"total_tokens": 4},
                "content": "{\"answer\":\"done\"}",
                "tool_calls": [],
            },
        },
    ]
    (raw_only_dir / "raw_llm_calls.jsonl").write_text(
        "\n".join(json.dumps(record) for record in raw_records),
        encoding="utf-8",
    )

    store = RunStore(runs_dir)
    runs = store.list_runs()

    assert [run.run_id for run in runs] == ["raw-only"]
    assert runs[0].user_query == "inspect repo"
    view = store.get_main_view("raw-only")
    assert view is not None
    assert view["summary"]["investigations"][0]["task"] == "Inspect tools"
    assert view["main_messages"][-1]["tool_calls"][0]["name"] == "request_investigation"
    assert view["main_messages"][-1]["tool_calls"][0]["arguments"] == {"task": "Inspect tools"}
