from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from repo_agent_viewer.store import LLMCallRecord, LLMCallStore

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))


def create_app(*, jsonl_path: str | Path) -> FastAPI:
    app = FastAPI(title="repo-agent LLM Call Viewer")
    store = LLMCallStore(jsonl_path)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        calls = store.list_calls()
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "calls": [_summary_payload(call) for call in calls],
                "jsonl_path": str(Path(jsonl_path)),
            },
        )

    @app.get("/api/calls")
    def list_calls() -> list[dict[str, Any]]:
        return [_summary_payload(call) for call in store.list_calls()]

    @app.get("/api/calls/{call_id}")
    def get_call(call_id: int) -> dict[str, Any]:
        call = store.get_call(call_id)
        if call is None:
            raise HTTPException(status_code=404, detail="Call not found")
        return {
            "id": call.id,
            "timestamp": call.timestamp,
            "status": call.status,
            "model": call.model,
            "summary": call.summary,
            "payload": call.payload,
            "pretty_json": json.dumps(call.payload, ensure_ascii=False, indent=2),
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View repo-agent LLM call logs in a browser.")
    parser.add_argument(
        "--jsonl-path",
        default=".cache/repo-agent/llm_calls.jsonl",
        help="Path to the llm_calls.jsonl file.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(jsonl_path=args.jsonl_path)
    uvicorn.run(app, host=args.host, port=args.port)


def _summary_payload(call: LLMCallRecord) -> dict[str, Any]:
    return {
        "id": call.id,
        "timestamp": call.timestamp,
        "status": call.status,
        "model": call.model,
        "summary": call.summary,
    }
