from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from repo_agent_viewer.store import RunStore

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))


def create_app(
    *,
    runs_dir: str | Path | None = None,
    repo_root: str | Path = ".",
    cache_dir: str = ".cache/repo-agent",
) -> FastAPI:
    resolved_runs_dir = Path(runs_dir) if runs_dir is not None else Path(repo_root).resolve() / cache_dir / "runs"
    app = FastAPI(title="repo-agent Run Viewer")
    store = RunStore(resolved_runs_dir)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": store.list_runs(),
                "runs_dir": str(resolved_runs_dir),
            },
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str) -> HTMLResponse:
        view = store.get_main_view(run_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return TEMPLATES.TemplateResponse(request, "run.html", view)

    @app.get("/runs/{run_id}/investigations/{investigation_id}", response_class=HTMLResponse)
    def investigation_detail(request: Request, run_id: str, investigation_id: str) -> HTMLResponse:
        view = store.get_investigation_view(run_id, investigation_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Investigation not found")
        return TEMPLATES.TemplateResponse(request, "investigation.html", view)

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [_run_payload(run) for run in store.list_runs()]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        view = store.get_main_view(run_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run": _run_payload(view["run"]),
            "summary": view["summary"],
            "main_calls": view["main_calls"],
            "main_messages": view["main_messages"],
            "reports": view["reports"],
        }

    @app.get("/api/runs/{run_id}/investigations/{investigation_id}")
    def get_investigation(run_id: str, investigation_id: str) -> dict[str, Any]:
        view = store.get_investigation_view(run_id, investigation_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Investigation not found")
        return {
            "investigation": view["investigation"],
            "messages": view["messages"],
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View repo-agent run traces in a browser.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing .cache/repo-agent/runs.")
    parser.add_argument("--cache-dir", default=".cache/repo-agent", help="Cache directory relative to repo root.")
    parser.add_argument("--runs-dir", default=None, help="Explicit path to a runs directory.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(runs_dir=args.runs_dir, repo_root=args.repo_root, cache_dir=args.cache_dir)
    uvicorn.run(app, host=args.host, port=args.port)


def _run_payload(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "status": run.status,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "user_query": run.user_query,
        "total_tokens": run.total_tokens,
        "investigation_count": run.investigation_count,
        "issue_count": run.issue_count,
    }
