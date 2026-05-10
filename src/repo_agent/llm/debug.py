from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from repo_agent.cache.paths import CachePaths
from repo_agent.llm.schemas import LLMResponse
from repo_agent.runtime.text import strip_surrogates_from_json


class LLMCallDebugRecorder(Protocol):
    def record_success(self, *, model: str, payload: dict[str, Any], response: LLMResponse) -> None: ...

    def record_error(self, *, model: str, payload: dict[str, Any], error: Exception) -> None: ...


class RunLLMCallDebugRecorder:
    def __init__(self, *, repo_path: str | Path, cache_dir: str = ".cache/repo-agent") -> None:
        self.paths = CachePaths(Path(repo_path), cache_dir)
        self.run_id = _new_run_id()
        self.run_dir = self.paths.runs_dir / self.run_id
        self.raw_path = self.run_dir / "raw_llm_calls.jsonl"
        self._call_index = 0
        self.started_at = _utc_timestamp()

    @classmethod
    def at_repo_cache(
        cls,
        repo_path: str | Path,
        *,
        cache_dir: str = ".cache/repo-agent",
    ) -> "RunLLMCallDebugRecorder":
        return cls(repo_path=repo_path, cache_dir=cache_dir)

    def record_success(self, *, model: str, payload: dict[str, Any], response: LLMResponse) -> None:
        self._append_record(
            {
                "timestamp": _utc_timestamp(),
                "run_id": self.run_id,
                "call_index": self._next_call_index(),
                "status": "success",
                "model": model,
                "agent": _infer_agent(payload),
                "request": payload,
                "response": {
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                    "usage": response.usage,
                    "raw": response.raw,
                },
            }
        )

    def record_error(self, *, model: str, payload: dict[str, Any], error: Exception) -> None:
        self._append_record(
            {
                "timestamp": _utc_timestamp(),
                "run_id": self.run_id,
                "call_index": self._next_call_index(),
                "status": "error",
                "model": model,
                "agent": _infer_agent(payload),
                "request": payload,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                },
            }
        )

    def finalize_run(
        self,
        *,
        user_query: str,
        final_answer: str | None,
        status: str,
        error: str | None = None,
    ) -> None:
        records = self._read_records()
        summary = strip_surrogates_from_json(build_run_summary(
            run_id=self.run_id,
            user_query=user_query,
            started_at=self.started_at,
            ended_at=_utc_timestamp(),
            final_answer=final_answer,
            status=status,
            error=error,
            records=records,
        ))
        self._write_json(self.run_dir / "run_summary.json", summary)
        self._write_json(self.run_dir / "main_agent.json", _main_agent_trace(records))

        investigations_dir = self.run_dir / "investigations"
        investigations_dir.mkdir(parents=True, exist_ok=True)
        for index, investigation in enumerate(summary["investigations"], start=1):
            self._write_json(investigations_dir / f"T{index:04d}.json", investigation)

        summaries_dir = self.run_dir / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate([r for r in records if _record_agent(r) == "summarizer"], start=1):
            self._write_json(summaries_dir / f"S{index:04d}.json", _call_summary(record))

    def _append_record(self, record: dict[str, Any]) -> None:
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        record = strip_surrogates_from_json(record)
        with self.raw_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            fh.write("\n")

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.raw_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _next_call_index(self) -> int:
        self._call_index += 1
        return self._call_index

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = strip_surrogates_from_json(payload)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )


def build_run_summary(
    *,
    run_id: str,
    user_query: str,
    started_at: str,
    ended_at: str,
    final_answer: str | None,
    status: str,
    error: str | None,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    calls = [_call_summary(record) for record in records]
    investigations = _investigation_traces(records)
    issues = _detect_issues(calls=calls, investigations=investigations)
    return {
        "run_id": run_id,
        "status": status,
        "user_query": user_query,
        "started_at": started_at,
        "ended_at": ended_at,
        "final_answer": final_answer,
        "error": error,
        "total_usage": _sum_usage(calls),
        "agents": _agent_usage(calls),
        "call_count": len(calls),
        "investigations": investigations,
        "issues": issues,
    }


def _call_summary(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    return {
        "call_index": record.get("call_index"),
        "timestamp": record.get("timestamp"),
        "status": record.get("status"),
        "agent": _record_agent(record),
        "model": record.get("model"),
        "usage": response.get("usage") or {},
        "tool_calls": [
            {
                "name": ((tool_call.get("function") or {}).get("name")),
                "arguments": _safe_json_loads((tool_call.get("function") or {}).get("arguments") or "{}"),
            }
            for tool_call in _tool_calls_from_response(response)
        ],
        "content_preview": str(response.get("content") or "")[:500],
        "tool_result_previews": _tool_result_previews(request),
        "error": record.get("error"),
    }


def _main_agent_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [_call_summary(record) for record in records if _record_agent(record) == "main"]
    return {"calls": calls, "usage": _sum_usage(calls)}


def _investigation_traces(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    investigations: list[dict[str, Any]] = []
    current: list[dict[str, Any]] | None = None
    current_task = ""
    for record in records:
        agent = _record_agent(record)
        if agent == "main":
            response = record.get("response") if isinstance(record.get("response"), dict) else {}
            for tool_call in _tool_calls_from_response(response):
                function = tool_call.get("function") or {}
                if function.get("name") == "request_investigation":
                    if current:
                        investigations.append(_build_investigation_trace(len(investigations) + 1, current_task, current))
                    current = []
                    current_task = str(_safe_json_loads(function.get("arguments") or "{}").get("task") or "")
            continue
        if agent in {"investigator", "summarizer", "repair"} and current is not None:
            current.append(record)
    if current:
        investigations.append(_build_investigation_trace(len(investigations) + 1, current_task, current))
    return investigations


def _build_investigation_trace(index: int, task: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [_call_summary(record) for record in records]
    tool_counts: dict[str, int] = {}
    read_files_calls: list[dict[str, Any]] = []
    summarize_files_calls: list[dict[str, Any]] = []
    missing_paths: list[str] = []
    for call in calls:
        for tool_call in call["tool_calls"]:
            name = str(tool_call.get("name") or "")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            if name == "read_files":
                read_files_calls.append(tool_call)
            if name == "summarize_files":
                summarize_files_calls.append(tool_call)
        previews = [str(call.get("content_preview") or "")]
        previews.extend(str(item) for item in call.get("tool_result_previews") or [])
        for preview in previews:
            if "file does not exist:" in preview:
                missing_paths.append(preview)
    return {
        "id": f"T{index:04d}",
        "task": task,
        "usage": _sum_usage(calls),
        "calls": calls,
        "tool_counts": tool_counts,
        "read_files_call_count": len(read_files_calls),
        "summarize_files_call_count": len(summarize_files_calls),
        "read_files_paths": _read_files_paths(read_files_calls),
        "missing_path_messages": missing_paths,
        "issues": _detect_investigation_issues(calls, read_files_calls, missing_paths),
    }


def _detect_issues(*, calls: list[dict[str, Any]], investigations: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    total = _sum_usage(calls).get("total_tokens", 0)
    if total > 150_000:
        issues.append(f"total token usage is high: {total}")
    for call in calls:
        prompt_tokens = call.get("usage", {}).get("prompt_tokens", 0)
        if isinstance(prompt_tokens, int) and prompt_tokens > 20_000:
            issues.append(f"call {call.get('call_index')} prompt is high: {prompt_tokens}")
    for investigation in investigations:
        for issue in investigation.get("issues", []):
            issues.append(f"{investigation['id']}: {issue}")
    return issues


def _detect_investigation_issues(
    calls: list[dict[str, Any]],
    read_files_calls: list[dict[str, Any]],
    missing_paths: list[str],
) -> list[str]:
    issues: list[str] = []
    if len(read_files_calls) > 1:
        issues.append(f"read_files called {len(read_files_calls)} times")
    if missing_paths:
        issues.append("missing path encountered")
    for call in calls:
        preview = call.get("content_preview") or ""
        if '"confidence": "high"' in preview and '"unresolved": [' in preview and '"unresolved": []' not in preview:
            issues.append("high confidence report contains unresolved items")
            break
    return issues


def _read_files_paths(read_files_calls: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for call in read_files_calls:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        files = arguments.get("files")
        if not isinstance(files, list):
            continue
        for item in files:
            if isinstance(item, dict):
                path = str(item.get("path") or "")
                if path and path not in paths:
                    paths.append(path)
    return paths


def _sum_usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for call in calls:
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return totals


def _agent_usage(calls: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    agents: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        agents.setdefault(str(call.get("agent") or "unknown"), []).append(call)
    return {
        agent: {
            "call_count": len(agent_calls),
            **_sum_usage(agent_calls),
        }
        for agent, agent_calls in agents.items()
    }


def _infer_agent(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return "unknown"
    system_prompt = str((messages[0] or {}).get("content") or "")
    if system_prompt.lstrip().startswith("你是 MainAgent") or "你是 MainAgent" in system_prompt[:80]:
        return "main"
    if "FileSummaryAgent" in system_prompt:
        return "summarizer"
    if "JsonRepairAgent" in system_prompt:
        return "repair"
    if system_prompt.lstrip().startswith("你是 InvestigatorAgent") or "你是 InvestigatorAgent" in system_prompt[:120]:
        return "investigator"
    if "InvestigatorAgent" in system_prompt:
        return "investigator"
    if "MainAgent" in system_prompt:
        return "main"
    return "unknown"


def _record_agent(record: dict[str, Any]) -> str:
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    inferred = _infer_agent(request)
    if inferred != "unknown":
        return inferred
    return str(record.get("agent") or "unknown")


def _tool_result_previews(request: dict[str, Any]) -> list[dict[str, str]]:
    previews: list[dict[str, str]] = []
    messages = request.get("messages")
    if not isinstance(messages, list):
        return previews
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        previews.append(
            {
                "name": str(message.get("name") or ""),
                "content": str(message.get("content") or "")[:500],
            }
        )
    return previews


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_object_text(text)
        if extracted is None:
            return {}
        try:
            value = json.loads(_remove_trailing_commas(extracted))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _tool_calls_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]

    raw = response.get("raw") if isinstance(response.get("raw"), dict) else {}
    choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
    extracted: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            extracted.extend(tool_call for tool_call in raw_tool_calls if isinstance(tool_call, dict))
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            extracted.append(
                {
                    "id": "",
                    "type": "function",
                    "function": function_call,
                }
            )
    return extracted


def _extract_json_object_text(text: str) -> str | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return repr(value)
