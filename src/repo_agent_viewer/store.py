from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from repo_agent.llm.debug import build_run_summary


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: str
    started_at: str
    ended_at: str
    user_query: str
    total_tokens: int
    investigation_count: int
    issue_count: int
    summary: dict[str, Any]


class RunStore:
    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)

    def list_runs(self) -> list[RunRecord]:
        records = [
            record
            for path in self._summary_paths()
            if (record := self._to_run_record(path)) is not None
        ]
        return sorted(
            records,
            key=lambda record: (self._sort_key(record.ended_at), record.run_id),
            reverse=True,
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        summary_path = self.runs_dir / run_id / "run_summary.json"
        raw_path = self.runs_dir / run_id / "raw_llm_calls.jsonl"
        if not summary_path.exists() and not raw_path.exists():
            return None
        return self._to_run_record(summary_path)

    def get_main_view(self, run_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        records = self._raw_records(run_id)
        summary = self._effective_summary(run, records)
        main_calls = [
            self._call_view(record)
            for record in records
            if _infer_agent(record) == "main"
        ]
        main_records = [
            record
            for record in records
            if _infer_agent(record) == "main"
        ]
        return {
            "run": run,
            "main_calls": main_calls,
            "main_messages": self._conversation_messages(main_records),
            "reports": self._report_links(records, summary.get("investigations", [])),
            "summary": summary,
        }

    def get_investigation_view(self, run_id: str, investigation_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        investigation_path = self.runs_dir / run_id / "investigations" / f"{investigation_id}.json"
        if investigation_path.exists():
            investigation = json.loads(investigation_path.read_text(encoding="utf-8"))
        else:
            summary = self._effective_summary(run, self._raw_records(run_id))
            investigation = next(
                (
                    item
                    for item in summary.get("investigations", [])
                    if isinstance(item, dict) and item.get("id") == investigation_id
                ),
                None,
            )
            if investigation is None:
                return None
        records = self._investigation_records(run_id, investigation_id)
        return {
            "run": run,
            "investigation": investigation,
            "messages": self._conversation_messages(
                [record for record in records if _infer_agent(record) == "investigator"]
            ),
        }

    def _summary_paths(self) -> list[Path]:
        if not self.runs_dir.exists():
            return []
        return [
            path / "run_summary.json"
            for path in self.runs_dir.iterdir()
            if path.is_dir()
            and ((path / "run_summary.json").exists() or (path / "raw_llm_calls.jsonl").exists())
        ]

    def _to_run_record(self, summary_path: Path) -> RunRecord | None:
        summary = self._load_summary(summary_path)
        if summary is None:
            records = self._raw_records(summary_path.parent.name)
            if not records:
                return None
            summary = build_run_summary(
                run_id=summary_path.parent.name,
                user_query=_latest_user_query(records),
                started_at=str(records[0].get("timestamp") or ""),
                ended_at=str(records[-1].get("timestamp") or ""),
                final_answer=None,
                status="unknown",
                error="run_summary.json is missing or invalid; rebuilt from raw_llm_calls.jsonl",
                records=records,
            )
        usage = summary.get("total_usage") if isinstance(summary.get("total_usage"), dict) else {}
        investigations = summary.get("investigations") if isinstance(summary.get("investigations"), list) else []
        issues = summary.get("issues") if isinstance(summary.get("issues"), list) else []
        return RunRecord(
            run_id=str(summary.get("run_id") or summary_path.parent.name),
            status=str(summary.get("status") or "unknown"),
            started_at=str(summary.get("started_at") or ""),
            ended_at=str(summary.get("ended_at") or ""),
            user_query=str(summary.get("user_query") or ""),
            total_tokens=int(usage.get("total_tokens") or 0),
            investigation_count=len(investigations),
            issue_count=len(issues),
            summary=summary,
        )

    @staticmethod
    def _load_summary(summary_path: Path) -> dict[str, Any] | None:
        if not summary_path.exists():
            return None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return summary if isinstance(summary, dict) else None

    def _raw_records(self, run_id: str) -> list[dict[str, Any]]:
        raw_path = self.runs_dir / run_id / "raw_llm_calls.jsonl"
        if not raw_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in raw_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(self._normalize_record(value))
        return records

    def _effective_summary(self, run: RunRecord, records: list[dict[str, Any]]) -> dict[str, Any]:
        investigations = run.summary.get("investigations")
        main_exists = any(_infer_agent(record) == "main" for record in records)
        if isinstance(investigations, list) and investigations and main_exists:
            return run.summary
        return build_run_summary(
            run_id=run.run_id,
            user_query=run.user_query,
            started_at=run.started_at,
            ended_at=run.ended_at,
            final_answer=run.summary.get("final_answer"),
            status=run.status,
            error=run.summary.get("error"),
            records=records,
        )

    @staticmethod
    def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized["agent"] = _infer_agent(record)
        return normalized

    def _call_view(self, record: dict[str, Any]) -> dict[str, Any]:
        request = record.get("request") if isinstance(record.get("request"), dict) else {}
        response = record.get("response") if isinstance(record.get("response"), dict) else {}
        return {
            "call_index": record.get("call_index"),
            "timestamp": record.get("timestamp"),
            "status": record.get("status"),
            "model": record.get("model"),
            "usage": response.get("usage") or {},
            "messages": request.get("messages") if isinstance(request.get("messages"), list) else [],
            "response": {
                "content": response.get("content") or "",
                "tool_calls": _tool_calls_from_response(response),
            },
        }

    def _conversation_messages(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        latest = records[-1]
        request = latest.get("request") if isinstance(latest.get("request"), dict) else {}
        response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        messages = [
            self._message_view(message)
            for message in request.get("messages", [])
            if isinstance(message, dict)
        ]
        response_message: dict[str, Any] = {
            "role": "assistant",
            "name": "",
            "content": response.get("content") or "",
            "tool_calls": [
                self._tool_call_message_view(tool_call)
                for tool_call in _tool_calls_from_response(response)
                if isinstance(tool_call, dict)
            ],
        }
        if response_message["content"] or response_message["tool_calls"]:
            messages.append(response_message)
        return self._merge_tool_results(messages)

    def _message_view(self, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": str(message.get("role") or ""),
            "name": str(message.get("name") or ""),
            "content": message.get("content") or "",
            "tool_call_id": str(message.get("tool_call_id") or ""),
            "tool_calls": [
                self._tool_call_message_view(tool_call)
                for tool_call in message.get("tool_calls") or []
                if isinstance(tool_call, dict)
            ],
        }

    def _merge_tool_results(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        pending_by_id: dict[str, dict[str, Any]] = {}
        pending_by_name: dict[str, list[dict[str, Any]]] = {}

        for message in messages:
            if message.get("role") == "tool":
                matched = self._match_tool_result(
                    message=message,
                    pending_by_id=pending_by_id,
                    pending_by_name=pending_by_name,
                )
                if matched is not None:
                    matched["result_content"] = str(message.get("content") or "")
                    matched["has_result"] = True
                    continue
                merged.append(message)
                continue

            tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
            for tool_call in tool_calls:
                call_id = str(tool_call.get("id") or "")
                name = str(tool_call.get("name") or "")
                if call_id:
                    pending_by_id[call_id] = tool_call
                if name:
                    pending_by_name.setdefault(name, []).append(tool_call)
            merged.append(message)

        return merged

    @staticmethod
    def _match_tool_result(
        *,
        message: dict[str, Any],
        pending_by_id: dict[str, dict[str, Any]],
        pending_by_name: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        tool_call_id = str(message.get("tool_call_id") or "")
        if tool_call_id and tool_call_id in pending_by_id:
            return pending_by_id.pop(tool_call_id)

        name = str(message.get("name") or "")
        calls = pending_by_name.get(name) if name else None
        if calls:
            return calls.pop(0)
        return None

    @staticmethod
    def _tool_call_message_view(tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "")
        raw_arguments = function.get("arguments", tool_call.get("arguments", {}))
        arguments = _parse_tool_arguments(raw_arguments)
        return {
            "id": str(tool_call.get("id") or ""),
            "name": name,
            "arguments": arguments,
            "argument_items": _argument_items(arguments),
            "arguments_text": _argument_text(arguments),
            "result_content": "",
            "has_result": False,
        }

    def _investigation_records(self, run_id: str, investigation_id: str) -> list[dict[str, Any]]:
        records = self._raw_records(run_id)
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] | None = None
        for record in records:
            if _infer_agent(record) == "main":
                if _starts_investigation(record):
                    if current is not None:
                        groups.append(current)
                    current = []
                continue
            if current is not None and _infer_agent(record) in {"investigator", "summarizer", "repair"}:
                current.append(record)
        if current is not None:
            groups.append(current)

        try:
            index = int(investigation_id.removeprefix("T")) - 1
        except ValueError:
            return []
        if index < 0 or index >= len(groups):
            return []
        return groups[index]

    def _report_links(
        self,
        records: list[dict[str, Any]],
        investigations: list[Any],
    ) -> list[dict[str, str]]:
        reports: list[dict[str, str]] = []
        seen_contents: set[str] = set()
        investigation_ids = [
            str(item.get("id") or "")
            for item in investigations
            if isinstance(item, dict)
        ]
        for record in records:
            if _infer_agent(record) != "main":
                continue
            request = record.get("request") if isinstance(record.get("request"), dict) else {}
            messages = request.get("messages") if isinstance(request.get("messages"), list) else []
            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "tool":
                    continue
                if message.get("name") != "request_investigation":
                    continue
                content = str(message.get("content") or "")
                if not content or content in seen_contents:
                    continue
                seen_contents.add(content)
                report_id = _extract_report_id(content)
                index = len(reports)
                investigation_id = investigation_ids[index] if index < len(investigation_ids) else f"T{index + 1:04d}"
                reports.append(
                    {
                        "report_id": report_id or f"Report {index}",
                        "investigation_id": investigation_id,
                        "preview": _truncate(content, 240),
                    }
                )
        return reports

    @staticmethod
    def _sort_key(timestamp: str) -> datetime:
        if not timestamp:
            return datetime.min
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min


def _extract_report_id(content: str) -> str:
    match = re.search(r"调查结果 \[(\d+)\]\s+([^:]+):", content)
    if not match:
        return ""
    return f"[{match.group(1)}] {match.group(2)}"


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


def _infer_agent(record: dict[str, Any]) -> str:
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return str(record.get("agent") or "unknown")
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
    return str(record.get("agent") or "unknown")


def _starts_investigation(record: dict[str, Any]) -> bool:
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    for tool_call in _tool_calls_from_response(response):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "")
        if name == "request_investigation":
            return True
    return False


def _truncate(text: str, limit: int = 80) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _parse_tool_arguments(raw_arguments: Any) -> Any:
    if isinstance(raw_arguments, str):
        text = raw_arguments.strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = _strip_markdown_fence(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            extracted = _extract_json_object_text(text)
            if extracted is not None:
                try:
                    return json.loads(_remove_trailing_commas(extracted))
                except json.JSONDecodeError:
                    pass
            return raw_arguments
    return raw_arguments if raw_arguments is not None else {}


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object_text(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _argument_items(arguments: Any) -> list[dict[str, str]]:
    if isinstance(arguments, dict):
        return [
            {"key": str(key), "value": _argument_text(value)}
            for key, value in arguments.items()
        ]
    return [{"key": "arguments", "value": _argument_text(arguments)}]


def _argument_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _latest_user_query(records: list[dict[str, Any]]) -> str:
    for record in reversed(records):
        if _infer_agent(record) != "main":
            continue
        request = record.get("request") if isinstance(record.get("request"), dict) else {}
        messages = request.get("messages") if isinstance(request.get("messages"), list) else []
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            match = re.search(r"当前用户问题:\n(.+?)(?:\n\n|$)", content, flags=re.DOTALL)
            if match:
                return match.group(1).strip()
            if content.strip():
                return content.strip()
    return ""
