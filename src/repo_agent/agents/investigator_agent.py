from __future__ import annotations

from pathlib import Path
from typing import Any

from repo_agent.agents.json_repair_agent import JsonRepairAgent
from repo_agent.investigation import (
    InvestigationReport,
    InvestigationTask,
    Observation,
)
from repo_agent.llm.client import LLMClient
from repo_agent.llm.schemas import InvestigatorReportPayload
from repo_agent.runtime.events import EventSink, NullEventSink
from repo_agent.tools.registry import ToolRegistry


class InvestigatorAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        repo_path: Path,
        tool_registry: ToolRegistry,
        *,
        investigator_prompt_path: Path | None = None,
        json_repair_agent: JsonRepairAgent | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.repo_path = Path(repo_path).resolve()
        self.tool_registry = tool_registry
        self.json_repair_agent = json_repair_agent or JsonRepairAgent(llm_client)
        self.event_sink = event_sink or NullEventSink()
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        self.investigator_prompt_path = investigator_prompt_path or prompts_dir / "investigator_agent.md"

    def investigate(self, task: InvestigationTask) -> InvestigationReport:
        if task.max_tool_calls <= 0:
            raise RuntimeError("InvestigationTask.max_tool_calls must be positive")
        if task.max_file_reads <= 0:
            raise RuntimeError("InvestigationTask.max_file_reads must be positive")

        prompt = self.investigator_prompt_path.read_text(encoding="utf-8").strip()
        response, executed_tools = self.llm_client.run_tool_calling_loop(
            system_prompt=prompt,
            user_content=(
                f"问题: {task.task}\n"
                f"目的: 回答用户问题：{task.user_query}\n"
                "期望证据: 当前源码、符号搜索结果、相关文件内容\n"
                "已知信息:\n无\n\n"
                "按需使用工具。必须保持在当前调查任务范围内。\n"
                f"{self._report_output_contract()}"
            ),
            tool_registry=self.tool_registry,
            max_tool_calls=task.max_tool_calls,
            max_files=task.max_file_reads,
            on_tool_result=lambda executed: self._emit_tool_event(
                task_id=task.id,
                executed=executed,
            ),
        )
        report_payload = self._parse_report_payload(response.content)
        files_checked, symbols_checked, file_contents, summary_regions = self._collect_execution_artifacts(
            executed_tools=executed_tools,
            max_files=task.max_file_reads,
        )
        observations, rejected_evidence_spans = self._observations_from_evidence_spans(
            report_payload["evidence_spans"],
            file_contents=file_contents,
            summary_regions=summary_regions,
            next_id_start=1,
        )
        unresolved = list(report_payload["unresolved"])
        if rejected_evidence_spans:
            unresolved.extend(rejected_evidence_spans)
            if report_payload["confidence"] == "high":
                report_payload["confidence"] = "medium"
        report = InvestigationReport(
            id=f"R-{task.id}",
            task_id=task.id,
            summary=report_payload["answer"],
            observations=observations,
            files_checked=files_checked,
            remaining_questions=unresolved,
        )
        self._emit_report_event(
            report,
            confidence=report_payload["confidence"],
            symbols_checked=symbols_checked,
            additional_tool_calls_needed=report_payload["additional_tool_calls_needed"],
            additional_file_reads_needed=report_payload["additional_file_reads_needed"],
        )
        return report

    @staticmethod
    def _report_output_contract() -> str:
        return """
最终输出契约:
必须只返回一个严格 JSON object，不要包含散文说明或 Markdown fence。

必需顶层 key:
{
  "answer": "基于已检查证据的简短综合",
  "confidence": "high | medium | low",
  "unresolved": [],
  "evidence_spans": [],
  "additional_tool_calls_needed": 0,
  "additional_file_reads_needed": 0
}

硬性要求:
- `confidence` 必须严格是小写 `high`、`medium` 或 `low`。
- `unresolved` 必须是字符串列表。
- `additional_tool_calls_needed` 和 `additional_file_reads_needed` 必须是整数。
- 每个 evidence span 必须包含 `file_path`、`start_line`、`end_line` 和 `summary`。
- evidence span 行号必须是正整数，绝不能使用 0。
- evidence span 只能引用已用 `read_file`、`summarize_file` 或 `summarize_files` 检查过的文件。
- 不要在 `evidence_spans` 中引用 read_repo_tree、find_text、trace_symbol、目录列表或未检查文件。
- 如果预算耗尽，请适当降低 confidence，在 `unresolved` 中列出缺失检查，并估计额外预算字段。
""".strip()

    def _parse_report_payload(self, response_content: str) -> dict[str, Any]:
        payload = self.llm_client.extract_json_object(
            response_content,
            repair_agent=self.json_repair_agent,
            target_name="InvestigatorReportPayload",
            json_schema=InvestigatorReportPayload.model_json_schema(),
        )
        try:
            validated = InvestigatorReportPayload.model_validate(payload)
        except Exception as exc:
            raise RuntimeError(
                "InvestigatorAgent received a payload with invalid field types: "
                f"{payload}"
            ) from exc

        if not validated.answer.strip():
            raise RuntimeError("InvestigatorAgent received an empty answer from the LLM")
        return validated.model_dump()

    def _collect_execution_artifacts(
        self,
        *,
        executed_tools: list[dict[str, Any]],
        max_files: int,
    ) -> tuple[
        list[str],
        list[str],
        dict[str, str],
        dict[str, list[dict[str, Any]]],
    ]:
        files_checked: list[str] = []
        symbols_checked: list[str] = []
        file_contents: dict[str, str] = {}
        summary_regions: dict[str, list[dict[str, Any]]] = {}
        for executed in executed_tools:
            tool_name = executed["name"]
            arguments = executed["arguments"]
            result = executed["result"]
            metadata = result.metadata
            if tool_name == "read_file":
                path = str(metadata.get("path") or arguments.get("path") or "")
                if path and path not in files_checked and len(files_checked) < max_files:
                    files_checked.append(path)
                if path:
                    file_contents[path] = result.content
            elif tool_name == "summarize_file":
                path = str(metadata.get("path") or arguments.get("path") or "")
                if path and path not in files_checked and len(files_checked) < max_files:
                    files_checked.append(path)
                if path:
                    file_contents[path] = result.content
                    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
                    regions = summary.get("evidence_regions") or []
                    if regions:
                        summary_regions[path] = regions
            elif tool_name == "summarize_files":
                paths = metadata.get("paths") if isinstance(metadata.get("paths"), list) else []
                for raw_path in paths:
                    path = str(raw_path)
                    if path and path not in files_checked and len(files_checked) < max_files:
                        files_checked.append(path)
                    if path:
                        file_contents[path] = result.content
                summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
                for file_summary in summary.get("files") or []:
                    if not isinstance(file_summary, dict):
                        continue
                    path = str(file_summary.get("path") or "")
                    regions = file_summary.get("evidence_regions") or []
                    if path and regions:
                        summary_regions[path] = regions
            elif tool_name == "trace_symbol":
                symbol_name = str(metadata.get("symbol_name") or arguments.get("symbol_name") or "")
                if symbol_name and symbol_name not in symbols_checked:
                    symbols_checked.append(symbol_name)
        return (
            files_checked,
            symbols_checked,
            file_contents,
            summary_regions,
        )

    def _emit_tool_event(
        self,
        *,
        task_id: str,
        executed: dict[str, Any],
    ) -> None:
        result = executed["result"]
        summary = self._summarize_tool_result(
            name=executed["name"],
            arguments=executed["arguments"],
            success=result.success,
            metadata=result.metadata,
        )
        self.event_sink.emit(
            "investigator.tool_call",
            {
                "task_id": task_id,
                "name": executed["name"],
                "arguments": executed["arguments"],
                "success": result.success,
                "summary": summary["summary"],
                "detail": summary["detail"],
                "metadata": summary["metadata"],
            },
        )

    @staticmethod
    def _summarize_tool_result(
        *,
        name: str,
        arguments: dict[str, Any],
        success: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not success:
            return {
                "summary": "tool failed",
                "detail": "-",
                "metadata": {},
            }

        if name == "read_repo_tree":
            path = str(metadata.get("path") or arguments.get("path") or ".")
            max_depth = metadata.get("max_depth", arguments.get("max_depth"))
            detail = f"depth={max_depth}" if max_depth is not None else "-"
            return {
                "summary": f"listed {path}",
                "detail": detail,
                "metadata": {
                    "path": path,
                    "max_depth": max_depth,
                },
            }

        if name == "read_file":
            path = str(metadata.get("path") or arguments.get("path") or "")
            line_count = metadata.get("line_count")
            start_line = metadata.get("start_line")
            end_line = metadata.get("end_line")
            detail_parts = []
            if line_count is not None:
                detail_parts.append(f"{line_count} lines")
            if start_line is not None and end_line is not None:
                detail_parts.append(f"L{start_line}-L{end_line}")
            if metadata.get("truncated"):
                detail_parts.append("truncated")
            return {
                "summary": f"read {path or 'file'}",
                "detail": "  ".join(detail_parts) or "file read",
                "metadata": {
                    "path": path,
                    "line_count": line_count,
                    "start_line": start_line,
                    "end_line": end_line,
                    "truncated": bool(metadata.get("truncated")),
                    "max_chars": metadata.get("max_chars"),
                },
            }

        if name == "summarize_file":
            path = str(metadata.get("path") or arguments.get("path") or "")
            summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
            regions = summary.get("evidence_regions") or []
            return {
                "summary": f"summarized {path or 'file'}",
                "detail": f"{len(regions)} evidence regions",
                "metadata": {
                    "path": path,
                    "line_count": metadata.get("line_count"),
                    "truncated_before_summary": bool(metadata.get("truncated_before_summary")),
                    "region_count": len(regions),
                },
            }

        if name == "summarize_files":
            paths = metadata.get("paths") if isinstance(metadata.get("paths"), list) else []
            summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
            findings = summary.get("cross_file_findings") or []
            return {
                "summary": f"summarized {len(paths)} files",
                "detail": f"{len(findings)} cross-file findings",
                "metadata": {
                    "paths": paths,
                    "file_count": metadata.get("file_count", len(paths)),
                    "truncated_before_summary_paths": metadata.get(
                        "truncated_before_summary_paths",
                        [],
                    ),
                    "cross_file_finding_count": len(findings),
                },
            }

        if name == "find_text":
            match_count = metadata.get("match_count", 0)
            return {
                "summary": f"{match_count} matches",
                "detail": "truncated" if metadata.get("truncated") else "search complete",
                "metadata": {
                    "match_count": match_count,
                    "truncated": bool(metadata.get("truncated")),
                    "literal_fallback": bool(metadata.get("literal_fallback")),
                },
            }

        if name == "trace_symbol":
            match_count = metadata.get("match_count", 0)
            symbol_name = str(metadata.get("symbol_name") or arguments.get("symbol_name") or "")
            return {
                "summary": f"{match_count} occurrences",
                "detail": "truncated" if metadata.get("truncated") else "trace complete",
                "metadata": {
                    "symbol_name": symbol_name,
                    "match_count": match_count,
                    "truncated": bool(metadata.get("truncated")),
                },
            }

        return {
            "summary": "tool executed",
            "detail": "-",
            "metadata": {},
        }

    def _emit_report_event(
        self,
        report: InvestigationReport,
        *,
        confidence: str,
        symbols_checked: list[str],
        additional_tool_calls_needed: int,
        additional_file_reads_needed: int,
    ) -> None:
        self.event_sink.emit(
            "investigator.report",
            {
                "id": report.id,
                "task_id": report.task_id,
                "answer": report.summary,
                "confidence": confidence,
                "files_checked": report.files_checked,
                "symbols_checked": symbols_checked,
                "unresolved": report.remaining_questions,
                "observations_count": len(report.observations),
                "additional_tool_calls_needed": additional_tool_calls_needed,
                "additional_file_reads_needed": additional_file_reads_needed,
            },
        )

    def _observations_from_evidence_spans(
        self,
        spans: list[dict[str, Any]],
        *,
        file_contents: dict[str, str],
        summary_regions: dict[str, list[dict[str, Any]]],
        next_id_start: int,
    ) -> tuple[list[Observation], list[str]]:
        observations: list[Observation] = []
        rejected: list[str] = []
        for index, span in enumerate(spans, start=next_id_start):
            file_path = span["file_path"]
            if file_path not in file_contents:
                rejected.append(
                    f"Dropped evidence span for unchecked file: {file_path}"
                )
                continue
            start_line = span["start_line"]
            end_line = span["end_line"]
            if start_line <= 0 or end_line < start_line:
                rejected.append(f"Dropped invalid evidence span range: {span}")
                continue
            try:
                if file_path in summary_regions:
                    excerpt = self._build_summary_excerpt(
                        summary_regions=summary_regions[file_path],
                        start_line=start_line,
                        end_line=end_line,
                    )
                else:
                    excerpt = self._extract_excerpt(
                        numbered_content=file_contents[file_path],
                        start_line=start_line,
                        end_line=end_line,
                    )
            except RuntimeError as exc:
                rejected.append(f"Dropped unavailable evidence span: {exc}")
                continue
            observations.append(
                Observation(
                    id=index,
                    summary=span["summary"],
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    excerpt=excerpt,
                )
            )
        return self._dedupe_observations(observations), rejected

    @staticmethod
    def _extract_excerpt(
        *,
        numbered_content: str,
        start_line: int,
        end_line: int,
    ) -> str:
        line_map: dict[int, str] = {}
        for raw_line in numbered_content.splitlines():
            if raw_line == "... [truncated]":
                break
            if " | " not in raw_line:
                continue
            line_no_text, code = raw_line.split(" | ", 1)
            if not line_no_text.isdigit():
                continue
            line_map[int(line_no_text)] = code

        excerpt_lines = []
        for line_no in range(start_line, end_line + 1):
            if line_no not in line_map:
                raise RuntimeError(
                    f"InvestigatorAgent evidence span points to unavailable line {line_no}"
                )
            excerpt_lines.append(f"{line_no} | {line_map[line_no]}")
        return "\n".join(excerpt_lines)

    @staticmethod
    def _build_summary_excerpt(
        *,
        summary_regions: list[dict[str, Any]],
        start_line: int,
        end_line: int,
    ) -> str:
        lines = []
        for region in summary_regions:
            r_start = region.get("start_line", 0)
            r_end = region.get("end_line", 0)
            if r_start <= end_line and r_end >= start_line:
                label = region.get("label", "")
                summary = region.get("summary", "")
                lines.append(f"L{r_start}-L{r_end} {label}: {summary}")
        if not lines:
            raise RuntimeError(
                f"No summary evidence region covers lines {start_line}-{end_line}"
            )
        return "\n".join(lines)

    @staticmethod
    def _dedupe_observations(observations: list[Observation]) -> list[Observation]:
        """remove duplicate observations"""
        deduped: list[Observation] = []
        seen: set[tuple[Any, ...]] = set()
        for observation in observations:
            key = (
                observation.summary,
                observation.file_path,
                observation.start_line,
                observation.end_line,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(observation.model_copy(update={"id": len(deduped) + 1}))
        return deduped
