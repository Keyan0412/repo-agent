from __future__ import annotations

from pathlib import Path
from typing import Any

from repo_agent.agents.json_repair_agent import JsonRepairAgent
from repo_agent.investigation import (
    InvestigationReport,
    InvestigationTask,
    Observation,
    SubInvestigationReport,
    SubInvestigationTask,
)
from repo_agent.llm.client import LLMClient
from repo_agent.llm.schemas import InvestigatorSubreportPayload
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
        subtask = SubInvestigationTask(
            id=f"{task.id}-S1",
            parent_task_id=task.id,
            question=task.task,
            purpose=f"回答用户问题：{task.user_query}",
            expected_evidence=["当前源码、符号搜索结果、相关文件内容"],
            known_information=None,
            max_tool_calls=task.max_tool_calls,
            max_files=6,
        )
        subreport = self.investigate_subtask(subtask)
        return InvestigationReport(
            id=f"R-{task.id}",
            task_id=task.id,
            summary=subreport.answer,
            observations=subreport.observations,
            files_checked=subreport.files_checked,
            remaining_questions=subreport.unresolved,
            subreports=[subreport],
        )

    def investigate_subtask(
        self,
        subtask: SubInvestigationTask,
    ) -> SubInvestigationReport:
        if subtask.max_tool_calls <= 0:
            raise RuntimeError("SubInvestigationTask.max_tool_calls must be positive")

        prompt = self.investigator_prompt_path.read_text(encoding="utf-8").strip()
        response, executed_tools = self.llm_client.run_tool_calling_loop(
            system_prompt=prompt,
            user_content=(
                f"问题: {subtask.question}\n"
                f"目的: {subtask.purpose}\n"
                f"期望证据: {', '.join(subtask.expected_evidence) or '无'}\n"
                f"已知信息:\n{subtask.known_information or '无'}\n\n"
                "按需使用工具。必须保持在当前子任务范围内。\n"
                f"{self._subreport_output_contract()}"
            ),
            tool_registry=self.tool_registry,
            max_tool_calls=subtask.max_tool_calls,
            max_files=subtask.max_files,
        )
        report_payload = self._parse_subreport_payload(response.content)
        files_checked, symbols_checked, file_contents = self._collect_execution_artifacts(
            executed_tools=executed_tools,
            max_files=subtask.max_files,
        )
        observations, rejected_evidence_spans = self._observations_from_evidence_spans(
            report_payload["evidence_spans"],
            file_contents=file_contents,
            next_id_start=1,
        )
        unresolved = list(report_payload["unresolved"])
        if rejected_evidence_spans:
            unresolved.extend(rejected_evidence_spans)
            if report_payload["confidence"] == "high":
                report_payload["confidence"] = "medium"
        report = SubInvestigationReport(
            id=f"{subtask.id}-report",
            parent_task_id=subtask.parent_task_id,
            subtask_id=subtask.id,
            question=subtask.question,
            answer=report_payload["answer"],
            confidence=report_payload["confidence"],
            observations=observations,
            files_checked=files_checked,
            symbols_checked=symbols_checked,
            unresolved=unresolved,
            additional_tool_calls_needed=report_payload["additional_tool_calls_needed"],
            additional_file_reads_needed=report_payload["additional_file_reads_needed"],
        )
        self._emit_tool_events(subtask=subtask, executed_tools=executed_tools)
        self._emit_report_event(report)
        return report

    @staticmethod
    def _subreport_output_contract() -> str:
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
- evidence span 只能引用已用 `read_file` 检查过的文件。
- 不要在 `evidence_spans` 中引用 read_repo_tree、find_text、trace_symbol、目录列表或未读文件。
- 如果预算耗尽，请适当降低 confidence，在 `unresolved` 中列出缺失检查，并估计额外预算字段。
""".strip()

    def _parse_subreport_payload(self, response_content: str) -> dict[str, Any]:
        payload = self.llm_client.extract_json_object(
            response_content,
            repair_agent=self.json_repair_agent,
            target_name="InvestigatorSubreportPayload",
            json_schema=InvestigatorSubreportPayload.model_json_schema(),
        )
        try:
            validated = InvestigatorSubreportPayload.model_validate(payload)
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
    ) -> tuple[list[str], list[str], dict[str, str]]:
        files_checked: list[str] = []
        symbols_checked: list[str] = []
        file_contents: dict[str, str] = {}
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
            elif tool_name == "trace_symbol":
                symbol_name = str(metadata.get("symbol_name") or arguments.get("symbol_name") or "")
                if symbol_name and symbol_name not in symbols_checked:
                    symbols_checked.append(symbol_name)
        return files_checked, symbols_checked, file_contents

    def _emit_tool_events(
        self,
        *,
        subtask: SubInvestigationTask,
        executed_tools: list[dict[str, Any]],
    ) -> None:
        for executed in executed_tools:
            result = executed["result"]
            self.event_sink.emit(
                "investigator.tool_call",
                {
                    "subtask_id": subtask.id,
                    "name": executed["name"],
                    "arguments": executed["arguments"],
                    "success": result.success,
                    "result": result.content,
                },
            )

    def _emit_report_event(self, report: SubInvestigationReport) -> None:
        self.event_sink.emit(
            "investigator.report",
            {
                "id": report.id,
                "subtask_id": report.subtask_id,
                "answer": report.answer,
                "confidence": report.confidence,
                "files_checked": report.files_checked,
                "symbols_checked": report.symbols_checked,
                "unresolved": report.unresolved,
                "observations_count": len(report.observations),
                "additional_tool_calls_needed": report.additional_tool_calls_needed,
                "additional_file_reads_needed": report.additional_file_reads_needed,
            },
        )

    def _observations_from_evidence_spans(
        self,
        spans: list[dict[str, Any]],
        *,
        file_contents: dict[str, str],
        next_id_start: int,
    ) -> tuple[list[Observation], list[str]]:
        observations: list[Observation] = []
        rejected: list[str] = []
        for index, span in enumerate(spans, start=next_id_start):
            file_path = span["file_path"]
            if file_path not in file_contents:
                rejected.append(
                    f"Dropped evidence span for unread file: {file_path}"
                )
                continue
            start_line = span["start_line"]
            end_line = span["end_line"]
            if start_line <= 0 or end_line < start_line:
                rejected.append(f"Dropped invalid evidence span range: {span}")
                continue
            try:
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
