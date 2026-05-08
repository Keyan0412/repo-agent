from __future__ import annotations

from pathlib import Path
from typing import Any

from repo_agent.agents.json_repair_agent import JsonRepairAgent
from repo_agent.investigation import Observation, SubInvestigationReport, SubInvestigationTask
from repo_agent.llm.client import LLMClient
from repo_agent.llm.schemas import InvestigatorSubreportPayload
from repo_agent.tools.registry import ToolRegistry


class InvestigatorAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        repo_path: Path,
        tool_registry: ToolRegistry,
        *,
        investigator_prompt_path: Path | None = None,
        repo_profile_initial_prompt_path: Path | None = None,
        json_repair_agent: JsonRepairAgent | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.repo_path = Path(repo_path).resolve()
        self.tool_registry = tool_registry
        self.json_repair_agent = json_repair_agent or JsonRepairAgent(llm_client)
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        self.investigator_prompt_path = investigator_prompt_path or prompts_dir / "investigator_agent.md"
        self.repo_profile_initial_prompt_path = (
            repo_profile_initial_prompt_path or prompts_dir / "repo_profile_initial.md"
        )

    def summarize_repo(
        self,
        user_query: str,
        task: str,
        *,
        max_ask_file_calls: int | None = 40,
    ) -> str:
        prompt = self.repo_profile_initial_prompt_path.read_text(encoding="utf-8").strip()
        response, _ = self.llm_client.run_tool_calling_loop(
            system_prompt=prompt,
            user_content=(
                f"User Query:\n{user_query}\n\n"
                f"Investigation Task:\n{task}\n\n"
                "You may use read_repo_tree, find_text, trace_symbol, ask_file, and read_file.\n"
                "After enough evidence is collected, fill the Repo Profile framework from the system prompt exactly."
            ),
            tool_registry=self.tool_registry,
            max_tool_calls=16,
            max_files=10,
            max_ask_file_calls=max_ask_file_calls,
        )
        profile = response.content.strip()
        if not profile:
            raise RuntimeError("InvestigatorAgent received an empty repo profile from the LLM")
        return profile

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
                f"Question: {subtask.question}\n"
                f"Purpose: {subtask.purpose}\n"
                f"Expected Evidence: {', '.join(subtask.expected_evidence) or 'None'}\n"
                f"Known Information:\n{subtask.known_information or 'None'}\n\n"
                "Use tools as needed. Stay within the subtask scope.\n"
                f"{self._subreport_output_contract()}"
            ),
            tool_registry=self.tool_registry,
            max_tool_calls=subtask.max_tool_calls,
            max_files=subtask.max_files,
            max_ask_file_calls=subtask.max_ask_file_calls,
        )
        report_payload = self._parse_subreport_payload(response.content)
        files_checked, symbols_checked, file_contents = self._collect_execution_artifacts(
            executed_tools=executed_tools,
            max_files=subtask.max_files,
        )
        observations = self._observations_from_evidence_spans(
            report_payload["evidence_spans"],
            file_contents=file_contents,
            next_id_start=1,
        )
        return SubInvestigationReport(
            id=f"{subtask.id}-report",
            parent_task_id=subtask.parent_task_id,
            subtask_id=subtask.id,
            question=subtask.question,
            answer=report_payload["answer"],
            confidence=report_payload["confidence"],
            observations=observations,
            files_checked=files_checked,
            symbols_checked=symbols_checked,
            unresolved=report_payload["unresolved"],
            profile_update_suggestion=report_payload.get("profile_update_suggestion"),
            additional_tool_calls_needed=report_payload["additional_tool_calls_needed"],
            additional_file_reads_needed=report_payload["additional_file_reads_needed"],
        )

    @staticmethod
    def _subreport_output_contract() -> str:
        return """
Final output contract:
Return exactly one strict JSON object and no prose or Markdown fences.

Required top-level keys:
{
  "answer": "short synthesis bounded by inspected evidence",
  "confidence": "high | medium | low",
  "unresolved": [],
  "profile_update_suggestion": null,
  "evidence_spans": [],
  "additional_tool_calls_needed": 0,
  "additional_file_reads_needed": 0
}

Hard requirements:
- `confidence` must be exactly lowercase `high`, `medium`, or `low`.
- `unresolved` must be a list of strings.
- `profile_update_suggestion` must be a string or null.
- `additional_tool_calls_needed` and `additional_file_reads_needed` must be integers.
- Every evidence span must contain `file_path`, `start_line`, `end_line`, and `summary`.
- Evidence span line numbers must be positive integers; never use 0.
- Evidence spans may only cite files inspected with `ask_file` or `read_file`.
- Do not cite read_repo_tree, find_text, trace_symbol, directory listings, or unread files in `evidence_spans`.
- If budget was exhausted, lower confidence when appropriate, list missing checks in `unresolved`, and estimate additional budget fields.
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
            elif tool_name == "ask_file":
                path = str(metadata.get("path") or arguments.get("path") or "")
                if path and path not in files_checked and len(files_checked) < max_files:
                    files_checked.append(path)
                numbered_content = metadata.get("numbered_content")
                if path and isinstance(numbered_content, str):
                    file_contents[path] = numbered_content
            elif tool_name == "trace_symbol":
                symbol_name = str(metadata.get("symbol_name") or arguments.get("symbol_name") or "")
                if symbol_name and symbol_name not in symbols_checked:
                    symbols_checked.append(symbol_name)
        return files_checked, symbols_checked, file_contents

    def _observations_from_evidence_spans(
        self,
        spans: list[dict[str, Any]],
        *,
        file_contents: dict[str, str],
        next_id_start: int,
    ) -> list[Observation]:
        observations: list[Observation] = []
        for index, span in enumerate(spans, start=next_id_start):
            file_path = span["file_path"]
            if file_path not in file_contents:
                raise RuntimeError(
                    f"InvestigatorAgent received evidence span for unread file: {file_path}"
                )
            start_line = span["start_line"]
            end_line = span["end_line"]
            if start_line <= 0 or end_line < start_line:
                raise RuntimeError(
                    f"InvestigatorAgent received invalid evidence span range: {span}"
                )
            excerpt = self._extract_excerpt(
                numbered_content=file_contents[file_path],
                start_line=start_line,
                end_line=end_line,
            )
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
        return self._dedupe_observations(observations)

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
