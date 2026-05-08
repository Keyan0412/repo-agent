from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repo_agent.agents.investigator_agent import InvestigatorAgent
from repo_agent.agents.json_repair_agent import JsonRepairAgent
from repo_agent.cache import RepoProfileStore, ReportStore
from repo_agent.investigation import (
    AnalysisPlan,
    InvestigationReport,
    InvestigationTask,
    Observation,
    SubInvestigationReport,
    SubInvestigationTask,
)
from repo_agent.llm.client import LLMClient
from repo_agent.llm.schemas import AnalyzerPlanPayload, AnalyzerReportPayload
from repo_agent.runtime.session import AgentSession


class AnalyzerAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        session: AgentSession,
        repo_path: Path,
        investigator: InvestigatorAgent,
        *,
        profile_store: RepoProfileStore | None = None,
        report_store: ReportStore | None = None,
        analyzer_prompt_path: Path | None = None,
        repo_profile_update_prompt_path: Path | None = None,
        recent_report_limit: int = 3,
        max_subtasks: int = 4,
        repo_profile_max_ask_file_calls: int | None = 40,
        json_repair_agent: JsonRepairAgent | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.session = session
        self.repo_path = Path(repo_path).resolve()
        self.investigator = investigator
        self.profile_store = profile_store or RepoProfileStore(self.repo_path)
        self.report_store = report_store or ReportStore(self.repo_path)
        self.recent_report_limit = recent_report_limit
        self.max_subtasks = max_subtasks
        self.repo_profile_max_ask_file_calls = repo_profile_max_ask_file_calls
        self.json_repair_agent = json_repair_agent or JsonRepairAgent(llm_client)

        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        self.analyzer_prompt_path = analyzer_prompt_path or prompts_dir / "analyzer_agent.md"
        self.repo_profile_update_prompt_path = (
            repo_profile_update_prompt_path or prompts_dir / "repo_profile_update.md"
        )

    def investigate(self, task: InvestigationTask) -> InvestigationReport:
        self._ensure_repo_profile(task)
        prior_reports = self.report_store.load_recent(limit=self.recent_report_limit)
        plan = self._make_analysis_plan(
            task=task,
            repo_profile=self.session.repo_profile or "",
            prior_reports=prior_reports,
        )

        subreports = [
            self.investigator.investigate_subtask(subtask)
            for subtask in plan.subquestions
        ]
        report = self._synthesize_report(
            task=task,
            plan=plan,
            subreports=subreports,
        )

        if self._should_update_repo_profile(report):
            updated_profile = self._rewrite_repo_profile(
                old_profile=self.session.repo_profile or "",
                report=report,
            )
            if updated_profile:
                self.session.repo_profile = updated_profile
                self.profile_store.save(updated_profile)

        self.report_store.save(report)
        self.session.reports.append(report)
        return report

    def _ensure_repo_profile(self, task: InvestigationTask) -> None:
        if self.session.repo_profile and self.session.repo_profile.strip():
            return

        cached_profile = self.profile_store.load()
        if cached_profile and cached_profile.strip():
            self.session.repo_profile = cached_profile
            return

        profile = self.investigator.summarize_repo(
            user_query=task.user_query,
            task=task.task,
            max_ask_file_calls=self.repo_profile_max_ask_file_calls,
        )
        self.session.repo_profile = profile
        self.profile_store.save(profile)

    def _make_analysis_plan(
        self,
        *,
        task: InvestigationTask,
        repo_profile: str,
        prior_reports: list[str],
    ) -> AnalysisPlan:
        prompt = self.analyzer_prompt_path.read_text(encoding="utf-8").strip()
        response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": self._plan_user_content(
                        task=task,
                        repo_profile=repo_profile,
                        prior_reports=prior_reports,
                    ),
                },
            ],
            tool_choice="none",
            temperature=0,
        )
        payload = self._parse_plan_payload(response.content)
        if not payload.subquestions:
            raise RuntimeError("AnalyzerAgent received an empty analysis plan from the LLM")
        if len(payload.subquestions) > self.max_subtasks:
            raise RuntimeError(
                f"AnalyzerAgent received too many subquestions: {len(payload.subquestions)}"
            )

        subquestions = []
        for index, item in enumerate(payload.subquestions, start=1):
            if item.max_tool_calls <= 0:
                raise RuntimeError("AnalyzerAgent received non-positive max_tool_calls")
            if item.max_files <= 0:
                raise RuntimeError("AnalyzerAgent received non-positive max_files")
            if item.max_ask_file_calls < item.max_files * 3:
                raise RuntimeError(
                    "AnalyzerAgent received max_ask_file_calls below 3x max_files"
                )
            subquestions.append(
                SubInvestigationTask(
                    id=f"{task.id}-S{index}",
                    parent_task_id=task.id,
                    question=item.question,
                    purpose=item.purpose,
                    expected_evidence=item.expected_evidence,
                    known_information=item.known_information,
                    max_tool_calls=item.max_tool_calls,
                    max_files=item.max_files,
                    max_ask_file_calls=item.max_ask_file_calls,
                )
            )

        return AnalysisPlan(
            task_id=task.id,
            goal=payload.goal,
            subquestions=subquestions,
            synthesis_strategy=payload.synthesis_strategy,
        )

    def _synthesize_report(
        self,
        *,
        task: InvestigationTask,
        plan: AnalysisPlan,
        subreports: list[SubInvestigationReport],
    ) -> InvestigationReport:
        response = self.llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are AnalyzerAgent. Synthesize sub-investigation reports into one concise "
                        "InvestigationReport for MainAgent. Use only the supplied subreports as evidence. "
                        "Return strict JSON with keys: summary, remaining_questions, profile_update_summary."
                    ),
                },
                {
                    "role": "user",
                    "content": self._synthesis_user_content(
                        task=task,
                        plan=plan,
                        subreports=subreports,
                    ),
                },
            ],
            tool_choice="none",
            temperature=0,
        )
        payload = self._parse_report_payload(response.content)
        if not payload.summary.strip():
            raise RuntimeError("AnalyzerAgent received an empty report summary from the LLM")

        return InvestigationReport(
            id=f"R-{task.id}",
            task_id=task.id,
            summary=payload.summary,
            observations=self._collect_observations(subreports),
            files_checked=self._collect_unique_files(subreports),
            remaining_questions=self._collect_remaining_questions(
                subreports=subreports,
                synthesized_remaining=payload.remaining_questions,
            ),
            subreports=subreports,
            profile_update_summary=payload.profile_update_summary,
        )

    def _should_update_repo_profile(self, report: InvestigationReport) -> bool:
        if report.profile_update_summary and report.profile_update_summary.strip().lower() not in {
            "none",
            "null",
            "no update",
        }:
            return True
        return any(
            subreport.profile_update_suggestion
            and subreport.profile_update_suggestion.strip()
            for subreport in report.subreports
        )

    def _rewrite_repo_profile(self, *, old_profile: str, report: InvestigationReport) -> str:
        prompt = self.repo_profile_update_prompt_path.read_text(encoding="utf-8").strip()
        response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Existing RepoProfile:\n{old_profile or 'None'}\n\n"
                        f"Investigation Report:\n{report.model_dump_json(indent=2)}\n\n"
                        "Return the complete updated RepoProfile as Markdown only."
                    ),
                },
            ],
            tool_choice="none",
            temperature=0,
        )
        return response.content.strip()

    def _parse_plan_payload(self, response_content: str) -> AnalyzerPlanPayload:
        payload = self.llm_client.extract_json_object(
            response_content,
            repair_agent=self.json_repair_agent,
            target_name="AnalyzerPlanPayload",
            json_schema=AnalyzerPlanPayload.model_json_schema(),
        )
        try:
            return AnalyzerPlanPayload.model_validate(payload)
        except Exception as exc:
            raise RuntimeError(
                "AnalyzerAgent received a plan with invalid field types: "
                f"{payload}"
            ) from exc

    def _parse_report_payload(self, response_content: str) -> AnalyzerReportPayload:
        payload = self.llm_client.extract_json_object(
            response_content,
            repair_agent=self.json_repair_agent,
            target_name="AnalyzerReportPayload",
            json_schema=AnalyzerReportPayload.model_json_schema(),
        )
        try:
            return AnalyzerReportPayload.model_validate(payload)
        except Exception as exc:
            raise RuntimeError(
                "AnalyzerAgent received a report with invalid field types: "
                f"{payload}"
            ) from exc

    @staticmethod
    def _plan_user_content(
        *,
        task: InvestigationTask,
        repo_profile: str,
        prior_reports: list[str],
    ) -> str:
        """user prompts for analyzer agent, including investigation task, repo_profile, prior reports."""
        reports_text = "\n\n---\n\n".join(prior_reports) if prior_reports else "None"
        return f"""
InvestigationTask:
{task.model_dump_json(indent=2)}

RepoProfile:
{repo_profile or 'None'}

Recent Reports:
{reports_text}

Budgeting guidance:
Each subquestion must set `max_ask_file_calls` separately from `max_files`.
Set `max_ask_file_calls` to at least 3-4 times `max_files` because `ask_file`
is cheaper and should be preferred for implementation-status and local-behavior checks.

Return strict JSON only:
{{
  "goal": "single sentence goal",
  "subquestions": [
    {{
      "question": "focused repo question",
      "purpose": "why this matters",
      "expected_evidence": ["specific evidence type"],
      "known_information": "short hints for InvestigatorAgent",
      "max_tool_calls": 4,
      "max_files": 3,
      "max_ask_file_calls": 12
    }}
  ],
  "synthesis_strategy": "how to combine the subreports"
}}
""".strip()

    @staticmethod
    def _synthesis_user_content(
        *,
        task: InvestigationTask,
        plan: AnalysisPlan,
        subreports: list[SubInvestigationReport],
    ) -> str:
        return f"""
InvestigationTask:
{task.model_dump_json(indent=2)}

AnalysisPlan:
{plan.model_dump_json(indent=2)}

SubInvestigationReports:
{json.dumps([subreport.model_dump() for subreport in subreports], ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _collect_observations(subreports: list[SubInvestigationReport]) -> list[Observation]:
        observations: list[Observation] = []
        seen: set[tuple[Any, ...]] = set()
        for subreport in subreports:
            for observation in subreport.observations:
                key = (
                    observation.summary,
                    observation.file_path,
                    observation.start_line,
                    observation.end_line,
                    observation.excerpt,
                )
                if key in seen:
                    continue
                seen.add(key)
                observations.append(observation.model_copy(update={"id": len(observations) + 1}))
        return observations

    @staticmethod
    def _collect_unique_files(subreports: list[SubInvestigationReport]) -> list[str]:
        files: list[str] = []
        for subreport in subreports:
            for path in subreport.files_checked:
                if path not in files:
                    files.append(path)
        return files

    @staticmethod
    def _collect_remaining_questions(
        *,
        subreports: list[SubInvestigationReport],
        synthesized_remaining: list[str],
    ) -> list[str]:
        remaining: list[str] = []
        for item in synthesized_remaining:
            if item not in remaining:
                remaining.append(item)
        for subreport in subreports:
            for item in subreport.unresolved:
                if item not in remaining:
                    remaining.append(item)
        return remaining
