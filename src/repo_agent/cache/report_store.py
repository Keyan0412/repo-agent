from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from repo_agent.investigation.report import InvestigationReport

from .paths import CachePaths


class ReportStore:
    def __init__(self, repo_path: Path, cache_dir: str = ".cache/repo-agent") -> None:
        self.paths = CachePaths(repo_path, cache_dir)

    def save(self, report: InvestigationReport, slug: str | None = None) -> Path:
        self.paths.ensure_dirs()
        filename = self._build_filename(report, slug)
        path = self.paths.reports_dir / filename
        path.write_text(self._to_markdown(report), encoding="utf-8")
        return path

    def list_reports(self) -> list[Path]:
        if not self.paths.reports_dir.exists():
            return []
        return sorted(self.paths.reports_dir.glob("*.md"))

    def load_recent(self, limit: int = 5) -> list[str]:
        reports = self.list_reports()[-limit:]
        return [path.read_text(encoding="utf-8") for path in reports]

    @staticmethod
    def _build_filename(report: InvestigationReport, slug: str | None) -> str:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        base = slug or report.task_id or report.id
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "report"
        return f"{ts}_{safe}.md"

    @staticmethod
    def _to_markdown(report: InvestigationReport) -> str:
        lines = [
            f"# 调查报告: {report.task_id}",
            "",
            f"- 报告 ID: {report.id}",
            f"- 任务 ID: {report.task_id}",
            "",
            "## 摘要",
            "",
            report.summary,
            "",
            "## 关键观察",
            "",
        ]
        if report.observations:
            for observation in report.observations:
                location = ""
                if observation.file_path and observation.start_line is not None:
                    end_line = observation.end_line or observation.start_line
                    location = f"[{observation.file_path}:L{observation.start_line}-L{end_line}] "
                lines.append(f"- {location}{observation.summary}")
        else:
            lines.append("- 无")

        lines.extend(["", "## 已检查文件", ""])
        if report.files_checked:
            lines.extend(f"- {path}" for path in report.files_checked)
        else:
            lines.append("- 无")

        lines.extend(["", "## 剩余问题", ""])
        if report.remaining_questions:
            lines.extend(f"- {item}" for item in report.remaining_questions)
        else:
            lines.append("- 无")

        lines.extend(["", "## 子报告", ""])
        if report.subreports:
            for subreport in report.subreports:
                lines.extend(
                    [
                        f"### {subreport.question}",
                        "",
                        f"- 回答: {subreport.answer}",
                        f"- 置信度: {subreport.confidence}",
                    ]
                )
                if subreport.observations:
                    lines.append("- 观察摘要:")
                    lines.extend(f"  - {obs.summary}" for obs in subreport.observations)
                if subreport.unresolved:
                    lines.append("- 未解决:")
                    lines.extend(f"  - {item}" for item in subreport.unresolved)
                lines.append("")
        else:
            lines.append("- 无")

        return "\n".join(lines)
