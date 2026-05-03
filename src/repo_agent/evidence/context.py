from __future__ import annotations

from repo_agent.investigation.report import InvestigationReport

from .graph import EvidenceGraph


class EvidenceContextBuilder:
    def build(
        self,
        user_query: str,
        evidence_graph: EvidenceGraph,
        reports: list[InvestigationReport],
    ) -> str:
        sections = [f"User Query:\n{user_query}"]
        sections.append(self._build_evidence_section(evidence_graph))
        sections.append(self._build_reports_section(reports))
        return "\n\n".join(sections)

    def _build_evidence_section(self, evidence_graph: EvidenceGraph) -> str:
        top_level_ids = evidence_graph.get_top_level_ids()
        if not top_level_ids:
            return "Current Top-Level Evidence:\nNone"

        lines = ["Current Top-Level Evidence:"]
        for evidence_id in top_level_ids:
            node = evidence_graph.get(evidence_id)
            lines.append(f"[E{evidence_id}] {node.claim}")
            if node.confidence is not None:
                lines.append(f"- Confidence: {node.confidence:.2f}")
            if node.based_on:
                based_on = ", ".join(f"E{parent_id}" for parent_id in node.based_on)
                lines.append(f"- Based on: {based_on}")
            if node.limitations:
                lines.append("- Limitations:")
                for limitation in node.limitations:
                    lines.append(f"  - {limitation}")
        return "\n".join(lines)

    def _build_reports_section(self, reports: list[InvestigationReport]) -> str:
        if not reports:
            return "Recent Investigation Reports:\nNone"

        lines = ["Recent Investigation Reports:"]
        for report in reports:
            report_id = str(report.id)
            label = report_id if report_id.startswith("R") else f"R{report_id}"
            lines.append(f"[{label}] {report.summary}")
            if report.remaining_questions:
                lines.append("- Remaining Questions:")
                for question in report.remaining_questions:
                    lines.append(f"  - {question}")
        return "\n".join(lines)
