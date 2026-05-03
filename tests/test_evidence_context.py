from repo_agent.evidence import EvidenceContextBuilder, EvidenceGraph, EvidenceNode
from repo_agent.investigation import InvestigationReport, Observation


def test_context_builder_uses_top_level_evidence_only() -> None:
    graph = EvidenceGraph()
    e0 = graph.add(EvidenceNode(type="investigate_result", claim="low-level fact"))
    graph.add(
        EvidenceNode(
            type="derived_claim",
            claim="high-level claim",
            based_on=[e0],
            confidence=0.82,
            limitations=["tool execution loop unverified"],
        )
    )

    report = InvestigationReport(
        id="R2",
        task_id="task-2",
        summary="Searched the workflow implementation",
        observations=[Observation(id=1, summary="Found planner orchestration")],
        remaining_questions=["Need to inspect retry behavior"],
    )

    context = EvidenceContextBuilder().build("What architecture is this?", graph, [report])

    assert "[E1] high-level claim" in context
    assert "[E0] low-level fact" not in context
    assert "- Confidence: 0.82" in context
    assert "- Based on: E0" in context
    assert "tool execution loop unverified" in context
    assert "[R2] Searched the workflow implementation" in context

    #print(context)


if __name__ == "__main__":
    test_context_builder_uses_top_level_evidence_only()
