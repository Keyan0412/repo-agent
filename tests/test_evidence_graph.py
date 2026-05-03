from repo_agent.evidence import EvidenceGraph, EvidenceNode, EvidenceStatus


def test_add_node_returns_index_id() -> None:
    graph = EvidenceGraph()

    first_id = graph.add(EvidenceNode(type="investigate_result", claim="first"))
    second_id = graph.add(EvidenceNode(type="derived_claim", claim="second"))

    assert first_id == 0
    assert second_id == 1


def test_add_validates_based_on_ids() -> None:
    graph = EvidenceGraph()
    graph.add(EvidenceNode(type="investigate_result", claim="root"))

    try:
        graph.add(
            EvidenceNode(
                type="derived_claim",
                claim="invalid",
                based_on=[99],
            )
        )
    except ValueError as exc:
        assert "invalid evidence id" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid based_on id")


def test_get_top_level_ids_marks_absorbed_evidence() -> None:
    graph = EvidenceGraph()
    e0 = graph.add(EvidenceNode(type="investigate_result", claim="A"))
    e1 = graph.add(EvidenceNode(type="investigate_result", claim="B"))
    e2 = graph.add(
        EvidenceNode(type="derived_claim", claim="A+B", based_on=[e0, e1])
    )

    assert graph.get_top_level_ids() == [e2]
    assert graph.get_status(e0) == EvidenceStatus.ABSORBED
    assert graph.get_status(e1) == EvidenceStatus.ABSORBED
    assert graph.get_status(e2) == EvidenceStatus.TOP_LEVEL


def test_expand_chain_returns_full_ancestor_chain() -> None:
    graph = EvidenceGraph()
    e0 = graph.add(EvidenceNode(type="investigate_result", claim="A"))
    e1 = graph.add(EvidenceNode(type="investigate_result", claim="B"))
    e2 = graph.add(EvidenceNode(type="derived_claim", claim="A+B", based_on=[e0, e1]))
    e3 = graph.add(EvidenceNode(type="final_claim", claim="final", based_on=[e2]))

    assert graph.expand_chain(e3) == [e0, e1, e2, e3]
    assert graph.expand_many([e2, e3]) == [e0, e1, e2, e3]
