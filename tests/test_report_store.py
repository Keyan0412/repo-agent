from pathlib import Path

from repo_agent.cache import ReportStore
from repo_agent.investigation import InvestigationReport, Observation


def test_report_store_saves_markdown_and_loads_recent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = ReportStore(repo)
    report = InvestigationReport(
        id="R1",
        task_id="T1",
        summary="Summary",
        observations=[
            Observation(
                id=1,
                summary="Found planner flow",
                file_path="app.py",
                start_line=10,
                end_line=12,
            )
        ],
        files_checked=["app.py"],
        remaining_questions=["Need to inspect retries"],
    )

    path = store.save(report, slug="architecture")

    assert path.exists() is True
    loaded = store.load_recent(limit=1)
    assert len(loaded) == 1
    assert "调查报告" in loaded[0]
    assert "planner flow" in loaded[0]
    assert "Profile" not in loaded[0]
