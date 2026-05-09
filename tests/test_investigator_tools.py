from pathlib import Path

from repo_agent.tools.repo import FindTextTool, ReadRepoTreeTool, TraceSymbolTool


def test_read_repo_tree_lists_directory_and_truncates_large_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    for index in range(4):
        (repo / f"file_{index}.txt").write_text("x\n", encoding="utf-8")

    tool = ReadRepoTreeTool(repo, max_entries_per_dir=3)
    result = tool.execute({"path": ".", "max_depth": 2})

    assert result.success is True
    assert "./" in result.content
    assert "src/" in result.content
    assert "main.py" in result.content
    assert "... (2 more entries)" in result.content


def test_read_repo_tree_read_all_requires_depth_one(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    tool = ReadRepoTreeTool(repo)
    result = tool.execute({"path": ".", "max_depth": 2, "read_all": True})

    assert result.success is False
    assert "read_all=True requires max_depth=1" in result.content


def test_find_text_returns_file_line_and_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("class Planner:\n    pass\n", encoding="utf-8")
    (repo / "b.py").write_text("planner = Planner()\n", encoding="utf-8")

    tool = FindTextTool(repo)
    result = tool.execute({"query": "planner", "max_results": 10})

    assert result.success is True
    assert "a.py:1: class Planner:" in result.content
    assert "b.py:1: planner = Planner()" in result.content


def test_find_text_respects_max_results(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("planner\nplanner\nplanner\n", encoding="utf-8")

    tool = FindTextTool(repo)
    result = tool.execute({"query": "planner", "max_results": 2})

    assert result.success is True
    assert result.metadata["truncated"] is True
    assert result.metadata["match_count"] == 2
    assert result.content.count("a.py:") == 2


def test_find_text_falls_back_to_literal_search_for_invalid_regex(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("available tools (request_investigation\n", encoding="utf-8")

    tool = FindTextTool(repo)
    result = tool.execute({"query": "available tools (request_investigation"})

    assert result.success is True
    assert "a.py:1: available tools (request_investigation" in result.content
    assert result.metadata["literal_fallback"] is True


def test_repo_tools_do_not_ignore_names_by_default(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".hidden").mkdir()
    (repo / ".hidden" / "note.txt").write_text("planner\n", encoding="utf-8")

    tree_tool = ReadRepoTreeTool(repo)
    tree_result = tree_tool.execute({"path": ".", "max_depth": 2})

    search_tool = FindTextTool(repo)
    search_result = search_tool.execute({"query": "planner"})

    assert tree_result.success is True
    assert ".hidden/" in tree_result.content
    assert "note.txt" in tree_result.content
    assert search_result.success is True
    assert ".hidden/note.txt:1: planner" in search_result.content


def test_trace_symbol_finds_definition_and_usages_via_ast(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        "class Planner:\n"
        "    pass\n"
        "\n"
        "planner = Planner()\n"
        "result = planner\n",
        encoding="utf-8",
    )
    (repo / "worker.py").write_text(
        "from app import Planner\n"
        "\n"
        "def build():\n"
        "    return Planner()\n",
        encoding="utf-8",
    )

    tool = TraceSymbolTool(repo)
    result = tool.execute({"symbol_name": "Planner", "max_results": 10})

    assert result.success is True
    assert "app.py:1:1: definition (class): class Planner:" in result.content
    assert "app.py:4:" in result.content
    assert "worker.py:1:" in result.content
    assert "worker.py:4:" in result.content
    assert result.metadata["match_count"] >= 4
    assert any(
        item["occurrence_type"] == "definition" and item["symbol_kind"] == "class"
        for item in result.metadata["occurrences"]
    )
    assert any(item["occurrence_type"] == "usage" for item in result.metadata["occurrences"])


def test_trace_symbol_finds_variable_definition_and_usage(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "vars.py").write_text(
        "value = 1\n"
        "other = value + 2\n",
        encoding="utf-8",
    )

    tool = TraceSymbolTool(repo)
    result = tool.execute({"symbol_name": "value"})

    assert result.success is True
    assert "vars.py:1:1: definition (variable): value = 1" in result.content
    assert "vars.py:2:" in result.content


def test_trace_symbol_respects_max_results(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "many.py").write_text(
        "target = 0\n"
        "a = target\n"
        "b = target\n"
        "c = target\n",
        encoding="utf-8",
    )

    tool = TraceSymbolTool(repo)
    result = tool.execute({"symbol_name": "target", "max_results": 2})

    assert result.success is True
    assert result.metadata["truncated"] is True
    assert len(result.metadata["occurrences"]) == 2
