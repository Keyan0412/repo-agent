from pathlib import Path

from repo_agent.tools.repo import FindFilesTool, FindTextTool, ListDirTool, TraceSymbolTool


def test_list_dir_lists_files_and_directories_with_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "pkg").mkdir()
    (repo / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    tool = ListDirTool(repo)
    result = tool.execute({"path": "src"})

    assert result.success is True
    assert '<directory_listing path="src"' in result.content
    assert "dir  src/pkg/" in result.content
    assert "file src/main.py  12 bytes, 1 lines" in result.content
    assert result.metadata["path"] == "src"
    assert result.metadata["entry_count"] == 2
    assert result.metadata["truncated"] is False
    assert result.metadata["entries"] == [
        {
            "path": "src/pkg",
            "type": "dir",
            "size_bytes": None,
            "line_count": None,
        },
        {
            "path": "src/main.py",
            "type": "file",
            "size_bytes": 12,
            "line_count": 1,
        },
    ]


def test_list_dir_supports_recursive_listing_and_truncation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "pkg").mkdir()
    (repo / "src" / "pkg" / "mod.py").write_text("x\n", encoding="utf-8")
    (repo / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    tool = ListDirTool(repo)
    result = tool.execute({"path": "src", "recursive": True, "max_entries": 2})

    assert result.success is True
    assert result.metadata["recursive"] is True
    assert result.metadata["entry_count"] == 2
    assert result.metadata["total_entry_count"] == 3
    assert result.metadata["truncated"] is True
    assert "... (1 more entries)" in result.content


def test_find_files_matches_filename_and_path_globs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "agents").mkdir()
    (repo / "src" / "agents" / "main_agent.py").write_text("x\n", encoding="utf-8")
    (repo / "src" / "tools.py").write_text("x\n", encoding="utf-8")
    (repo / "README.md").write_text("x\n", encoding="utf-8")

    tool = FindFilesTool(repo)
    result = tool.execute({"pattern": "*agent.py"})

    assert result.success is True
    assert "src/agents/main_agent.py" in result.content
    assert result.metadata["paths"] == ["src/agents/main_agent.py"]
    assert result.metadata["match_count"] == 1

    path_result = tool.execute({"pattern": "src/*.py"})

    assert path_result.success is True
    assert path_result.metadata["paths"] == ["src/tools.py"]


def test_find_files_respects_path_and_truncation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pkg").mkdir()
    for index in range(3):
        (repo / "pkg" / f"file_{index}.py").write_text("x\n", encoding="utf-8")
    (repo / "outside.py").write_text("x\n", encoding="utf-8")

    tool = FindFilesTool(repo)
    result = tool.execute({"path": "pkg", "pattern": "*.py", "max_results": 2})

    assert result.success is True
    assert result.metadata["path"] == "pkg"
    assert result.metadata["match_count"] == 2
    assert result.metadata["total_match_count"] == 3
    assert result.metadata["truncated"] is True
    assert "outside.py" not in result.content


def test_find_text_returns_file_line_and_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("class Planner:\n    pass\n", encoding="utf-8")
    (repo / "b.py").write_text("planner = Planner()\n", encoding="utf-8")

    tool = FindTextTool(repo)
    result = tool.execute({"query": "planner"})

    assert result.success is True
    assert "a.py:1: class Planner:" in result.content
    assert "b.py:1: planner = Planner()" in result.content
    assert "以上为所有结果。" in result.content
    assert result.metadata["page"] == 1
    assert result.metadata["page_size"] == 20
    assert result.metadata["has_next_page"] is False


def test_find_text_paginates_twenty_results_at_a_time(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("".join("planner\n" for _ in range(25)), encoding="utf-8")

    tool = FindTextTool(repo)
    result = tool.execute({"query": "planner"})

    assert result.success is True
    assert result.metadata["truncated"] is True
    assert result.metadata["has_next_page"] is True
    assert result.metadata["next_page"] == 2
    assert result.metadata["match_count"] == 20
    assert result.content.count("a.py:") == 20
    assert "可以使用 page=2 继续读取" in result.content

    second_page = tool.execute({"query": "planner", "page": 2})

    assert second_page.success is True
    assert second_page.metadata["truncated"] is False
    assert second_page.metadata["has_next_page"] is False
    assert second_page.metadata["match_count"] == 5
    assert second_page.content.count("a.py:") == 5
    assert "以上为所有结果。" in second_page.content


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

    list_tool = ListDirTool(repo)
    list_result = list_tool.execute({"path": ".hidden"})

    search_tool = FindTextTool(repo)
    search_result = search_tool.execute({"query": "planner"})

    assert list_result.success is True
    assert ".hidden/note.txt" in list_result.content
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
