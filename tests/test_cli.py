from __future__ import annotations

from repo_agent.cli import _clean_query_input


def test_clean_query_input_removes_arrow_key_escape_sequences() -> None:
    assert _clean_query_input("分析Main\x1b[D\x1b[CAgent") == "分析MainAgent"
    assert _clean_query_input("\x1b[A/exit\x1b[B") == "/exit"


def test_clean_query_input_removes_surrogates_and_control_chars() -> None:
    assert _clean_query_input("分析\udce8schemas\x07") == "分析schemas"
