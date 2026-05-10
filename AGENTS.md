# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 project using a `src/` package layout. Core agent code lives in `src/repo_agent/`: `agents/` contains the main and investigator agents, `tools/` and `toolsets/` define callable repository tools, `runtime/` holds session/config/event types, `llm/` wraps model calls, `cache/` stores reports, and `prompts/` contains packaged Markdown prompts. The FastAPI log viewer lives in `src/repo_agent_viewer/`, with HTML templates under `src/repo_agent_viewer/templates/`. Tests are in `tests/`, and runnable demos are in `inspect/`. Design notes and cleanup plans are kept in `development doc/`.

## Build, Test, and Development Commands

- `pip install -e .` installs the package and console scripts in editable mode.
- `pytest` runs the full test suite configured by `pyproject.toml`.
- `repo-agent --repo-root . "How does this project implement the MainAgent workflow?"` runs the main CLI against this repository.
- `python inspect/investigator_agent_demo.py --repo-root .` runs the investigator demo directly.
- `repo-agent-viewer --jsonl-path .cache/repo-agent/llm_calls.jsonl` starts the local viewer at `http://127.0.0.1:8000`.

## Coding Style & Naming Conventions

Use idiomatic Python with 4-space indentation, type hints where they clarify interfaces, and small modules that match existing ownership boundaries. Prefer `snake_case` for functions, variables, modules, and test files; use `PascalCase` for classes such as `MainAgent` and `InvestigatorAgent`. Keep prompt files as Markdown in `src/repo_agent/prompts/`, and add new package data to `pyproject.toml` if a new non-Python asset must ship with the package.

## Testing Guidelines

Tests use pytest and should be added under `tests/` with names like `test_report_store.py` or `test_main_agent.py`. Keep tests focused on public behavior: tool execution, cache behavior, LLM client request shaping, and agent control flow. Prefer fixtures in `tests/conftest.py` for shared setup. Run `pytest` before opening a PR; add regression tests for bug fixes and behavior changes.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `add summarizer`, `add viewer`, and `initialize`. Follow that style: concise, lowercase when natural, and focused on the change. Pull requests should include a brief description, commands run (`pytest`, demos, or viewer smoke tests), linked issues when applicable, and screenshots only for visible viewer changes.

## Security & Configuration Tips

Do not commit `.env` or API keys. Use `.env.example` as the template for `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, and model settings such as `REPO_AGENT_COMPLEX_MODEL`. Runtime cache and debug output belong under `.cache/repo-agent/`; avoid committing generated reports or JSONL logs.
