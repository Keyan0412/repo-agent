# repo-agent

`repo-agent` is an LLM-powered repository analysis assistant built for codebase inspection, evidence collection, and structured investigation.

Instead of treating repository analysis as a single flat chat loop, the project separates high-level reasoning from low-level code inspection. The goal is to make repository answers more traceable, more debuggable, and easier to extend.

## What It Does

- Analyzes a code repository with tool-assisted LLM workflows.
- Separates planning, investigation, and final synthesis into different agent roles.
- Stores repository-level working memory under `.cache/repo-agent/`.
- Records every model request, response, and model-call failure for debugging.

## Current Status

This project is still under active development.

What is already implemented:

- `LLMClient` with tool-calling support.
- `InvestigatorAgent` for repository inspection under tool/file budgets.
- Repository tools such as `read_repo_tree`, `find_text`, `trace_symbol`, and `read_file`.
- Persistent cache for repository profiles, reports, and LLM call logs.

What is still incomplete:

- The full end-to-end `MainAgent` workflow.
- Complete `AnalyzerAgent` / `MainAgent` orchestration.
- A polished user-facing CLI beyond the current inspection/demo flow.

## Architecture

The project uses a multi-layer design:

- `MainAgent`: high-level reasoning, investigation scheduling, and final answer synthesis.
- `AnalyzerAgent`: task decomposition and repository-profile-oriented analysis.
- `InvestigatorAgent`: direct repository inspection through tools with explicit budgets.

This separation is intentional: high-level reasoning should not directly perform unrestricted repository reads, and low-level code inspection should not be responsible for final answer policy.

## Installation

Requirements:

- Python 3.12+
- A DashScope-compatible API key

Install dependencies:

```bash
pip install -e .
```

## Configuration

Set the following environment variables in `.env` or in your shell:

```env
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
REPO_AGENT_MODEL=qwen-plus
REPO_AGENT_ENABLE_THINKING=false
```

Notes:

- `DASHSCOPE_API_KEY` is required.
- `REPO_AGENT_MODEL` defaults to `qwen-plus`.
- `DASHSCOPE_BASE_URL` defaults to the DashScope OpenAI-compatible endpoint.
- `REPO_AGENT_ENABLE_THINKING` controls the extra request flag sent to the backend.

## Usage

The most practical entrypoint today is the investigator demo:

```bash
python inspect/investigator_agent_demo.py summarize --repo-root .
```

Run a sub-investigation:

```bash
python inspect/investigator_agent_demo.py subtask --repo-root .
```

Useful options:

- `--model`: override the model name.
- `--repo-root`: choose which repository the tools inspect.
- `--max-tool-calls`: limit tool usage for a subtask.
- `--max-files`: limit `read_file` calls for a subtask.

## Cache and Debug Output

`repo-agent` writes repository-scoped state under:

```text
.cache/repo-agent/
```

Files currently used:

- `repo_profile.md`: natural-language repository profile used by higher-level analysis.
- `reports/`: cached investigation reports.
- `llm_calls.jsonl`: one JSON object per model call, including request payloads, model responses, and model-call exceptions.

The `llm_calls.jsonl` file is especially useful when debugging:

- prompt formatting issues
- invalid tool-call arguments
- unexpected model output
- backend/API failures

## Tool Boundaries

Repository inspection tools are intentionally scoped:

- `read_repo_tree`
- `find_text`
- `trace_symbol`
- `read_file`

These are used by the investigation layer rather than exposed as unrestricted high-level behavior. This keeps repository access explicit and easier to reason about.

## Testing

Run the full test suite:

```bash
pytest
```

Run only the LLM client tests:

```bash
pytest tests/test_llm_client.py
```

## Repository Layout

Key directories:

- `src/repo_agent/agents/`: agent implementations.
- `src/repo_agent/llm/`: model client and LLM-related utilities.
- `src/repo_agent/tools/`: repository and agent tools.
- `src/repo_agent/cache/`: cache-path and persistence helpers.
- `inspect/`: runnable inspection/demo scripts.
- `tests/`: automated test suite.

## Limitations

- The project is not yet a finished general-purpose CLI product.
- Some top-level orchestration paths are still being built out.
- Current usage is best suited for development, experimentation, and architecture iteration.

## Development Direction

The next major steps are:

- finish `AnalyzerAgent` and `MainAgent` integration
- complete investigation request routing
- strengthen the end-to-end user workflow
- continue improving observability and debugging around model behavior
