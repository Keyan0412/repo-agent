# repo-agent

`repo-agent` is an LLM-powered repository analysis assistant for codebase inspection and structured investigation.

The current runtime path is intentionally simple:

```text
MainAgent -> InvestigatorAgent -> repository tools
```

`MainAgent` decides what information is missing, asks `InvestigatorAgent` to inspect the repository, and then produces the final answer. `InvestigatorAgent` searches for relevant symbols/text, reads the necessary files directly, and returns a structured investigation report.

## What Is Implemented

- `LLMClient` with OpenAI-compatible tool-calling support.
- `MainAgent` with two tools: `request_investigation` and `final_answer`.
- `InvestigatorAgent` for repository inspection under tool/file budgets.
- Repository tools: `read_repo_tree`, `find_text`, `trace_symbol`, and `read_file`.
- Persistent report cache and JSONL logging for every model call.
- A small FastAPI viewer for inspecting LLM call logs.

The previous extra planning and memory layers have been removed from the active architecture.

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
REPO_AGENT_COMPLEX_MODEL=qwen-plus
REPO_AGENT_SIMPLE_MODEL=qwen-turbo
REPO_AGENT_ENABLE_THINKING=false
```

Notes:

- `REPO_AGENT_COMPLEX_MODEL` is used by `MainAgent` and JSON repair.
- `REPO_AGENT_SIMPLE_MODEL` is used by `InvestigatorAgent`.
- `REPO_AGENT_MODEL` is still supported as a fallback if a tier-specific model is unset.

## Usage

Run the main agent:

```bash
repo-agent --repo-root .
repo-agent> How does this project implement the MainAgent workflow?
```

Run the investigator demo:

```bash
python inspect/investigator_agent_demo.py --repo-root .
```

Useful options:

- `--model`: override the model name.
- `--repo-root`: choose which repository the tools inspect.
- `--max-tool-calls`: limit tool usage for an investigator task.
- `--max-files`: limit `read_file` calls for an investigator task.

## LLM Call Viewer

Start the viewer with:

```bash
repo-agent-viewer --jsonl-path .cache/repo-agent/llm_calls.jsonl
```

Then open:

```text
http://127.0.0.1:8000
```

## Cache and Debug Output

`repo-agent` writes repository-scoped state under:

```text
.cache/repo-agent/
```

Files currently used:

- `reports/`: cached investigation reports.
- `llm_calls.jsonl`: one JSON object per model call, including request payloads, model responses, token usage, and model-call exceptions.

## Tool Boundaries

Repository inspection tools are intentionally scoped:

- `read_repo_tree`
- `find_text`
- `trace_symbol`
- `read_file`

These are used by `InvestigatorAgent`, not exposed as unrestricted top-level behavior.

## Testing

Run the full test suite:

```bash
pytest
```
