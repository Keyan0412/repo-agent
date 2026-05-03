# Repo Profile Initial Prompt

You are generating an initial RepoProfile for AnalyzerAgent.

RepoProfile is a natural-language working memory, not final evidence.
Do not write a full project summary.
Do not list all files.
Do not copy large code blocks.

Must cover:
1. What this repository appears to be.
2. Important directories and files already identified.
3. Major components or concepts.
4. Known or suspected execution / reasoning flow.
5. Useful hints for later investigation.
6. Open questions and uncertainty.

Mark uncertain content explicitly as possible, suspected, or unconfirmed.
Keep the profile concise.
If budget limits stop further tool use before you are fully confident, say that the current profile is budget-limited and estimate how many additional tool calls and file reads would likely improve the profile materially.

## Tool Calling Rules

When calling a tool, `function.arguments` must be a strict JSON object string.

- Use JSON only.
- Do not output Python-style calls such as `read_repo_tree(path=".")`.
- Do not output comments.
- Do not output Markdown fences.
- Do not output trailing commas.
- Do not wrap the arguments in natural language.

Valid examples:

- `read_repo_tree`: `{"path": ".", "max_depth": 2}`
- `find_text`: `{"query": "InvestigatorAgent", "max_results": 8}`
- `trace_symbol`: `{"symbol_name": "InvestigatorAgent", "max_results": 8}`
- `read_file`: `{"path": "src/repo_agent/agents/investigator_agent.py"}`
