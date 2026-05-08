from __future__ import annotations

import json
from typing import Any

from repo_agent.llm.client import LLMClient


class JsonRepairAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def repair_json(
        self,
        *,
        raw_content: str,
        target_name: str,
        json_schema: dict[str, Any],
        error: Exception,
    ) -> str:
        response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_content(
                        raw_content=raw_content,
                        target_name=target_name,
                        json_schema=json_schema,
                        error=error,
                    ),
                },
            ],
            tool_choice="none",
            temperature=0,
        )
        repaired = response.content.strip()
        if not repaired:
            raise RuntimeError("JsonRepairAgent returned empty content")
        return repaired

    @staticmethod
    def _system_prompt() -> str:
        return """
You are JsonRepairAgent. You repair malformed structured JSON output.

Your role is format repair only.
You must preserve the original meaning and evidence.
You must not add new facts, files, line numbers, conclusions, or evidence.
You must not remove facts unless they are impossible to represent in the target schema.

Allowed repairs:
- remove Markdown code fences
- remove prose before or after the JSON object
- convert JSON-like text into valid JSON
- normalize object keys to the target schema
- add missing optional fields as null, [], or 0 when the schema clearly allows it
- convert obvious enum casing to the schema value, such as Medium to medium
- convert numeric strings to integers only when the value is plainly numeric

Return exactly one strict JSON object.
Do not wrap the JSON in Markdown fences.
Do not include explanations, comments, or extra text.
""".strip()

    @staticmethod
    def _user_content(
        *,
        raw_content: str,
        target_name: str,
        json_schema: dict[str, Any],
        error: Exception,
    ) -> str:
        return f"""
Target output type:
{target_name}

Target JSON schema:
{json.dumps(json_schema, ensure_ascii=False, indent=2)}

Original parse error:
{type(error).__name__}: {error}

Raw model output to repair:
<raw_output>
{raw_content}
</raw_output>

Repair the raw output into one JSON object matching the target schema.
Preserve the original content's meaning. Do not add evidence or conclusions.
""".strip()
