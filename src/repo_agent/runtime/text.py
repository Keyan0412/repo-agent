from __future__ import annotations

from typing import Any


def strip_surrogates(text: str) -> str:
    return "".join(char for char in text if not 0xD800 <= ord(char) <= 0xDFFF)


def strip_surrogates_from_json(value: Any) -> Any:
    if isinstance(value, str):
        return strip_surrogates(value)
    if isinstance(value, list):
        return [strip_surrogates_from_json(item) for item in value]
    if isinstance(value, tuple):
        return [strip_surrogates_from_json(item) for item in value]
    if isinstance(value, dict):
        return {
            strip_surrogates_from_json(key): strip_surrogates_from_json(item)
            for key, item in value.items()
        }
    return value
