from __future__ import annotations

from pydantic import BaseModel


class Observation(BaseModel):
    id: int
    summary: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    excerpt: str | None = None
