from __future__ import annotations

from pathlib import Path


class CachePaths:
    def __init__(self, repo_path: Path, cache_dir: str = ".cache/repo-agent") -> None:
        self.repo_path = Path(repo_path).resolve()
        self.cache_root = self.repo_path / cache_dir
        self.profile_path = self.cache_root / "repo_profile.md"
        self.reports_dir = self.cache_root / "reports"
        self.llm_calls_path = self.cache_root / "llm_calls.jsonl"

    def ensure_dirs(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
