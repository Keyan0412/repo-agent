from __future__ import annotations

from pathlib import Path


class CachePaths:
    def __init__(self, repo_path: Path, cache_dir: str = ".cache/repo-agent") -> None:
        self.repo_path = Path(repo_path).resolve()
        self.cache_root = self.repo_path / cache_dir
        self.reports_dir = self.cache_root / "reports"
        self.runs_dir = self.cache_root / "runs"

    def ensure_dirs(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
