from __future__ import annotations

from pathlib import Path

from .paths import CachePaths


class RepoProfileStore:
    def __init__(self, repo_path: Path, cache_dir: str = ".cache/repo-agent") -> None:
        self.paths = CachePaths(repo_path, cache_dir)

    def load(self) -> str | None:
        if not self.paths.profile_path.exists():
            return None
        return self.paths.profile_path.read_text(encoding="utf-8")

    def save(self, profile: str) -> None:
        self.paths.ensure_dirs()
        self.paths.profile_path.write_text(profile.strip() + "\n", encoding="utf-8")

    def exists(self) -> bool:
        return self.paths.profile_path.exists()
