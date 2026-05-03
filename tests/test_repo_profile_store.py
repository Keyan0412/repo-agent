from pathlib import Path

from repo_agent.cache import RepoProfileStore


def test_repo_profile_store_save_and_load(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RepoProfileStore(repo)

    store.save("# Repo Profile\n\nExample profile")

    assert store.exists() is True
    assert store.load() == "# Repo Profile\n\nExample profile\n"
