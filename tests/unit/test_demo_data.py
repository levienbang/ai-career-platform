"""Compare the new commands with snapshots captured before the M14 refactor."""

from pathlib import Path

from app.config import Settings


def test_taxonomy_directory_default_empty_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SKILL_DATA_DIR", raising=False)
    assert Settings(_env_file=None).skill_data_dir == Path("data/taxonomy")
    monkeypatch.setenv("SKILL_DATA_DIR", "")
    assert Settings(_env_file=None).skill_data_dir == Path("data/taxonomy")
    monkeypatch.setenv("SKILL_DATA_DIR", str(tmp_path))
    assert Settings(_env_file=None).skill_data_dir == tmp_path
