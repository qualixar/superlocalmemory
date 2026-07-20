"""Ingestion, backup, and migration state must share the canonical root."""

from pathlib import Path


def test_ingestion_runtime_paths_follow_environment_after_import(
    monkeypatch, tmp_path: Path,
) -> None:
    from superlocalmemory.ingestion import (
        adapter_manager,
        calendar_adapter,
        credentials,
        gmail_adapter,
        transcript_adapter,
    )

    selected = (tmp_path / "selected").resolve()
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong-legacy"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook"))

    assert adapter_manager._adapters_config_path() == selected / "adapters.json"
    assert adapter_manager._pid_file("gmail") == selected / "adapter-gmail.pid"
    assert adapter_manager._adapter_log_dir() == selected / "logs"
    assert credentials._credential_dir() == selected / "credentials"
    assert gmail_adapter._adapters_config_path() == selected / "adapters.json"
    assert gmail_adapter._import_dir() == selected / "import"
    assert calendar_adapter._adapters_config_path() == selected / "adapters.json"
    assert calendar_adapter._import_dir() == selected / "import"
    assert transcript_adapter._adapters_config_path() == selected / "adapters.json"


def test_backup_manager_defaults_to_selected_root(monkeypatch, tmp_path: Path) -> None:
    from superlocalmemory.infra.backup import BackupManager

    selected = (tmp_path / "selected").resolve()
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))

    manager = BackupManager()

    assert manager.base_dir == selected
    assert manager.db_path == selected / "memory.db"
    assert manager.backup_dir == selected / "backups"
    assert manager._config_file == selected / "backup_config.json"


def test_v2_source_stays_in_legacy_home_while_v3_destination_is_canonical(
    monkeypatch, tmp_path: Path,
) -> None:
    from superlocalmemory.storage.v2_migrator import V2Migrator

    legacy_home = tmp_path / "legacy-home"
    selected = (tmp_path / "selected-v3").resolve()
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))

    migrator = V2Migrator(home=legacy_home)

    assert migrator._v2_base == legacy_home / ".claude-memory"
    assert migrator._v3_base == selected
    assert migrator._v3_db == selected / "memory.db"
