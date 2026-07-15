"""V3.7 canonical runtime data-root contract.

These tests intentionally keep all aliases different.  Using the same path for
every alias masks partial implementations and was the reason split-root state
survived earlier isolation tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_ALIASES = ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")


def _clear_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ALIASES:
        monkeypatch.delenv(name, raising=False)


def test_environment_alias_precedence_is_canonical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.infra.data_root import canonical_data_root

    primary = tmp_path / "primary" / ".." / "primary"
    legacy = tmp_path / "legacy"
    hook = tmp_path / "hook"
    monkeypatch.setenv("SLM_DATA_DIR", str(primary))
    monkeypatch.setenv("SL_MEMORY_PATH", str(legacy))
    monkeypatch.setenv("SLM_HOME", str(hook))

    assert canonical_data_root() == primary.resolve(strict=False)


@pytest.mark.parametrize(
    ("winner", "losers"),
    [
        ("SL_MEMORY_PATH", ("SLM_DATA_DIR",)),
        ("SLM_HOME", ("SLM_DATA_DIR", "SL_MEMORY_PATH")),
    ],
)
def test_legacy_aliases_remain_input_shims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    winner: str,
    losers: tuple[str, ...],
) -> None:
    from superlocalmemory.infra.data_root import canonical_data_root

    _clear_aliases(monkeypatch)
    for name in losers:
        monkeypatch.setenv(name, "   ")
    selected = tmp_path / winner.lower()
    monkeypatch.setenv(winner, str(selected))

    assert canonical_data_root() == selected.resolve(strict=False)


def test_explicit_environment_root_cannot_be_redirected_by_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.infra.data_root import canonical_data_root

    selected = tmp_path / "selected"
    conflicting = tmp_path / "conflicting"
    selected.mkdir()
    (selected / "config.json").write_text(
        json.dumps({"mode": "a", "base_dir": str(conflicting)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "legacy"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "hook"))

    config = SLMConfig.load()

    assert canonical_data_root() == selected.resolve(strict=False)
    assert config.base_dir == selected.resolve(strict=False)
    assert config.db_path == selected.resolve(strict=False) / "memory.db"


def test_legacy_default_configured_root_is_discoverable_without_moving_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.infra.data_root import canonical_data_root

    _clear_aliases(monkeypatch)
    default_root = tmp_path / ".superlocalmemory"
    configured = tmp_path / "existing-custom-root"
    default_root.mkdir()
    configured.mkdir()
    (default_root / "config.json").write_text(
        json.dumps({"mode": "a", "base_dir": str(configured)}),
        encoding="utf-8",
    )

    assert canonical_data_root(home=tmp_path) == configured.resolve(strict=False)


def test_corrupt_or_relative_legacy_redirect_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.infra.data_root import canonical_data_root

    _clear_aliases(monkeypatch)
    default_root = tmp_path / ".superlocalmemory"
    default_root.mkdir()
    config = default_root / "config.json"

    config.write_text("not-json", encoding="utf-8")
    assert canonical_data_root(home=tmp_path) == default_root.resolve(strict=False)

    config.write_text(json.dumps({"base_dir": "relative/root"}), encoding="utf-8")
    assert canonical_data_root(home=tmp_path) == default_root.resolve(strict=False)


def test_state_path_cannot_escape_the_selected_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.infra.data_root import state_path

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "root"))

    assert state_path("logs", "daemon.log") == (
        tmp_path / "root" / "logs" / "daemon.log"
    ).resolve(strict=False)
    with pytest.raises(ValueError):
        state_path("../outside")
    with pytest.raises(ValueError):
        state_path("/absolute")


def test_two_populated_roots_fail_closed_without_merging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.infra.data_root import (
        DataRootConflictError,
        assert_no_durable_root_conflict,
    )

    default_root = tmp_path / "home" / ".superlocalmemory"
    selected = tmp_path / "selected"
    default_root.mkdir(parents=True)
    selected.mkdir()
    (default_root / "memory.db").write_bytes(b"legacy")
    (selected / "learning.db").write_bytes(b"selected")
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))

    with pytest.raises(DataRootConflictError) as caught:
        assert_no_durable_root_conflict(home=tmp_path / "home")

    message = str(caught.value)
    assert str(default_root.resolve()) in message
    assert str(selected.resolve()) in message
    assert "will not merge or move" in message


@pytest.mark.parametrize("default_marker", [None, "config.json"])
def test_empty_or_config_only_default_root_does_not_block_custom_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    default_marker: str | None,
) -> None:
    from superlocalmemory.infra.data_root import assert_no_durable_root_conflict

    default_root = tmp_path / "home" / ".superlocalmemory"
    selected = tmp_path / "selected"
    default_root.mkdir(parents=True)
    selected.mkdir()
    if default_marker:
        (default_root / default_marker).write_text("{}", encoding="utf-8")
    (selected / "memory.db").write_bytes(b"selected")
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))

    assert_no_durable_root_conflict(home=tmp_path / "home")


def test_unreadable_existing_root_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.infra.data_root import (
        DataRootConflictError,
        assert_no_durable_root_conflict,
    )

    home = tmp_path / "home"
    default_root = (home / ".superlocalmemory").resolve()
    selected = (tmp_path / "selected").resolve()
    default_root.mkdir(parents=True)
    selected.mkdir()
    (selected / "memory.db").write_bytes(b"selected")
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path.resolve() == default_root:
            raise PermissionError("denied for test")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

    with pytest.raises(DataRootConflictError, match="cannot inspect"):
        assert_no_durable_root_conflict(home=home)


def test_compatibility_resolvers_delegate_to_one_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.cli._lazy_init import slm_home
    from superlocalmemory.infra.daemon_identity import canonical_data_root as daemon_root

    selected = tmp_path / "canonical"
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong-legacy"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook"))

    assert slm_home() == selected.resolve(strict=False)
    assert daemon_root() == selected.resolve(strict=False)


def test_critical_runtime_components_resolve_the_same_state_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.cli import daemon, service_installer
    from superlocalmemory.core import embeddings
    from superlocalmemory.core import health_monitor
    from superlocalmemory.core.security_primitives import _install_token_path
    from superlocalmemory.hooks import hook_daemon, session_registry
    from superlocalmemory.retrieval import reranker
    from superlocalmemory.server import bandit_loops

    selected = (tmp_path / "selected").resolve()
    default_root = (tmp_path / "home" / ".superlocalmemory").resolve()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong-legacy"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook"))

    paths = (
        daemon._lock_file_path(),
        daemon._pid_file_path(),
        daemon._port_file_path(),
        service_installer.get_log_path(),
        service_installer.get_error_log_path(),
        embeddings._embedding_lock_file(),
        embeddings._embedding_pid_file(),
        reranker._reranker_pid_file(),
        _install_token_path(),
        hook_daemon._default_sock_path(),
        hook_daemon._default_queue_db_path(),
        session_registry._registry_file(),
        bandit_loops._learning_db(None),
        bandit_loops._memory_db(None),
    )
    assert all(path.is_relative_to(selected) for path in paths)
    assert all(not path.is_relative_to(default_root) for path in paths)

    # Structured logging is exercised, not inferred from source.
    health_monitor.setup_structured_logging()
    try:
        assert (selected / "logs" / "daemon.json.log").exists()
        assert not default_root.exists()
    finally:
        logger = health_monitor._json_logger
        if logger is not None:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        health_monitor._json_logger = None
