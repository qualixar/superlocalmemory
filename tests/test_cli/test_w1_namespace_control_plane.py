"""W1 contract: CLI-owned state follows one canonical data namespace."""

from __future__ import annotations

import json
import sqlite3
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


def _select_root(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "home"
    selected = tmp_path / "selected"
    home.mkdir()
    selected.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong-legacy"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook"))
    return selected.resolve(), home.resolve()


def test_direct_config_uses_call_time_canonical_namespace(monkeypatch, tmp_path):
    from superlocalmemory.core.config import SLMConfig

    selected, _ = _select_root(monkeypatch, tmp_path)
    config = SLMConfig()

    assert config.base_dir == selected
    assert config.db_path == selected / "memory.db"


def test_direct_config_preserves_explicit_runtime_paths(monkeypatch, tmp_path):
    from superlocalmemory.core.config import SLMConfig

    _select_root(monkeypatch, tmp_path)
    explicit_root = tmp_path / "explicit-root"
    explicit_db = tmp_path / "external-databases" / "facts.sqlite"

    config = SLMConfig(base_dir=explicit_root, db_path=explicit_db)
    root_only = SLMConfig(base_dir=explicit_root)

    assert config.base_dir == explicit_root
    assert config.db_path == explicit_db
    assert root_only.base_dir == explicit_root
    assert root_only.db_path == explicit_root / "memory.db"


def test_config_command_reads_and_writes_selected_namespace(monkeypatch, tmp_path):
    from superlocalmemory.cli.commands import cmd_config

    selected, home = _select_root(monkeypatch, tmp_path)
    cmd_config(Namespace(action="set", key="mesh_enabled", value="false", json=False))

    payload = json.loads((selected / "config.json").read_text())
    assert payload["mesh_enabled"] is False
    assert not (home / ".superlocalmemory" / "config.json").exists()


def test_evolve_command_uses_selected_config_and_database(monkeypatch, tmp_path):
    from superlocalmemory.cli.commands import cmd_evolve

    selected, _ = _select_root(monkeypatch, tmp_path)
    (selected / "config.json").write_text(
        json.dumps({"evolution": {"enabled": True}}), encoding="utf-8",
    )
    (selected / "memory.db").touch()
    calls: list[tuple[Path, str, str]] = []

    class FakeEvolver:
        def __init__(self, db_path):
            self.db_path = Path(db_path)

        def run_post_session(self, session_id, profile):
            calls.append((self.db_path, session_id, profile))

    fake_module = types.ModuleType("superlocalmemory.evolution.skill_evolver")
    fake_module.SkillEvolver = FakeEvolver
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    cmd_evolve(Namespace(session="session-1", profile="work"))

    assert calls == [(selected / "memory.db", "session-1", "work")]


def test_cli_learning_signals_follow_config_explicit_root(monkeypatch, tmp_path):
    from superlocalmemory.cli.commands import _cli_record_signals
    from superlocalmemory.learning import feedback, signals

    _select_root(monkeypatch, tmp_path)
    explicit = tmp_path / "explicit-config-root"
    seen: list[Path] = []

    class FakeFeedbackCollector:
        def __init__(self, db_path):
            seen.append(Path(db_path))

        def record_implicit(self, **_kwargs):
            return None

    class FakeLearningSignals:
        def __init__(self, db_path):
            seen.append(Path(db_path))

        def record_co_retrieval(self, *_args):
            return None

        @staticmethod
        def boost_confidence(db_path, _fact_id):
            seen.append(Path(db_path))

    monkeypatch.setattr(feedback, "FeedbackCollector", FakeFeedbackCollector)
    monkeypatch.setattr(signals, "LearningSignals", FakeLearningSignals)
    config = SimpleNamespace(base_dir=explicit, active_profile="default")
    result = SimpleNamespace(fact=SimpleNamespace(fact_id="f1"))

    _cli_record_signals(config, "query", [result])

    assert seen
    assert all(path.is_relative_to(explicit) for path in seen)


def test_context_sync_db_defaults_to_selected_root_but_honors_override(
    monkeypatch, tmp_path,
):
    from superlocalmemory.cli.context_commands import _default_sync_log_db

    selected, _ = _select_root(monkeypatch, tmp_path)
    monkeypatch.delenv("SLM_MEMORY_DB", raising=False)
    assert _default_sync_log_db() == selected / "memory.db"

    external = tmp_path / "external" / "sync.sqlite"
    monkeypatch.setenv("SLM_MEMORY_DB", str(external))
    assert _default_sync_log_db() == external


def test_db_migrate_defaults_to_selected_root_and_preserves_explicit_paths(
    monkeypatch, tmp_path,
):
    from superlocalmemory.cli.db_migrate import _resolve_paths

    selected, _ = _select_root(monkeypatch, tmp_path)
    learning, memory = _resolve_paths(Namespace())
    assert (learning, memory) == (
        selected / "learning.db",
        selected / "memory.db",
    )

    external_learning = tmp_path / "external" / "learning.sqlite"
    external_memory = tmp_path / "external" / "memory.sqlite"
    learning, memory = _resolve_paths(Namespace(
        learning_db_path=external_learning,
        memory_db_path=external_memory,
    ))
    assert (learning, memory) == (external_learning, external_memory)


def test_ingest_writer_targets_selected_memory_db(monkeypatch, tmp_path):
    from superlocalmemory.cli.ingest_cmd import _write_tool_events

    selected, home = _select_root(monkeypatch, tmp_path)
    db_path = selected / "memory.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE tool_events ("
            "session_id TEXT, profile_id TEXT, project_path TEXT, tool_name TEXT, "
            "event_type TEXT, input_summary TEXT, output_summary TEXT, "
            "duration_ms INTEGER, metadata TEXT, created_at TEXT)"
        )

    inserted = _write_tool_events([{
        "session_id": "s1",
        "tool_name": "Read",
        "event_type": "complete",
    }])

    assert inserted == 1
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT tool_name FROM tool_events").fetchone() == ("Read",)
    assert not (home / ".superlocalmemory" / "memory.db").exists()


def test_pending_store_honors_full_alias_precedence(monkeypatch, tmp_path):
    from superlocalmemory.cli import pending_store

    selected, _ = _select_root(monkeypatch, tmp_path)
    row_id = pending_store.store_pending("remember this")

    assert row_id == 1
    assert (selected / "pending.db").exists()


def test_setup_marker_resolves_namespace_at_call_time(monkeypatch, tmp_path):
    from superlocalmemory.cli import setup_wizard

    selected, home = _select_root(monkeypatch, tmp_path)
    setup_wizard._mark_complete()

    assert (selected / ".setup-complete").exists()
    assert setup_wizard.is_setup_complete() is True
    assert not (home / ".superlocalmemory" / ".setup-complete").exists()


def test_noninteractive_first_use_writes_config_to_selected_namespace(
    monkeypatch, tmp_path,
):
    from superlocalmemory.cli import setup_wizard

    selected, home = _select_root(monkeypatch, tmp_path)
    existing = {"mode": "c", "sentinel": "preserve-user-config"}
    (selected / "config.json").write_text(json.dumps(existing), encoding="utf-8")
    monkeypatch.setattr(setup_wizard, "is_interactive", lambda: False)

    setup_wizard.check_first_use("status")

    assert (selected / "config.json").exists()
    assert json.loads((selected / "config.json").read_text()) == existing
    assert (selected / ".setup-complete").exists()
    assert not (home / ".superlocalmemory" / "config.json").exists()


def test_version_banner_honors_legacy_alias_precedence(monkeypatch, tmp_path):
    from superlocalmemory.cli.version_banner import _data_dir

    monkeypatch.delenv("SLM_DATA_DIR", raising=False)
    legacy = tmp_path / "legacy"
    monkeypatch.setenv("SL_MEMORY_PATH", str(legacy))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook"))

    assert _data_dir() == legacy.resolve()


def test_disable_marker_uses_canonical_precedence(monkeypatch, tmp_path):
    from superlocalmemory.core import slm_disabled

    selected, _ = _select_root(monkeypatch, tmp_path)
    marker = slm_disabled.write_marker("maintenance")

    assert marker == selected / ".disabled"
    assert slm_disabled.is_disabled() is True


def test_route_helpers_delegate_to_canonical_resolver(monkeypatch, tmp_path):
    from superlocalmemory.server.routes.helpers import _resolve_slm_home

    selected, _ = _select_root(monkeypatch, tmp_path)
    assert _resolve_slm_home() == selected
