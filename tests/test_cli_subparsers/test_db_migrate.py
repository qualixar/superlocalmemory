# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-06 §9.5

"""Tests for the ``slm db migrate`` CLI (LLD-06 §7.2).

Exercises the thin wrapper in
``src/superlocalmemory/cli/db_migrate.py`` which delegates to the
canonical ``apply_all`` / ``status`` runner (LLD-07 §4).
"""
from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_LEARNING_BASELINE = """
CREATE TABLE IF NOT EXISTS learning_signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id    TEXT NOT NULL,
    query         TEXT NOT NULL,
    fact_id       TEXT NOT NULL,
    signal_type   TEXT NOT NULL,
    value         REAL DEFAULT 1.0,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_features (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id     TEXT NOT NULL,
    query_id       TEXT NOT NULL,
    fact_id        TEXT NOT NULL,
    features_json  TEXT NOT NULL,
    label          REAL NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_model_state (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id    TEXT NOT NULL UNIQUE,
    state_bytes   BLOB NOT NULL,
    updated_at    TEXT NOT NULL
);
"""


@pytest.fixture()
def dual_db(tmp_path) -> tuple[Path, Path]:
    """Fresh learning + memory DBs seeded with v3.4.20 baseline schema.

    M001/M002 ALTER the baseline tables — without them the runner
    reports per-migration failures. This mirrors the real install state
    on any 3.4.19 / 3.4.20 machine (see LLD-07 §1.1).
    """
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    conn = sqlite3.connect(learning)
    conn.executescript(_LEARNING_BASELINE)
    conn.commit()
    conn.close()
    sqlite3.connect(memory).close()
    return learning, memory


def _make_args(
    learning: Path, memory: Path,
    *, status: bool = False, dry_run: bool = False,
) -> Namespace:
    return Namespace(
        command="db",
        db_command="migrate",
        status=status,
        dry_run=dry_run,
        learning_db_path=learning,
        memory_db_path=memory,
    )


# ---------------------------------------------------------------------------
# status mode
# ---------------------------------------------------------------------------


def test_status_on_empty_db_prints_missing_for_each(dual_db, capsys):
    from superlocalmemory.cli.db_migrate import cmd_db_migrate
    learning, memory = dual_db
    rc = cmd_db_migrate(_make_args(learning, memory, status=True))
    captured = capsys.readouterr().out
    assert rc == 0
    # Each migration appears once with status 'missing'.
    assert "M001" in captured
    assert "M003" in captured
    assert "missing" in captured


def test_status_reports_complete_after_apply(dual_db, capsys):
    from superlocalmemory.cli.db_migrate import cmd_db_migrate
    learning, memory = dual_db
    rc = cmd_db_migrate(_make_args(learning, memory))
    assert rc == 0
    capsys.readouterr()  # flush
    rc_status = cmd_db_migrate(_make_args(learning, memory, status=True))
    assert rc_status == 0
    report = capsys.readouterr().out
    assert "complete" in report


def test_status_handles_no_registered_migrations(monkeypatch, dual_db,
                                                  capsys):
    """Patch the runner to report an empty dict; the CLI should still
    produce a clean output and exit 0."""
    from superlocalmemory.cli import db_migrate as mod
    monkeypatch.setattr(
        "superlocalmemory.storage.migration_runner.status",
        lambda a, b: {},
    )
    learning, memory = dual_db
    rc = mod.cmd_db_migrate(_make_args(learning, memory, status=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no migrations registered)" in out


# ---------------------------------------------------------------------------
# default (apply) mode
# ---------------------------------------------------------------------------


def test_apply_runs_migrations_prints_summary(dual_db, capsys):
    from superlocalmemory.cli.db_migrate import cmd_db_migrate
    learning, memory = dual_db
    rc = cmd_db_migrate(_make_args(learning, memory))
    captured = capsys.readouterr().out
    assert rc == 0
    assert "Applied=" in captured
    assert "Skipped=" in captured
    assert "Failed=" in captured
    # At least one migration applied on fresh DBs.
    assert "Applied=0" not in captured


def test_apply_is_idempotent(dual_db, capsys):
    """Second run -> zero applied, all skipped, exit 0."""
    from superlocalmemory.cli.db_migrate import cmd_db_migrate
    learning, memory = dual_db
    cmd_db_migrate(_make_args(learning, memory))
    capsys.readouterr()
    rc = cmd_db_migrate(_make_args(learning, memory))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Applied=0" in out
    assert "Failed=0" in out


def test_dry_run_does_not_modify_db(dual_db, capsys):
    """With --dry-run, the runner MUST NOT create any rows.

    Exit code reflects whatever apply_all reports. Contract under test
    here is the 'no writes' guarantee, independent of exit code.
    """
    from superlocalmemory.cli.db_migrate import cmd_db_migrate
    learning, memory = dual_db
    cmd_db_migrate(_make_args(learning, memory, dry_run=True))
    capsys.readouterr()
    # migration_log should NOT exist — dry-run never bootstraps it.
    conn = sqlite3.connect(learning)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='migration_log'"
        ).fetchone()
        if row is not None:
            count = conn.execute(
                "SELECT COUNT(*) FROM migration_log"
            ).fetchone()[0]
            assert count == 0
    finally:
        conn.close()


def test_dry_run_on_bootstrapped_db_is_noop(dual_db, capsys):
    """After a real apply run, a dry-run reports everything skipped."""
    from superlocalmemory.cli.db_migrate import cmd_db_migrate
    learning, memory = dual_db
    cmd_db_migrate(_make_args(learning, memory))  # real apply first
    capsys.readouterr()
    rc = cmd_db_migrate(_make_args(learning, memory, dry_run=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Applied=0" in out
    assert "Failed=0" in out


def test_failed_migration_returns_nonzero_exit(monkeypatch, dual_db,
                                                capsys):
    """When apply_all reports a failed migration, exit code is 1 and
    the FAILED detail line is printed."""
    from superlocalmemory.cli import db_migrate as mod

    def fake_apply(learn, mem, *, dry_run=False):
        return {
            "applied": [], "skipped": [], "failed": ["M999_bad"],
            "details": {"M999_bad": "planted failure"},
        }

    monkeypatch.setattr(
        "superlocalmemory.storage.migration_runner.apply_all",
        fake_apply,
    )
    learning, memory = dual_db
    rc = mod.cmd_db_migrate(_make_args(learning, memory))
    out = capsys.readouterr().out
    assert rc == 1
    assert "Failed=1" in out
    assert "FAILED M999_bad" in out
    assert "planted failure" in out


def test_resolve_paths_defaults(monkeypatch, tmp_path):
    """When args lack paths, module falls back to DEFAULT_HOME."""
    from superlocalmemory.cli import db_migrate as mod
    monkeypatch.setattr(mod, "DEFAULT_HOME", tmp_path / "home")
    args = Namespace(status=True, dry_run=False)
    # The helper is private but testing the fallback is important.
    l, m = mod._resolve_paths(args)
    assert l == tmp_path / "home" / "learning.db"
    assert m == tmp_path / "home" / "memory.db"


def test_resolve_paths_uses_args(tmp_path):
    from superlocalmemory.cli import db_migrate as mod
    args = Namespace(
        learning_db_path=tmp_path / "x.db",
        memory_db_path=tmp_path / "y.db",
    )
    l, m = mod._resolve_paths(args)
    assert l == tmp_path / "x.db"
    assert m == tmp_path / "y.db"


# ---------------------------------------------------------------------------
# Dispatcher wiring
# ---------------------------------------------------------------------------


def test_db_dispatch_routes_migrate(monkeypatch, dual_db, capsys):
    """`_cmd_db_dispatch` with db_command='migrate' must invoke
    cmd_db_migrate."""
    from superlocalmemory.cli import commands as cmd_mod
    called = {}

    def fake_handler(args):
        called["args"] = args
        return 0

    monkeypatch.setattr(
        "superlocalmemory.cli.db_migrate.cmd_db_migrate", fake_handler,
    )
    learning, memory = dual_db
    args = _make_args(learning, memory)
    # Should not raise / not call sys.exit on success.
    cmd_mod._cmd_db_dispatch(args)
    assert called["args"] is args


def test_db_dispatch_unknown_subcommand_exits(capsys):
    from superlocalmemory.cli import commands as cmd_mod
    args = Namespace(db_command=None)
    with pytest.raises(SystemExit) as exc:
        cmd_mod._cmd_db_dispatch(args)
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "slm db migrate" in out


def test_db_dispatch_propagates_failed_exit(monkeypatch, dual_db):
    """When cmd_db_migrate returns 1, dispatcher must sys.exit(1)."""
    from superlocalmemory.cli import commands as cmd_mod
    monkeypatch.setattr(
        "superlocalmemory.cli.db_migrate.cmd_db_migrate",
        lambda a: 1,
    )
    args = _make_args(*dual_db)
    with pytest.raises(SystemExit) as exc:
        cmd_mod._cmd_db_dispatch(args)
    assert exc.value.code == 1
