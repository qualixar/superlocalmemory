# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db repair`` and ``slm db integrity`` through the CLI against a REAL daemon.

The repair runs inside the daemon (the only writer), refuses any data folder
but the one it serves, and remember/recall keep working around it.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import uuid

import pytest

from tests.test_integration.test_erasure_integrity_e2e import (  # noqa: F401 - fixture
    _remember_complete,
    daemon,
    stub_embedder,  # the fixture ``daemon`` uses: it must be visible in this module too
)
from tests.test_integration.test_per_request_profile_e2e import REPO_ROOT, RealDaemon


def _slm(daemon: RealDaemon, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "db", *args],
                          env=daemon.env, cwd=str(REPO_ROOT), capture_output=True, text=True,
                          timeout=600)


def _json(proc: subprocess.CompletedProcess) -> dict:
    return json.loads(proc.stdout)


def _orphaned(daemon: RealDaemon) -> list[str]:
    """Delete two facts with foreign keys off, as the old deduplication did."""
    texts = ("Synthetic pelican roster for the harbour {t}", "Quarry gravel invoice {t} note")
    facts = sorted({_remember_complete(daemon, text.format(t=uuid.uuid4().hex[:6]), "")[0]
                    for text in texts})
    conn = sqlite3.connect(daemon.data_root / "memory.db", timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=OFF")
        for fact_id in facts:
            conn.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact_id,))
        conn.commit()
    finally:
        conn.close()
    return facts


def test_repair_refuses_a_folder_it_was_not_given(daemon: RealDaemon, tmp_path) -> None:
    missing = _slm(daemon, "repair", "--apply", "--json")
    other = _slm(daemon, "repair", "--apply", "--root", str(tmp_path), "--json")

    for proc in (missing, other):
        assert proc.returncode == 2, (proc.stdout, proc.stderr)
        assert _json(proc)["error"]["code"] == "WRONG_ROOT"
        assert "Nothing was changed" in _json(proc)["error"]["message"]
    conn = sqlite3.connect(f"file:{daemon.data_root / 'memory.db'}?mode=ro", uri=True)
    try:
        has_runs = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = "
                                "'integrity_repair_runs'").fetchone()[0]
        runs = conn.execute("SELECT COUNT(*) FROM integrity_repair_runs").fetchone()[0] \
            if has_runs else 0
    finally:
        conn.close()
    assert runs == 0


def test_repair_runs_in_the_daemon_and_undo_restores(daemon: RealDaemon) -> None:
    gone = _orphaned(daemon)
    plan = _json(_slm(daemon, "repair", "--json"))["data"]["plan"]
    before = {o["table"]: o["rows"] for o in plan["orphans"]}
    assert before["bm25_tokens"] >= len(gone)

    applied = _slm(daemon, "repair", "--apply", "--root", str(daemon.data_root),
                   "--batch-size", "1", "--json")
    assert applied.returncode == 0, (applied.stdout, applied.stderr)
    summary = _json(applied)["data"]["summary"]
    assert summary["status"] == "finished"
    assert summary["ran_in"] == "daemon"
    assert summary["done"]["orphans.bm25_tokens"] == before["bm25_tokens"]
    after = {o["table"]: o["rows"] for o in summary["after"]["orphans"]}
    assert after["bm25_tokens"] == 0
    assert after["fact_temporal_validity"] == before["fact_temporal_validity"]  # history kept

    # The daemon kept serving: a new memory is stored and found.
    tag = uuid.uuid4().hex[:6]
    _remember_complete(daemon, f"Probe {tag} synthetic heron note", "")
    assert any(tag in str(r.get("content", "")) for r in
               daemon.recall(f"Probe {tag} heron", "default").get("results", []))

    health = _json(_slm(daemon, "integrity", "--json"))["data"]
    assert list(health) == ["page_integrity", "relational_integrity", "source_fidelity",
                            "projection_readiness", "active_repair"]
    assert health["active_repair"]["last"]["run_id"] == summary["run_id"]
    assert health["relational_integrity"]["orphans"].get("bm25_tokens") is None

    undone = _slm(daemon, "repair", "--undo", summary["run_id"], "--root",
                  str(daemon.data_root), "--json")
    assert undone.returncode == 0, (undone.stdout, undone.stderr)
    assert _json(undone)["data"]["restored"]["bm25_tokens"] == before["bm25_tokens"]


@pytest.mark.skipif(os.name != "posix", reason="path case rules differ")
def test_repair_preview_changes_nothing(daemon: RealDaemon) -> None:
    db = daemon.data_root / "memory.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        before = conn.execute("SELECT COUNT(*) FROM bm25_tokens").fetchone()[0]
    finally:
        conn.close()
    preview = _slm(daemon, "repair", "--json")
    assert preview.returncode == 0 and _json(preview)["data"]["changed"] is False
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM bm25_tokens").fetchone()[0] == before
    finally:
        conn.close()
