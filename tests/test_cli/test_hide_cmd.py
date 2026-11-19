# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm hide``: one command to keep a memory out of answers, and to undo it.

The automatic rule hides machine bookkeeping. On the author's store (9 Nov
2026) 26 vague session lines it cannot recognise ("Session ended with a quick
test") had to be hidden by hand in SQL, because no command did it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from superlocalmemory.cli.hide_cmd import cmd_hide
from superlocalmemory.encoding.non_answer import mark
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

PID = "default"
VAGUE = "Session ended with a quick test of the dashboard."
REAL = "Varun prefers prose over bullet lists in reports."


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"mode": "a"}), encoding="utf-8")
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    return db


def _save(db, content: str) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id=PID, content=content))
    return db.store_fact(AtomicFact(profile_id=PID, memory_id=memory_id, content=content,
                                    fact_type=FactType.SEMANTIC, importance=0.95))


def _flags(db, fact_id: str) -> tuple:
    row = db.execute("SELECT non_answer, non_answer_user FROM atomic_facts WHERE fact_id=?",
                     (fact_id,))[0]
    return row["non_answer"], row["non_answer_user"]


def _hide(*ids: str, undo: bool = False, as_json: bool = False) -> int:
    return cmd_hide(Namespace(fact_ids=list(ids), undo=undo, json=as_json))


def _session_context(capsys) -> str:
    from superlocalmemory.cli.commands import cmd_session_context

    cmd_session_context(Namespace(json=False, full=False, query="", max_age_days=0))
    return capsys.readouterr().out


def test_hide_keeps_a_memory_out_of_session_start_and_undo_brings_it_back(store, capsys):
    vague, real = _save(store, VAGUE), _save(store, REAL)
    assert "quick test" in _session_context(capsys)

    assert _hide(vague) == 0
    assert "Hidden from answers" in capsys.readouterr().out
    assert _flags(store, vague) == (1, 1)
    out = _session_context(capsys)
    assert "quick test" not in out and "prose over bullet" in out

    assert _hide(vague, undo=True) == 0
    assert _flags(store, vague) == (0, 0)
    assert "quick test" in _session_context(capsys)
    assert _flags(store, real) == (0, None)  # untouched


def test_undo_overrules_the_automatic_rule_for_good(store):
    stub = _save(store, "session ended 2026-11-09 13:46")
    mark(store, stub, PID, "session ended 2026-11-09 13:46")
    assert _flags(store, stub) == (1, None)
    assert _hide(stub, undo=True) == 0
    mark(store, stub, PID, "session ended 2026-11-09 13:46")  # a later re-save
    assert _flags(store, stub) == (0, 0)


def test_nothing_is_deleted_or_edited(store):
    vague = _save(store, VAGUE)
    _hide(vague)
    row = store.execute("SELECT content, lifecycle FROM atomic_facts WHERE fact_id=?", (vague,))[0]
    assert row["content"] == VAGUE


def test_unknown_ids_are_reported_and_fail_only_when_nothing_matched(store, capsys):
    vague = _save(store, VAGUE)
    assert _hide("no-such-fact") == 1
    assert _hide(vague, "no-such-fact", as_json=True) == 0
    data = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"]
    assert data["changed"] == [vague] and data["not_found"] == ["no-such-fact"]
    assert _hide(vague) == 0
    assert "Already hidden" in capsys.readouterr().out


def test_another_profiles_fact_is_not_found(store):
    store.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'work')")
    memory_id = store.store_memory(MemoryRecord(profile_id="work", content=VAGUE))
    other = store.store_fact(AtomicFact(profile_id="work", memory_id=memory_id, content=VAGUE,
                                        fact_type=FactType.SEMANTIC))
    assert _hide(other) == 1
    assert _flags(store, other) == (0, None)


def test_help_needs_no_store_or_daemon(tmp_path):
    env = {**os.environ, "SLM_DATA_DIR": str(tmp_path / "home"),
           "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    done = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "hide", "--help"],
                          capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "--undo" in done.stdout
    assert not (tmp_path / "home" / "memory.db").exists()
