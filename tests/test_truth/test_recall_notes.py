# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Recall results carry open disputes and source-file state (D-37..D-39), fail-open."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, env_name
from superlocalmemory.server import recall_core
from superlocalmemory.truth import leases
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _fact, _flags


@pytest.fixture
def store(tmp_path):
    db = _db(tmp_path / "memory.db")
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    src = tmp_path / "config.yaml"
    src.write_text("port: 8765\n")
    leased = _fact(db, "The daemon listens on port 8765", "2026-11-16T00:00:00Z")
    db.execute("UPDATE memories SET metadata_json = ? WHERE memory_id = "
               "(SELECT memory_id FROM atomic_facts WHERE fact_id = ?)",
               (json.dumps({leases.METADATA_KEY: leases.fingerprint(str(src)).as_meta()}), leased))
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    src.write_text("port: 9999 and more\n")
    return db, old, new, leased


def _notes(db, *fact_ids):
    engine = SimpleNamespace(_db=db, profile_id="default")
    call = SimpleNamespace(profile_id="default", caller_view="")
    return recall_core._truth_notes(engine, call, [{"fact_id": f, "content": "x"} for f in fact_ids])


def test_results_carry_the_dispute_and_the_changed_file(store, monkeypatch) -> None:
    db, old, new, leased = store
    monkeypatch.setenv(env_name("truth.slots"), "shadow")
    by_fact = {r["fact_id"]: r for r in _notes(db, new, leased, "plain")}
    assert by_fact[new]["disputed"] is True
    assert {s["fact_id"] for s in by_fact[new]["disputes"][0]["sides"]} == {old, new}
    assert by_fact[leased]["source"]["state"] == "changed"
    assert "disputes" not in by_fact[leased] and "source" not in by_fact[new]
    assert by_fact["plain"] == {"fact_id": "plain", "content": "x"}


def test_nothing_is_added_with_the_flag_off_or_the_kill_switch_on(store, monkeypatch) -> None:
    db, _old, new, leased = store
    monkeypatch.setenv(env_name("truth.slots"), "off")
    assert all(set(r) == {"fact_id", "content"} for r in _notes(db, new, leased))
    monkeypatch.setenv(env_name("truth.slots"), "on")
    monkeypatch.setenv(KILL_ENV, "1")
    assert all(set(r) == {"fact_id", "content"} for r in _notes(db, new, leased))


def test_a_broken_store_leaves_the_results_as_they_were(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(env_name("truth.slots"), "on")
    engine = SimpleNamespace(_db=SimpleNamespace(db_path=tmp_path / "missing.db"), profile_id="p")
    call = SimpleNamespace(profile_id="p", caller_view="")
    results = [{"fact_id": "f", "content": "x"}]
    assert recall_core._truth_notes(engine, call, results) == results
