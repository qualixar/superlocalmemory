# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-6c (D-45): the ranker learns from what agents used of what recall delivered.

Before this the trainer ran only by hand, recall logged nothing to train on
with ranking off, and its only labels were explicit outcome reports (F-48).
Now a pull receipt names the recall it answered, recall logs its exposures in
shadow, every delivered memory is labelled used / not used once its window
closes, the trainer runs on the consolidation timer, and a candidate is served
to nobody until it holds an offline certificate.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from superlocalmemory.learning import exposures, retrain_trigger, training_rows
from superlocalmemory.learning.offline_certificate import certified
from superlocalmemory.learning.signals import SignalBatch, SignalCandidate, record_signal_batch
from superlocalmemory.outcomes import receipts
from superlocalmemory.storage.migrations import M075_injection_receipts as M075

OLD = (datetime.now(UTC) - timedelta(days=3)).isoformat(timespec="microseconds")
NEW = (datetime.now(UTC) - timedelta(hours=2)).isoformat(timespec="microseconds")


# -- the receipt names its recall ------------------------------------------------------------------

def test_m075_carries_the_recall_id_and_repairs_an_older_table() -> None:
    conn = sqlite3.connect(":memory:")
    M075.apply(conn)
    assert "recall_query_id" in {r[1] for r in conn.execute(
        "PRAGMA table_info(injection_deliveries)")}
    assert M075.verify(conn)
    old = sqlite3.connect(":memory:")
    M075.apply(old)
    old.execute("DROP INDEX idx_injection_deliveries_recall_query")
    old.execute("ALTER TABLE injection_deliveries DROP COLUMN recall_query_id")
    assert not M075.verify(old)
    M075.repair(old)
    assert M075.verify(old)


def test_a_pull_receipt_stores_the_recall_id(tmp_path) -> None:
    from tests.test_core.test_coverage_backfill import _db

    store = _db(tmp_path / "memory.db")
    pull = receipts.Pull(tmp_path, store.db_path, "default", "conv-1", "mcp_recall", "q",
                         (("f1", 0.9), ("f2", 0.4)), recall_query_id="rq-42")
    assert receipts.record_pull(pull)
    conn = sqlite3.connect(str(store.db_path))
    assert conn.execute("SELECT recall_query_id FROM injection_deliveries").fetchone() == ("rq-42",)


def test_the_daemon_receipt_takes_the_envelope_query_id(monkeypatch) -> None:
    from superlocalmemory.server import recall_core

    sent = []
    monkeypatch.setattr(receipts, "engine_paths", lambda e: ("/h", "/h/memory.db"))
    monkeypatch.setattr(receipts, "submit_pull", sent.append)
    call = SimpleNamespace(origin="", via="mcp", profile_id="default", session_id="",
                           query="q", agent_id="claude")
    recall_core._receipt(object(), call, {"ok": True, "query_id": "rq-7",
                                          "results": [{"fact_id": "f1", "score": 0.5}]})
    assert sent and sent[0].recall_query_id == "rq-7"


# -- recall logs what it offered, in shadow ---------------------------------------------------------

def _response(n=25):
    results = [SimpleNamespace(fact=SimpleNamespace(fact_id=f"f{i}"), score=1 - i / 100,
                               channel_scores={"semantic": 0.5}) for i in range(n)]
    return SimpleNamespace(results=results, query_type="single_hop")


def test_ranking_off_still_logs_the_top_20_without_reordering(monkeypatch) -> None:
    from superlocalmemory.core import recall_pipeline

    queued = []
    monkeypatch.setattr("superlocalmemory.learning.signals.enqueue", queued.append)
    response = _response()
    out = recall_pipeline.apply_ranking(response, "q", "default", "rq-1",
                                        pipeline_version="off", record_signals=True)
    assert out is response and [r.fact.fact_id for r in out.results][:3] == ["f0", "f1", "f2"]
    (batch,) = queued
    assert batch.query_id == "rq-1" and len(batch.candidates) == 20
    recall_pipeline.apply_ranking(response, "q", "default", "rq-2", pipeline_version="off",
                                  record_signals=False)
    assert len(queued) == 1


def test_measuring_follows_the_receipts_flag_and_the_kill_switch(tmp_path, monkeypatch) -> None:
    config = SimpleNamespace(base_dir=tmp_path)
    monkeypatch.delenv("SLM_FEATURE_OUTCOMES_RECEIPTS", raising=False)
    monkeypatch.delenv("SLM_4230_OFF", raising=False)
    assert exposures.measuring(config)
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_RECEIPTS", "off")
    assert not exposures.measuring(config)
    monkeypatch.delenv("SLM_FEATURE_OUTCOMES_RECEIPTS")
    monkeypatch.setenv("SLM_4230_OFF", "1")
    assert not exposures.measuring(config)
    assert not exposures.measuring(SimpleNamespace(base_dir=None))


def test_an_exposure_without_a_query_id_or_results_is_not_logged(monkeypatch) -> None:
    queued = []
    monkeypatch.setattr("superlocalmemory.learning.signals.enqueue", queued.append)
    assert not exposures.enqueue_exposures(_response(), "q", "default", "")
    assert not exposures.enqueue_exposures(_response(0), "q", "default", "rq")
    assert queued == []


# -- labels from the ledger -------------------------------------------------------------------------

@pytest.fixture
def stores(tmp_path):
    from tests.test_learning._signal_fixtures import make_db_with_migrations

    make_db_with_migrations(tmp_path)
    learning, memory = tmp_path / "learning.db", tmp_path / "memory.db"
    conn = sqlite3.connect(str(memory))
    M075.apply(conn)
    conn.commit()
    conn.close()
    return learning, memory


def _exposed(learning, query_id, facts, profile="default") -> None:
    conn = sqlite3.connect(str(learning))
    record_signal_batch(conn, SignalBatch(
        profile_id=profile, query_id=query_id, query_text="q",
        candidates=tuple(SignalCandidate(fact_id=f, channel_scores={"semantic": 0.5},
                                         cross_encoder_score=None,
                                         result_dict={"fact_id": f, "score": 0.5})
                         for f in facts),
        query_context={"query_type": "single_hop"}))
    conn.close()


def _delivered(memory, did, query_id, facts, *, at=OLD, cut=(), profile="default") -> None:
    conn = sqlite3.connect(str(memory))
    conn.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, path,"
                 " design, holdout_state, epsilon, eligible_count, candidates, delivered_at,"
                 " recall_query_id) VALUES (?, ?, 's', 'mcp_recall', 'one-out-v2', 'off', 0, 0,"
                 " ?, ?, ?)", (did, profile, len(facts), at, query_id))
    for i, f in enumerate(facts):
        conn.execute("INSERT INTO injection_receipts VALUES (?, ?, ?, 'recall', ?, NULL, 0,"
                     " 'path_not_randomised', 1.0, 1, ?)", (did, f, profile, i + 1,
                                                            int(f not in cut)))
    conn.commit()
    conn.close()


def _used(memory, did, fact, *, signal="marker_edit", hours=1) -> None:
    at = (datetime.fromisoformat(OLD) + timedelta(hours=hours)).isoformat()
    conn = sqlite3.connect(str(memory))
    conn.execute("INSERT INTO injection_usage VALUES (?, ?, 'default', ?, NULL, ?)",
                 (did, fact, signal, at))
    conn.commit()
    conn.close()


def _rows(learning, memory, profile="default"):
    from superlocalmemory.learning.database import LearningDatabase

    return {r["fact_id"]: r for r in LearningDatabase(learning).fetch_training_examples(
        profile_id=profile, memory_db=memory)}


def test_every_delivered_memory_is_labelled_used_or_not(stores) -> None:
    learning, memory = stores
    _exposed(learning, "rq-1", ["a", "b", "c"])
    _delivered(memory, "d1", "rq-1", ["a", "b", "c"], cut={"c"})
    _used(memory, "d1", "a")
    _used(memory, "d1", "b", signal="overlap")       # not a use
    rows = _rows(learning, memory)
    assert set(rows) == {"a", "b"}                  # c was cut: no label
    assert (rows["a"]["outcome_reward"], rows["a"]["label_source"]) == (1.0, "ledger_used")
    assert (rows["b"]["outcome_reward"], rows["b"]["label_source"]) == (0.0, "ledger_unused")
    assert rows["a"]["propensity"] == 1.0 and rows["a"]["features"]


def test_a_use_after_the_window_or_an_open_window_is_no_label(stores) -> None:
    learning, memory = stores
    _exposed(learning, "rq-1", ["a"])
    _exposed(learning, "rq-2", ["z"])
    _delivered(memory, "d1", "rq-1", ["a"])
    _used(memory, "d1", "a", hours=30)
    _delivered(memory, "d2", "rq-2", ["z"], at=NEW)
    rows = _rows(learning, memory)
    assert rows["a"]["outcome_reward"] == 0.0 and "z" not in rows


def test_labels_are_per_profile_and_per_recall(stores) -> None:
    learning, memory = stores
    _exposed(learning, "rq-1", ["a"])
    _exposed(learning, "rq-9", ["a"])
    _delivered(memory, "d1", "rq-1", ["a"], profile="work")
    _delivered(memory, "d9", "rq-other", ["a"])
    assert _rows(learning, memory) == {}


def test_an_explicit_outcome_wins_over_the_ledger(stores) -> None:
    learning, memory = stores
    _exposed(learning, "rq-1", ["a", "b"])
    _delivered(memory, "d1", "rq-1", ["a", "b"])
    conn = sqlite3.connect(str(learning))
    conn.execute("CREATE TABLE action_outcomes (outcome_id TEXT, profile_id TEXT, reward REAL,"
                 " recall_query_id TEXT, fact_ids_json TEXT, timestamp TEXT)")
    conn.execute("INSERT INTO action_outcomes VALUES ('o1', 'default', 0.7, 'rq-1', ?, ?)",
                 (json.dumps(["b"]), OLD))
    conn.commit()
    conn.close()
    rows = _rows(learning, memory)
    assert rows["b"]["label_source"] == "outcome" and rows["b"]["outcome_reward"] == 0.7
    assert rows["a"]["label_source"] == "ledger_unused"   # not the neutral 0.5
    assert len([r for r in rows.values() if r["fact_id"] == "a"]) == 1


def test_ledger_source_needs_the_new_column(tmp_path) -> None:
    conn = sqlite3.connect(":memory:")
    assert training_rows.ledger_source(conn, tmp_path / "missing.db") is None
    M075.apply(conn)
    assert training_rows.ledger_source(conn, None) == ""


# -- the trainer runs on the timer, and nothing unproven is served -------------------------------

def test_the_trigger_counts_ledger_labels(stores) -> None:
    _learning, memory = stores
    _delivered(memory, "d1", "rq-1", ["a", "b", "c"])
    _delivered(memory, "d2", "rq-2", ["z"], at=NEW)            # window open
    _delivered(memory, "d3", "", ["y"])                         # names no recall
    assert retrain_trigger.labelled_outcomes_since(memory, "default", None) == 3


def _candidate(learning, shadow) -> None:
    conn = sqlite3.connect(str(learning))
    conn.execute("INSERT INTO learning_model_state (profile_id, model_version, state_bytes,"
                 " bytes_sha256, trained_on_count, feature_names, metrics_json, is_active,"
                 " trained_at, updated_at, is_candidate, shadow_results_json) VALUES"
                 " ('default', 't', X'00', 'h', 1, '[]', '{}', 0, ?, ?, 1, ?)",
                 (OLD, OLD, json.dumps(shadow)))
    conn.commit()
    conn.close()


def test_only_a_certified_candidate_blocks_a_retrain(stores) -> None:
    learning, memory = stores
    _candidate(learning, {"holdout_gate_pass": True})
    assert retrain_trigger.should_retrain(str(learning), str(memory), "default",
                                          new_outcomes_threshold=0, hours_threshold=24)
    conn = sqlite3.connect(str(learning))
    conn.execute("UPDATE learning_model_state SET shadow_results_json = ?",
                 (json.dumps({"offline_certificate": True}),))
    conn.commit()
    conn.close()
    assert not retrain_trigger.should_retrain(str(learning), str(memory), "default",
                                              new_outcomes_threshold=0, hours_threshold=24)


def test_an_uncertified_candidate_is_served_to_nobody(stores, monkeypatch) -> None:
    from superlocalmemory.learning import shadow_serving

    learning, _memory = stores
    _candidate(learning, {"holdout_gate_pass": True})
    active = object()
    routed: list[str] = []

    class Router:
        # serving_model swallows exceptions, so record calls instead of raising.
        def refresh_candidate(self):
            conn = sqlite3.connect(str(learning))
            return conn.execute("SELECT id FROM learning_model_state").fetchone()[0]

        def route_query(self, q):
            routed.append(q)
            return "candidate"

        def assign(self, q, **_):
            routed.append("assign:" + q)

    monkeypatch.setattr("superlocalmemory.core.shadow_router.get_shadow_router",
                        lambda **_: Router())
    assert shadow_serving.serving_model(learning, "default", "rq-1", active) is active
    assert routed == []                              # not routed, no assignment recorded
    conn = sqlite3.connect(str(learning))
    conn.execute("UPDATE learning_model_state SET shadow_results_json = ?",
                 (json.dumps({"offline_certificate": True}),))
    conn.commit()
    conn.close()
    shadow_serving.serving_model(learning, "default", "rq-2", active)
    assert routed and routed[0] == "rq-2"            # certified: it enters the live test


def test_the_certificate_must_be_explicit() -> None:
    assert certified(json.dumps({"offline_certificate": True}))
    assert not certified(json.dumps({"offline_certificate": "yes"}))
    assert not certified("not json") and not certified(None) and not certified("{}")


def test_the_timer_step_trains_only_when_due_and_never_under_pressure(tmp_path, monkeypatch):
    from superlocalmemory.infra import memory_pressure
    from superlocalmemory.learning import lightgbm_subprocess
    from superlocalmemory.server import consolidation_runner

    (tmp_path / "learning.db").touch()
    config = SimpleNamespace(base_dir=tmp_path, db_path=tmp_path / "memory.db")
    ran = []
    monkeypatch.setattr(lightgbm_subprocess, "run_shadow_retrain_isolated",
                        lambda *a, **k: ran.append(a) or {"cycle": {"aborted": None}})
    monkeypatch.setattr(retrain_trigger, "should_retrain", lambda *a, **k: True)
    monkeypatch.setattr(memory_pressure, "pressed", lambda: True)
    assert consolidation_runner.ranker_step("default", config=config)["skipped"] == \
        "memory_pressure"
    monkeypatch.setattr(memory_pressure, "pressed", lambda: False)
    monkeypatch.setattr(retrain_trigger, "should_retrain", lambda *a, **k: False)
    assert consolidation_runner.ranker_step("default", config=config)["skipped"] == "not_due"
    monkeypatch.setattr(retrain_trigger, "should_retrain", lambda *a, **k: True)
    assert consolidation_runner.ranker_step("default", config=config)["cycle"] == {
        "aborted": None}
    assert ran == [(config.db_path, tmp_path / "learning.db", "default")]


def test_the_retrain_runs_in_a_real_subprocess(stores) -> None:
    """The LightGBM-first child really starts, runs only the retrain step and
    answers in JSON; with no labels it refuses honestly."""
    from superlocalmemory.learning.lightgbm_subprocess import run_shadow_retrain_isolated

    learning, memory = stores
    verdict = run_shadow_retrain_isolated(memory, learning, "default", timeout_sec=120)
    assert verdict["error"] is None, verdict
    assert verdict["cycle"]["aborted"] == "insufficient_data"
    assert verdict["cycle"]["candidate_persisted"] is False


# -- a real engine: the exposure, the response and the receipt share one id ----------------------

from tests.test_modes.test_mode_a_never_fails import mode_a_engine  # noqa: E402,F401


def test_a_real_recall_logs_exposures_under_the_id_its_receipt_carries(
        mode_a_engine, monkeypatch) -> None:  # noqa: F811
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    monkeypatch.delenv("SLM_RANKING", raising=False)
    queued = []
    monkeypatch.setattr("superlocalmemory.learning.signals.enqueue", queued.append)
    mode_a_engine.store("The invoice run happens on the last working day of the month.")
    out = EngineRecallAdapter(mode_a_engine).recall("invoice run", 5, "conv-real")
    assert out["results"] and out["query_id"]
    assert receipts.PULL_WRITER.drain()
    conn = sqlite3.connect(str(mode_a_engine.db.db_path))
    (rq,) = conn.execute("SELECT recall_query_id FROM injection_deliveries").fetchone()
    conn.close()
    assert rq == out["query_id"]
    assert [b.query_id for b in queued] == [rq]          # logged in shadow, same id
    shown = {r["fact_id"] for r in out["results"]}
    assert shown <= {c.fact_id for c in queued[0].candidates}


def test_a_full_consolidation_runs_the_retrain_step_and_a_light_one_does_not(monkeypatch):
    import asyncio

    from superlocalmemory.server import consolidation_runner as cr

    monkeypatch.setattr(cr, "_consolidate_blocking", lambda *a: {})
    calls = []
    monkeypatch.setattr(cr, "ranker_step", lambda pid: calls.append(pid) or {"skipped": "x"})
    full = asyncio.run(cr.run_full_consolidation(None, "default", trigger="timer"))
    assert full["ranker"] == {"skipped": "x"} and calls == ["default"]
    light = asyncio.run(cr.run_full_consolidation(None, "default", lightweight=True))
    assert "ranker" not in light and calls == ["default"]
