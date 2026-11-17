# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-6b (D-45): every learner's state, so a stalled learner is never silent (F-48)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.learning import learning_health as lh
from superlocalmemory.outcomes import memory_report as mr


def _db(tables: dict[str, str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for name, column in tables.items():
        conn.execute(f"CREATE TABLE {name} (profile_id TEXT, {column} TEXT)")
    return conn


def _put(conn, table, column, at, profile="default", times=1) -> None:
    for _ in range(times):
        conn.execute(f"INSERT INTO {table} (profile_id, {column}) VALUES (?, ?)", (profile, at))


@pytest.fixture
def stores():
    learning = _db({"learning_model_state": "trained_at", "co_retrieval_edges": "updated_at",
                    "_store_patterns": "updated_at", "learning_signals": "created_at",
                    "bandit_plays": "played_at", "agent_experiences": "created_at",
                    "execution_learning_events": "created_at", "workflow_patterns": "created_at",
                    "transferable_patterns": "pattern"})
    memory = _db({"action_outcomes": "timestamp", "injection_usage": "observed_at",
                  "fact_access_log": "accessed_at", "atomic_facts": "created_at"})
    return learning, memory


def _by_name(learning, memory, env=None, profile="default") -> dict[str, lh.Learner]:
    return {x.name: x for x in lh.measure(learning, memory, profile, environ=env or {})}


def test_a_learner_behind_its_input_by_more_than_its_schedule_is_stalled(stores) -> None:
    learning, memory = stores
    _put(learning, "learning_model_state", "trained_at", "2026-04-18T07:50:21+00:00")
    _put(memory, "action_outcomes", "timestamp", "2026-09-12T13:08:02+00:00")
    x = _by_name(learning, memory)["ranker_training"]
    assert x.state == lh.STALLED and "147 days behind" in x.reason
    assert x.last.startswith("2026-04-18T07:50:21")


def test_usage_counts_as_new_feedback_for_the_ranker(stores) -> None:
    learning, memory = stores
    _put(learning, "learning_model_state", "trained_at", "2026-11-01T00:00:00+00:00")
    _put(memory, "injection_usage", "observed_at", "2026-11-10T00:00:00+00:00")
    assert _by_name(learning, memory)["ranker_training"].state == lh.STALLED


def test_within_its_schedule_a_learner_is_learning(stores) -> None:
    learning, memory = stores
    _put(learning, "learning_model_state", "trained_at", "2026-11-10T00:00:00+00:00")
    _put(memory, "action_outcomes", "timestamp", "2026-11-10T20:00:00+00:00")  # 20 h < 24 h
    _put(learning, "co_retrieval_edges", "updated_at", "2026-11-10 12:00:00")
    _put(memory, "fact_access_log", "accessed_at", "2026-11-10 12:30:00")      # 0.5 h < 1 h
    got = _by_name(learning, memory)
    assert got["ranker_training"].state == lh.LEARNING
    assert got["co_retrieval"].state == lh.LEARNING


def test_input_but_no_output_ever_is_stalled_and_no_input_is_said(stores) -> None:
    learning, memory = stores
    _put(memory, "atomic_facts", "created_at", "2026-11-10T00:00:00+00:00")
    got = _by_name(learning, memory)
    assert got["pattern_mining"].state == lh.STALLED and "never" in got["pattern_mining"].reason
    assert got["ranker_training"].state == lh.NO_INPUT


def test_ranking_learners_are_off_unless_the_ensemble_is_set(stores) -> None:
    learning, memory = stores
    _put(learning, "bandit_plays", "played_at", "2026-07-20T04:31:38+00:00", times=3)
    off = _by_name(learning, memory)
    assert {off[n].state for n in ("ranking", "feature_logging", "bandit")} == {lh.OFF}
    assert "SLM_RANKING is not set" in off["bandit"].reason and off["bandit"].rows == 3
    on = _by_name(learning, memory, env={"SLM_RANKING": "v2-ensemble"})
    assert {on[n].state for n in ("ranking", "feature_logging", "bandit")} == {lh.LEARNING}
    v1 = _by_name(learning, memory, env={"SLM_RANKING": "v1"})
    assert v1["ranking"].state == lh.LEARNING and v1["bandit"].state == lh.OFF


def test_agent_fed_learners_wait_and_unwired_ones_are_named(stores) -> None:
    learning, memory = stores
    got = _by_name(learning, memory)
    assert got["agent_experiences"].state == lh.WAITING
    assert "record_agent_experience" in got["agent_experiences"].reason
    assert got["execution_learning"].state == lh.WAITING
    _put(learning, "agent_experiences", "created_at", "2026-11-10T00:00:00+00:00")
    assert _by_name(learning, memory)["agent_experiences"].state == lh.LEARNING
    assert got["cross_project"].state == lh.RETIRED and "profile isolation" in \
        got["cross_project"].reason
    assert got["workflow_mining"].state == lh.NO_INPUT
    _put(memory, "atomic_facts", "created_at", "2026-11-10T00:00:00+00:00")
    assert _by_name(learning, memory)["workflow_mining"].state == lh.STALLED
    _put(learning, "workflow_patterns", "created_at", "2026-11-10T06:00:00+00:00")
    assert _by_name(learning, memory)["workflow_mining"].state == lh.LEARNING


def test_counts_are_per_profile(stores) -> None:
    learning, memory = stores
    _put(learning, "bandit_plays", "played_at", "2026-07-01", profile="slm-video-demo", times=86)
    assert _by_name(learning, memory)["bandit"].rows == 0
    assert _by_name(learning, memory, profile="slm-video-demo")["bandit"].rows == 86


def test_a_store_without_the_tables_says_not_created() -> None:
    got = {x.name: x for x in lh.measure(None, sqlite3.connect(":memory:"), "default",
                                         environ={})}
    assert got["ranker_training"].rows is None and got["ranker_training"].state == lh.NO_INPUT
    assert lh.summary(list(got.values()))[lh.RETIRED] == 1


def test_a_quote_in_a_profile_name_is_data_not_sql(stores) -> None:
    learning, memory = stores
    got = _by_name(learning, memory, profile="o'brien'); DROP TABLE bandit_plays; --")
    assert got["bandit"].rows == 0
    assert learning.execute("SELECT COUNT(*) FROM bandit_plays").fetchone()[0] == 0


def test_the_report_carries_the_learning_section(tmp_path, stores) -> None:
    learning, memory = stores
    path = tmp_path / "learning.db"
    disk = sqlite3.connect(str(path))
    disk.execute("CREATE TABLE learning_model_state (profile_id TEXT, trained_at TEXT)")
    disk.execute("INSERT INTO learning_model_state VALUES ('default', '2026-04-18T07:50:21Z')")
    disk.commit()
    disk.close()
    _put(memory, "action_outcomes", "timestamp", "2026-09-12T13:08:02+00:00")
    report = mr.build(memory, "default", slm_home=tmp_path)
    names = {x["name"]: x for x in report["learning"]["learners"]}
    assert names["ranker_training"]["state"] == lh.STALLED
    text = "\n".join(mr.describe(report))
    assert "Is SLM learning?" in text and "ranker_training: stalled" in text
