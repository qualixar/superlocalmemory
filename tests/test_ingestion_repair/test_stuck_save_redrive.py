# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The upgrade clears #325's stuck saves by itself: nobody runs 700 commands.

Bounded per run, only the retryable error classes, each save once per version, the
running profile only, never under the kill switch, reached from the materializer.
"""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core import stuck_save_redrive as redrive
from tests.test_ingestion_repair.test_stuck_saves_retry import (
    EMB_ERROR,
    PROJ_ERROR,
    _client,
    _dead_letters,
    _embeddings_late,
    _saved_and_complete,
    _state,
    _stuck,
)


@pytest.fixture(autouse=True)
def _fresh_clock(monkeypatch):
    monkeypatch.setattr(redrive, "_last_run", 0.0)


def _stuck_save(engine, client, key, *, error=EMB_ERROR):
    op = _saved_and_complete(engine, client, key)
    state = _embeddings_late(op) if error == EMB_ERROR else dict(op.derivation_state)
    _stuck(engine, op, state=state, error=error)
    return op


def test_a_run_clears_the_stuck_saves(engine_with_mock_deps, tmp_path) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        ops = [_stuck_save(engine, client, f"redrive-{i}") for i in range(3)]

        out = redrive.run(engine, tmp_path / "s.json", version="9.9.9")

        assert out == {"retried": 3, "cleared": 3}
        assert all(_state(engine, op) == "complete" for op in ops)
        assert all(_dead_letters(engine, op) == 0 for op in ops)


def test_other_failures_are_left_for_a_person(engine_with_mock_deps, tmp_path) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _stuck_save(engine, client, "redrive-other", error="worker crashed: out of memory")

        out = redrive.run(engine, tmp_path / "s.json", version="9.9.9")

        assert out == {"retried": 0, "cleared": 0}
        assert _state(engine, op) == "failed"
        assert _dead_letters(engine, op) == 1


def test_a_run_is_bounded(engine_with_mock_deps, tmp_path, monkeypatch) -> None:
    engine = engine_with_mock_deps
    monkeypatch.setattr(redrive, "BATCH", 2)
    with _client(engine) as client:
        ops = [_stuck_save(engine, client, f"redrive-b{i}") for i in range(3)]

        first = redrive.run(engine, tmp_path / "s.json", version="9.9.9")
        assert first["retried"] == 2
        assert [_state(engine, op) for op in ops] == ["complete", "complete", "failed"]  # oldest first
        second = redrive.run(engine, tmp_path / "s.json", version="9.9.9")
        assert second["retried"] == 1
        assert _state(engine, ops[2]) == "complete"


def test_a_save_that_still_fails_is_tried_once_per_version(engine_with_mock_deps,
                                                          tmp_path) -> None:
    engine = engine_with_mock_deps
    state = tmp_path / "s.json"
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "redrive-still")
        engine._db.execute("DELETE FROM atomic_facts WHERE fact_id=?", (op.final_fact_ids[0],))
        _stuck(engine, op, state=dict(op.derivation_state), error=PROJ_ERROR)

        assert redrive.run(engine, state, version="9.9.9") == {"retried": 1, "cleared": 0}
        assert redrive.run(engine, state, version="9.9.9") == {"retried": 0, "cleared": 0}
        assert json.loads(state.read_text())["tried"] == [op.operation_id]
        # the next upgrade tries it again
        assert redrive.run(engine, state, version="9.9.10") == {"retried": 1, "cleared": 0}
        assert _dead_letters(engine, op) == 1


def test_only_the_running_profile(engine_with_mock_deps, tmp_path) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _stuck_save(engine, client, "redrive-profile")
        engine._db.execute("UPDATE dead_letter_operations SET profile_id='work' "
                           "WHERE original_op_id=?", (op.operation_id,))

        assert redrive.run(engine, tmp_path / "s.json", version="9.9.9")["retried"] == 0
        assert _state(engine, op) == "failed"


def test_step_waits_its_interval_and_honours_the_kill_switch(engine_with_mock_deps,
                                                            monkeypatch) -> None:
    engine = engine_with_mock_deps
    calls: list[str] = []
    monkeypatch.setattr(redrive, "run", lambda *a, **k: calls.append("run") or {})
    assert redrive.step(engine, now=1000.0) == {}
    assert redrive.step(engine, now=1000.0 + redrive.INTERVAL_S - 1) is None
    assert redrive.step(engine, now=1000.0 + redrive.INTERVAL_S + 1) == {}
    assert calls == ["run", "run"]

    monkeypatch.setenv("SLM_KILL_SWITCH", "1")
    from superlocalmemory.core import feature_flags
    monkeypatch.setattr(feature_flags, "kill_switch_on", lambda *a, **k: True)
    assert redrive.step(engine, now=5000.0) is None
    assert calls == ["run", "run"]


def test_step_never_raises(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(redrive, "run", boom)
    assert redrive.step(object(), now=1.0) is None


def test_the_materializer_reaches_it_only_when_idle(monkeypatch) -> None:
    from superlocalmemory.daemon import materializer

    seen: list[object] = []
    monkeypatch.setattr(redrive, "step", lambda engine, **_k: seen.append(engine))

    class Repo:
        def __init__(self, ops):
            self.ops = ops

        def list_materializable(self, **_k):
            return self.ops

    class Command:
        def __init__(self, ops):
            self.repository = Repo(ops)

        def materialize(self, _op_id):
            raise RuntimeError("busy")

    class Engine:
        _db = None
        _embedder = None

    engine = Engine()
    monkeypatch.setattr("superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
                        lambda _e: Command([]))
    materializer.ingestion_pass(engine, hooks=materializer.PassHooks(), emit_event=lambda *a, **k: None)
    assert seen == [engine]

    class Op:
        operation_id = "x"

    monkeypatch.setattr("superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
                        lambda _e: Command([Op()]))
    materializer.ingestion_pass(engine, hooks=materializer.PassHooks(), emit_event=lambda *a, **k: None)
    assert seen == [engine], "a pass that had work does not redrive"
