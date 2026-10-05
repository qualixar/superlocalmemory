# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

""""Classify my memories": the user-triggered, resumable, revertible backfill.

What a user's store must never suffer from it: a confirmed kind overwritten, a
recall slowed or starved, a run that cannot be undone exactly, a crash that
leaves half a batch behind, or memory text sent off the device without a
second, explicit yes. Every test here runs the real runner against a real
SQLite store with the real M052 migration applied; only the model is faked.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core import recall_gate
from superlocalmemory.core.memory_kind_backfill import BackfillRefused, BackfillRunner
from superlocalmemory.core.memory_kind_config import MemoryKindConfig
from superlocalmemory.encoding.memory_kind_classifier import KindClassifier
from superlocalmemory.encoding.memory_kind_recipe import KindAnswer
from superlocalmemory.encoding.memory_kind_rules import suggest_by_rules
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.memory_kind_store import KindCandidate, MemoryKindStore
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord, Mode

TEXTS = (
    "Never push to main without a review.",          # rule cue
    "We decided to use SQLite for the store.",       # decision cue
    "Alice likes green tea.",                         # opinion cue
    "The office is in Pune.",                          # no cue -> legacy
    "Status: the migration is in progress.",          # status cue
    "Draft the cost slide by Friday.",                # prospective cue
    "Yesterday we shipped 4.1.18.",                   # episodic cue
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    from superlocalmemory.storage.migrations import M052_memory_kinds as m052

    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    with mgr.raw_connection() as conn:
        m052.apply(conn)
    mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')")
    return mgr


class _Judge:
    """A live answer-check judge that also answers kind questions."""

    def __init__(self, backend: str = "laya", choice: str = "semantic",
                 confidence: float = 0.6, answer: bool = True) -> None:
        self.backend = backend
        self.ready = True
        self.choice = choice
        self.confidence = confidence
        self.answer = answer
        self.calls: list[list[str]] = []

    def ask_kinds(self, documents, recipe, verify, consent=None):
        self.calls.append(list(documents))
        if not self.answer:
            return None
        return [KindAnswer(self.choice, {self.choice: self.confidence}, self.confidence, None)
                for _ in documents]


def _engine(db: DatabaseManager, *, cfg: MemoryKindConfig | None = None,
            judge: object | None = None, mode: Mode = Mode.A, llm: object | None = None):
    config = SimpleNamespace(mode=mode, memory_kinds=cfg or MemoryKindConfig())
    engine = SimpleNamespace(
        db=db, _db=db, _config=config, _llm=llm, profile_id="default",
        _retrieval_engine=SimpleNamespace(_sufficiency_judge=judge),
    )
    engine._kind_classifier = KindClassifier(
        config=_Live(engine), mode=mode,
        judge_supplier=lambda: engine._retrieval_engine._sufficiency_judge,
        llm_available=llm is not None,
    )
    return engine


class _Live:
    def __init__(self, engine) -> None:
        self._engine = engine

    def __getattr__(self, name):
        return getattr(self._engine._config.memory_kinds, name)


def _cfg(**kw) -> MemoryKindConfig:
    from superlocalmemory.core.memory_kind_config import memory_kind_config_from
    return memory_kind_config_from(kw)


def _fact(db: DatabaseManager, content: str, *, profile_id: str = "default",
          kind: str | None = None, source: str | None = None,
          quarantined: bool = False, scope: str = "personal",
          shared_with: list[str] | None = None) -> str:
    mid = db.store_memory(MemoryRecord(profile_id=profile_id, content="session"))
    f = AtomicFact(profile_id=profile_id, memory_id=mid, content=content,
                   fact_type=FactType.SEMANTIC, scope=scope, shared_with=shared_with)
    f.memory_kind = kind
    f.memory_kind_source = source
    fid = db.store_fact(f)
    if quarantined:
        db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (fid,))
    return fid


def _runner(engine, **kw) -> BackfillRunner:
    kw.setdefault("clock", time.monotonic)
    return BackfillRunner(engine_supplier=lambda: engine, **kw)


def _drain(runner: BackfillRunner, limit: int = 200) -> list[str]:
    actions = []
    for _ in range(limit):
        step = runner.run_once()
        actions.append(step.action)
        if step.action == "idle":
            break
    return actions


def _kinds(db: DatabaseManager, profile_id: str = "default") -> dict[str, tuple]:
    rows = db.execute(
        "SELECT fact_id, memory_kind, memory_kind_source, fact_type FROM atomic_facts "
        "WHERE profile_id = ?", (profile_id,))
    return {r["fact_id"]: (r["memory_kind"], r["memory_kind_source"], r["fact_type"])
            for r in rows}


def _snapshot(db: DatabaseManager) -> list[tuple]:
    """Every column of every fact, for an exact row-level comparison."""
    conn = sqlite3.connect(str(db.db_path))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")]
        rows = conn.execute(
            f"SELECT rowid, {', '.join(cols)} FROM atomic_facts ORDER BY rowid").fetchall()
    finally:
        conn.close()
    return rows


def _history(db: DatabaseManager, run_id: str, origin: str = "backfill") -> int:
    return int(db.execute(
        "SELECT COUNT(*) AS n FROM memory_kind_history WHERE run_id = ? AND origin = ?",
        (run_id, origin))[0]["n"])


# ---------------------------------------------------------------------------
# Runs: create, one per profile, confirmation
# ---------------------------------------------------------------------------


def test_one_active_run_per_profile(db: DatabaseManager) -> None:
    _fact(db, TEXTS[0])
    _fact(db, "work fact", profile_id="work")
    runner = _runner(_engine(db))
    runner.create_run("default", mode="untyped", requested_by="test")
    with pytest.raises(BackfillRefused) as refused:
        runner.create_run("default", mode="untyped", requested_by="test")
    assert refused.value.code == "run_active"
    other = runner.create_run("work", mode="untyped", requested_by="test")
    assert other["status"] == "queued"


def test_jev_needs_device_leaving_confirmation(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    judge = _Judge(backend="jev")
    engine = _engine(db, cfg=_cfg(backend="jev", jev_consent=True), judge=judge)
    runner = _runner(engine)
    with pytest.raises(BackfillRefused) as refused:
        runner.create_run("default", mode="untyped", requested_by="test")
    assert refused.value.code == "needs_confirmation"
    payload = refused.value.payload
    assert payload["needs_confirmation"] is True and payload["leaves_device"] is True
    assert payload["facts"] == len(TEXTS) and payload["requests"] >= 1
    run = runner.create_run("default", mode="untyped", requested_by="test",
                            confirm_data_leaves_device=True)
    assert run["backend"] == "jev"
    assert judge.calls == []   # nothing was sent by asking


def test_mode_c_backfill_stays_on_device_and_needs_no_confirmation(db: DatabaseManager) -> None:
    _fact(db, TEXTS[0])
    engine = _engine(db, mode=Mode.C, llm=object())
    runner = _runner(engine)
    run = runner.create_run("default", mode="untyped", requested_by="test")
    assert run["backend"] == "rules"
    status = runner.status("default")
    assert status["backend"]["active"] == "rules"
    assert "saved" in status["backend"]["reason"]


def test_turned_off_refuses_and_pauses(db: DatabaseManager) -> None:
    _fact(db, TEXTS[0])
    engine = _engine(db)
    runner = _runner(engine)
    run = runner.create_run("default", mode="untyped", requested_by="test")
    engine._config.memory_kinds = _cfg(enabled=False)
    runner.run_once()
    assert runner._store_for(engine).get_run(run["run_id"])["status"] == "paused"
    with pytest.raises(BackfillRefused) as refused:
        runner.create_run("work", mode="untyped", requested_by="test")
    assert refused.value.code == "disabled"


def test_switch_to_jev_mid_run_pauses_without_sending(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    engine = _engine(db, cfg=_cfg(batch_size={"rules": 2}))
    runner = _runner(engine)
    run = runner.create_run("default", mode="untyped", requested_by="test")
    runner.run_once()
    judge = _Judge(backend="jev")
    engine._retrieval_engine._sufficiency_judge = judge
    engine._config.memory_kinds = _cfg(backend="jev", jev_consent=True)
    runner.run_once()
    got = runner._store_for(engine).get_run(run["run_id"])
    assert got["status"] == "paused"
    assert judge.calls == []
    assert "online" in (got["last_error"] or "")


def pre_m052_store(path: Path) -> DatabaseManager:
    """A store as 4.1.18 left it: no kind columns (the fresh schema now has them)."""
    built = DatabaseManager(path)
    built.initialize(real_schema)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("DROP INDEX IF EXISTS idx_facts_memory_kind")
        for col in ("memory_kind", "memory_kind_source", "memory_kind_confidence",
                    "memory_kind_recipe", "memory_kind_at"):
            conn.execute(f"ALTER TABLE atomic_facts DROP COLUMN {col}")
        conn.commit()
    finally:
        conn.close()
    return DatabaseManager(path)


def test_schema_not_ready_refuses(tmp_path: Path) -> None:
    mgr = pre_m052_store(tmp_path / "old.db")
    assert mgr.has_memory_kind_columns() is False
    runner = _runner(_engine(mgr))
    assert runner.run_once().action == "idle"
    assert runner.status("default")["schema_ready"] is False
    with pytest.raises(BackfillRefused) as refused:
        runner.create_run("default", mode="untyped", requested_by="test")
    assert refused.value.code == "schema_not_ready"


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_types_every_untyped_fact_without_changing_fact_type(db: DatabaseManager) -> None:
    ids = [_fact(db, t) for t in TEXTS]
    runner = _runner(_engine(db, cfg=_cfg(batch_size={"rules": 3})))
    run = runner.create_run("default", mode="untyped", requested_by="test")
    _drain(runner)
    kinds = _kinds(db)
    assert all(kinds[i][1] == "rules" for i in ids)
    assert all(kinds[i][2] == "semantic" for i in ids)        # I2: fact_type untouched
    assert kinds[ids[0]][0] == "rule" and kinds[ids[1]][0] == "decision"
    got = runner._store_for(runner._engine()).get_run(run["run_id"])
    assert got["status"] == "completed" and got["finished_at"]


def test_resumes_from_committed_cursor(db: DatabaseManager) -> None:
    ids = [_fact(db, t) for t in TEXTS]
    engine = _engine(db, cfg=_cfg(batch_size={"rules": 3}))
    first = _runner(engine)
    run = first.create_run("default", mode="untyped", requested_by="test")
    assert first.run_once().action == "batch"
    cursor = first._store_for(engine).get_run(run["run_id"])["cursor_rowid"]
    assert cursor > 0
    # A restart: a brand-new runner continues from the committed cursor.
    second = _runner(engine)
    _drain(second)
    rows = db.execute(
        "SELECT fact_id, COUNT(*) AS n FROM memory_kind_history WHERE run_id = ? "
        "GROUP BY fact_id", (run["run_id"],))
    assert {r["fact_id"] for r in rows} == set(ids)
    assert all(r["n"] == 1 for r in rows)          # nothing typed twice


def test_pause_then_resume(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    engine = _engine(db, cfg=_cfg(batch_size={"rules": 2}))
    runner = _runner(engine)
    run = runner.create_run("default", mode="untyped", requested_by="test")
    runner.run_once()
    runner.pause(run["run_id"], requested_by="test")
    before = _kinds(db)
    assert runner.run_once().action == "idle"
    assert _kinds(db) == before
    runner.resume(run["run_id"], requested_by="test")
    _drain(runner)
    store = runner._store_for(engine)
    assert store.get_run(run["run_id"])["status"] == "completed"
    assert all(v[0] is not None for v in _kinds(db).values())


class _CancellingClassifier:
    """Cancels the run while the batch is being classified (no lock held)."""

    def __init__(self, runner_ref: list) -> None:
        self.runner_ref = runner_ref
        self.calls = 0

    def suggest(self, facts, *, caller_kind=None):
        self.calls += 1
        runner, run_id = self.runner_ref
        runner.cancel(run_id, requested_by="test")
        return [suggest_by_rules(f.content, "semantic") for f in facts]


def test_cancel_rolls_back_the_in_flight_batch(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    engine = _engine(db)
    ref: list = []
    classifier = _CancellingClassifier(ref)
    runner = _runner(engine, classifier_supplier=lambda: classifier)
    run = runner.create_run("default", mode="untyped", requested_by="test")
    ref.extend([runner, run["run_id"]])
    runner.run_once()
    assert classifier.calls == 1
    got = runner._store_for(engine).get_run(run["run_id"])
    assert got["status"] == "cancelled"
    assert got["cursor_rowid"] == 0 and got["changed"] == 0
    assert all(v[0] is None for v in _kinds(db).values())
    assert _history(db, run["run_id"]) == 0


class _RecordingClassifier:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def suggest(self, facts, *, caller_kind=None):
        self.seen.extend(f.content for f in facts)
        return [suggest_by_rules(f.content, "semantic") for f in facts]


def test_never_reads_other_profiles_or_rows_shared_in(db: DatabaseManager) -> None:
    _fact(db, "mine")
    shared_in = _fact(db, "owned by work, shared with default", profile_id="work",
                      scope="shared", shared_with=["default"])
    classifier = _RecordingClassifier()
    runner = _runner(_engine(db), classifier_supplier=lambda: classifier)
    runner.create_run("default", mode="untyped", requested_by="test")
    _drain(runner)
    assert classifier.seen == ["mine"]
    assert _kinds(db, "work")[shared_in][0] is None


def test_skips_quarantined_and_tombstoned(db: DatabaseManager) -> None:
    _fact(db, "visible")
    _fact(db, "withheld", quarantined=True)
    gone = _fact(db, "erased")
    db.execute(
        "CREATE TABLE IF NOT EXISTS projection_tombstones (profile_id TEXT NOT NULL, "
        "fact_id TEXT NOT NULL, erasure_id TEXT NOT NULL, memory_id TEXT, "
        "created_at REAL NOT NULL, PRIMARY KEY (profile_id, fact_id))")
    db.execute("INSERT INTO projection_tombstones VALUES ('default', ?, 'e1', NULL, 0)", (gone,))
    classifier = _RecordingClassifier()
    runner = _runner(_engine(db), classifier_supplier=lambda: classifier)
    runner.create_run("default", mode="untyped", requested_by="test")
    _drain(runner)
    assert classifier.seen == ["visible"]


class _LeakyStore(MemoryKindStore):
    """A store whose selection wrongly offers a confirmed row (a bug or a race)."""

    def select_batch(self, profile_id, *, after_rowid, limit, mode):
        rows = self.db.execute(
            "SELECT rowid, fact_id, profile_id, content, fact_type, memory_kind, "
            "memory_kind_source FROM atomic_facts WHERE profile_id = ? AND rowid > ? "
            "ORDER BY rowid LIMIT ?", (profile_id, after_rowid, limit))
        return [KindCandidate(int(r["rowid"]), r["fact_id"], r["profile_id"], r["content"],
                              r["fact_type"], r["memory_kind"], r["memory_kind_source"])
                for r in rows]


def test_never_overwrites_confirmed_kinds(db: DatabaseManager) -> None:
    user = _fact(db, "Never push to main without a review.", kind="opinion", source="user")
    caller = _fact(db, "We decided to use SQLite.", kind="status", source="caller")
    loose = _fact(db, "Alice likes green tea.", kind="semantic", source="rules")
    engine = _engine(db)
    runner = _runner(engine, store=_LeakyStore(db))
    runner.create_run("default", mode="refresh", requested_by="test")
    _drain(runner)
    kinds = _kinds(db)
    assert kinds[user][:2] == ("opinion", "user")
    assert kinds[caller][:2] == ("status", "caller")
    assert kinds[loose][:2] == ("opinion", "rules")   # a suggestion may be refreshed


def test_refresh_mode_with_the_real_store_leaves_confirmed_rows(db: DatabaseManager) -> None:
    user = _fact(db, "Never push to main.", kind="opinion", source="user")
    runner = _runner(_engine(db))
    runner.create_run("default", mode="refresh", requested_by="test")
    _drain(runner)
    assert _kinds(db)[user][:2] == ("opinion", "user")


def test_no_batch_while_a_recall_is_in_flight(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    classifier = _RecordingClassifier()
    runner = _runner(_engine(db), classifier_supplier=lambda: classifier)
    runner.create_run("default", mode="untyped", requested_by="test")
    recall_gate.begin_recall()
    try:
        step = runner.run_once()
        assert step.action == "yield"
        assert classifier.seen == []
        assert all(v[0] is None for v in _kinds(db).values())
    finally:
        recall_gate.end_recall()
    assert runner.run_once().action == "batch"


def test_rate_limit_with_fake_clock(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    now = [100.0]
    engine = _engine(db, cfg=_cfg(batch_size={"rules": 2}, rate_per_second={"rules": 10.0}))
    runner = _runner(engine, clock=lambda: now[0])
    runner.create_run("default", mode="untyped", requested_by="test")
    runner.run_once()               # queued -> running happens inside the first batch
    step = runner.run_once()
    assert step.action == "batch"
    # 2 facts at 10 facts/s -> the next batch may start 0.2 s after this one began.
    assert step.delay == pytest.approx(0.2, abs=1e-6)


class _LockProbeClassifier:
    """Proves the model call runs with no write lock held anywhere."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.free: list[bool] = []

    def suggest(self, facts, *, caller_kind=None):
        probe = sqlite3.connect(str(self.db.db_path), timeout=0)
        try:
            probe.execute("BEGIN IMMEDIATE")
            probe.execute("ROLLBACK")
            sqlite_free = True
        except sqlite3.OperationalError:
            sqlite_free = False
        finally:
            probe.close()
        got: list[bool] = []

        def probe_manager_lock() -> None:
            acquired = self.db._lock.acquire(blocking=False)
            got.append(acquired)
            if acquired:
                self.db._lock.release()   # released by the thread that owns it

        t = threading.Thread(target=probe_manager_lock)
        t.start()
        t.join()
        self.free.append(sqlite_free and bool(got and got[0]))
        return [suggest_by_rules(f.content, "semantic") for f in facts]


def test_model_call_holds_no_write_lock(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    probe = _LockProbeClassifier(db)
    runner = _runner(_engine(db), classifier_supplier=lambda: probe)
    runner.create_run("default", mode="untyped", requested_by="test")
    _drain(runner)
    assert probe.free and all(probe.free)


def test_counts_equal_history_rows(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    _fact(db, "already a suggestion", kind="semantic", source="rules")
    engine = _engine(db, cfg=_cfg(batch_size={"rules": 3}))
    runner = _runner(engine)
    run = runner.create_run("default", mode="refresh", requested_by="test")
    _drain(runner)
    got = runner._store_for(engine).get_run(run["run_id"])
    assert got["changed"] == _history(db, run["run_id"])
    assert got["processed"] == len(TEXTS) + 1
    assert got["changed"] + got["skipped"] == got["processed"]
    assert got["total_estimate"] == len(TEXTS) + 1


def test_model_miss_defers_then_falls_back_to_rules(db: DatabaseManager) -> None:
    for t in TEXTS[:3]:
        _fact(db, t)
    judge = _Judge(backend="laya", answer=False)
    engine = _engine(db, cfg=_cfg(backend="laya"), judge=judge)
    runner = _runner(engine, max_model_misses=2)
    run = runner.create_run("default", mode="untyped", requested_by="test")
    assert run["backend"] == "laya"
    assert runner.run_once().action == "deferred"
    assert all(v[0] is None for v in _kinds(db).values())   # nothing half-typed
    assert runner.run_once().action == "deferred"
    assert runner.run_once().action == "batch"                # bounded: rules now
    assert all(v[1] == "rules" for v in _kinds(db).values())
    got = runner._store_for(engine).get_run(run["run_id"])
    assert "rules" in (got["last_error"] or "")


def test_laya_answers_are_stored_as_model_suggestions(db: DatabaseManager) -> None:
    ids = [_fact(db, t) for t in ("The office is in Pune.", "Never push to main.")]
    judge = _Judge(backend="laya", choice="status", confidence=0.55)
    runner = _runner(_engine(db, cfg=_cfg(backend="laya"), judge=judge))
    runner.create_run("default", mode="untyped", requested_by="test")
    _drain(runner)
    kinds = _kinds(db)
    assert kinds[ids[0]][:2] == ("status", "model:laya")   # no cue: the model decides
    assert kinds[ids[1]][:2] == ("rule", "rules")          # strong cue kept
    assert judge.calls


# ---------------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------------


def test_revert_restores_exact_prior_state(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    _fact(db, "kept as the user said", kind="decision", source="user")
    before = _snapshot(db)
    engine = _engine(db, cfg=_cfg(batch_size={"rules": 3}))
    runner = _runner(engine)
    run = runner.create_run("default", mode="untyped", requested_by="test")
    _drain(runner)
    assert _snapshot(db) != before
    runner.revert(run["run_id"], requested_by="test")
    _drain(runner)
    assert runner._store_for(engine).get_run(run["run_id"])["status"] == "reverted"
    assert _snapshot(db) == before


def test_revert_of_a_refresh_run_restores_exact_prior_state(db: DatabaseManager) -> None:
    # A refresh run re-types memories that already carry a suggestion; undo
    # must give back the earlier recipe and time too, not only the kind.
    for t in TEXTS:
        fid = _fact(db, t)
        db.execute(
            "UPDATE atomic_facts SET memory_kind = 'semantic', memory_kind_source = 'model:laya', "
            "memory_kind_confidence = 0.41, memory_kind_recipe = 'kinds-v0', "
            "memory_kind_at = '2026-09-01T10:00:00+00:00' WHERE fact_id = ?", (fid,))
    before = _snapshot(db)
    judge = _Judge(backend="laya", choice="status", confidence=0.7)
    engine = _engine(db, cfg=_cfg(backend="laya", batch_size={"laya": 3}), judge=judge)
    runner = _runner(engine)
    run = runner.create_run("default", mode="refresh", requested_by="test")
    _drain(runner)
    assert _snapshot(db) != before
    runner.revert(run["run_id"], requested_by="test")
    _drain(runner)
    assert _snapshot(db) == before


def test_revert_whole_run_is_resumable(db: DatabaseManager) -> None:
    ids = [_fact(db, t) for t in TEXTS]
    engine = _engine(db)
    runner = _runner(engine, revert_batch_size=2)
    run = runner.create_run("default", mode="untyped", requested_by="test")
    _drain(runner)
    runner.revert(run["run_id"], requested_by="test")
    assert runner.run_once().action == "revert"
    # Restart mid-revert.
    second = _runner(engine, revert_batch_size=2)
    _drain(second)
    assert second._store_for(engine).get_run(run["run_id"])["status"] == "reverted"
    assert all(_kinds(db)[i][0] is None for i in ids)
    assert _history(db, run["run_id"], "revert") == len(ids)


def test_revert_never_clobbers_a_later_user_edit(db: DatabaseManager) -> None:
    fid = _fact(db, TEXTS[0])
    runner = _runner(_engine(db))
    run = runner.create_run("default", mode="untyped", requested_by="test")
    _drain(runner)
    db.execute("UPDATE atomic_facts SET memory_kind='decision', memory_kind_source='user' "
               "WHERE fact_id = ?", (fid,))
    runner.revert(run["run_id"], requested_by="test")
    _drain(runner)
    assert _kinds(db)[fid][:2] == ("decision", "user")


def test_only_finished_or_stopped_runs_can_be_reverted(db: DatabaseManager) -> None:
    _fact(db, TEXTS[0])
    runner = _runner(_engine(db))
    run = runner.create_run("default", mode="untyped", requested_by="test")
    with pytest.raises(BackfillRefused) as refused:
        runner.revert(run["run_id"], requested_by="test")
    assert refused.value.code == "bad_state"


# ---------------------------------------------------------------------------
# Status, thread lifecycle, bounded waiting
# ---------------------------------------------------------------------------


def test_status_reports_progress(db: DatabaseManager) -> None:
    for t in TEXTS:
        _fact(db, t)
    engine = _engine(db, cfg=_cfg(batch_size={"rules": 3}))
    runner = _runner(engine)
    runner.create_run("default", mode="untyped", requested_by="test")
    runner.run_once()
    status = runner.status("default")
    assert status["schema_ready"] is True and status["enabled"] is True
    active = status["active_run"]
    assert active["status"] == "running" and active["processed"] == 3
    assert active["total_estimate"] == len(TEXTS)
    assert active["eta_seconds"] is not None
    assert status["counts"]["untyped"] + status["counts"]["legacy"] >= 0
    assert status["recent_runs"] and "content" not in str(status["recent_runs"])
    _drain(runner)
    assert runner.status("default")["active_run"] is None


def test_runner_thread_stops_within_5s_and_leaves_no_thread(
    db: DatabaseManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for t in TEXTS:
        _fact(db, t)
    runner = _runner(_engine(db))
    runner.create_run("default", mode="untyped", requested_by="test")
    # Set once the thread has yielded to the recall: it is now waiting, which
    # is the state stop() must be able to end.
    yielded = threading.Event()
    real_run_once = runner.run_once

    def run_once():
        step = real_run_once()
        if step.action == "yield":
            yielded.set()
        return step
    monkeypatch.setattr(runner, "run_once", run_once)
    recall_gate.begin_recall()        # a recall that never ends: the runner just waits
    try:
        runner.start()
        assert yielded.wait(timeout=5.0), "the runner never yielded to the recall"
        assert any(t.name == "slm-kind-backfill" for t in threading.enumerate())
        # stop() itself enforces the 5 s bound: True means the thread joined
        # within it, so no stopwatch is needed on top.
        assert runner.stop(timeout_s=5.0) is True
    finally:
        recall_gate.end_recall()
    assert not any(t.name == "slm-kind-backfill" for t in threading.enumerate())
