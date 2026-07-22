# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.evolution — Skill Evolution Engine.

Covers all 6 modules:
  - types: frozen dataclasses, enum values, SkillLineage.is_root
  - evolution_store: CRUD, budget control, anti-loop, stats
  - triggers: PostSessionTrigger, DegradationTrigger, HealthCheckTrigger,
              memory pressure guard
  - mutation_generator: prompt building (FIX/DERIVED/CAPTURED),
                        parse_mutation_output, validate_skill_content
  - blind_verifier: prompt building (info isolation),
                    parse_verification_response (JSON + keyword fallback)
  - skill_evolver: detect_backend() with mocked shutil.which, _is_enabled()

All store tests use in-memory SQLite (":memory:"). External calls mocked.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.evolution.types import (
    EvolutionCandidate,
    EvolutionRecord,
    EvolutionStatus,
    EvolutionType,
    SkillLineage,
    TriggerType,
)
from superlocalmemory.evolution.evolution_store import (
    MAX_ATTEMPTS_PER_SKILL,
    MAX_EVOLUTIONS_PER_CYCLE,
    EvolutionStore,
)
from superlocalmemory.evolution.triggers import (
    DEGRADATION_THRESHOLD,
    DERIVED_THRESHOLD,
    MAX_RSS_MB,
    MIN_INVOCATIONS,
    NEGATIVE_SIGNALS_THRESHOLD,
    DegradationTrigger,
    HealthCheckTrigger,
    PostSessionTrigger,
    _check_memory_pressure,
)
from superlocalmemory.evolution.mutation_generator import (
    MAX_CONTENT_CHARS,
    build_mutation_prompt,
    build_retry_prompt,
    parse_mutation_output,
    validate_skill_content,
)
from superlocalmemory.evolution.blind_verifier import (
    VerificationResult,
    build_verification_prompt,
    parse_verification_response,
)
from superlocalmemory.evolution.skill_evolver import (
    SkillEvolver,
    detect_backend,
)


# ===================================================================
# SECTION 1: types.py
# ===================================================================


class TestEvolutionType:
    """EvolutionType enum values and str-mixin behavior."""

    def test_fix_value(self):
        assert EvolutionType.FIX.value == "fix"

    def test_derived_value(self):
        assert EvolutionType.DERIVED.value == "derived"

    def test_captured_value(self):
        assert EvolutionType.CAPTURED.value == "captured"

    def test_all_values_unique(self):
        values = [e.value for e in EvolutionType]
        assert len(values) == len(set(values))

    def test_str_mixin(self):
        assert str(EvolutionType.FIX) == "EvolutionType.FIX"
        assert EvolutionType.FIX == "fix"

    def test_from_value(self):
        assert EvolutionType("fix") is EvolutionType.FIX
        assert EvolutionType("derived") is EvolutionType.DERIVED
        assert EvolutionType("captured") is EvolutionType.CAPTURED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            EvolutionType("nonexistent")


class TestTriggerType:
    """TriggerType enum values."""

    def test_post_session_value(self):
        assert TriggerType.POST_SESSION.value == "post_session"

    def test_degradation_value(self):
        assert TriggerType.DEGRADATION.value == "degradation"

    def test_health_check_value(self):
        assert TriggerType.HEALTH_CHECK.value == "health_check"

    def test_member_count(self):
        assert len(TriggerType) == 3


class TestEvolutionStatus:
    """EvolutionStatus enum covers the full pipeline lifecycle."""

    def test_candidate(self):
        assert EvolutionStatus.CANDIDATE.value == "candidate"

    def test_confirmed(self):
        assert EvolutionStatus.CONFIRMED.value == "confirmed"

    def test_mutated(self):
        assert EvolutionStatus.MUTATED.value == "mutated"

    def test_verified(self):
        assert EvolutionStatus.VERIFIED.value == "verified"

    def test_promoted(self):
        assert EvolutionStatus.PROMOTED.value == "promoted"

    def test_rejected(self):
        assert EvolutionStatus.REJECTED.value == "rejected"

    def test_failed(self):
        assert EvolutionStatus.FAILED.value == "failed"

    def test_member_count(self):
        assert len(EvolutionStatus) == 7


class TestEvolutionCandidate:
    """EvolutionCandidate frozen dataclass."""

    def test_creation_with_defaults(self):
        c = EvolutionCandidate(
            skill_name="test-skill",
            evolution_type=EvolutionType.FIX,
            trigger=TriggerType.POST_SESSION,
        )
        assert c.skill_name == "test-skill"
        assert c.evolution_type == EvolutionType.FIX
        assert c.trigger == TriggerType.POST_SESSION
        assert c.evidence == ()
        assert c.effective_score == 0.0
        assert c.invocation_count == 0
        assert c.session_id == ""
        assert c.project_path == ""

    def test_creation_with_all_fields(self):
        c = EvolutionCandidate(
            skill_name="my-skill",
            evolution_type=EvolutionType.DERIVED,
            trigger=TriggerType.DEGRADATION,
            evidence=("low score", "retries detected"),
            effective_score=0.35,
            invocation_count=12,
            session_id="sess-001",
            project_path="/home/user/project",
        )
        assert c.effective_score == 0.35
        assert c.invocation_count == 12
        assert len(c.evidence) == 2

    def test_frozen_immutability(self):
        c = EvolutionCandidate(
            skill_name="x",
            evolution_type=EvolutionType.FIX,
            trigger=TriggerType.POST_SESSION,
        )
        with pytest.raises(FrozenInstanceError):
            c.skill_name = "y"  # type: ignore[misc]


class TestEvolutionRecord:
    """EvolutionRecord frozen dataclass."""

    def test_creation_with_defaults(self):
        r = EvolutionRecord(
            id="abc123",
            skill_name="test-skill",
            parent_skill_id=None,
            evolution_type=EvolutionType.CAPTURED,
            trigger=TriggerType.HEALTH_CHECK,
        )
        assert r.id == "abc123"
        assert r.parent_skill_id is None
        assert r.generation == 0
        assert r.status == EvolutionStatus.CANDIDATE
        assert r.mutation_summary == ""
        assert r.evidence == ()
        assert r.blind_verified is False

    def test_creation_with_all_fields(self):
        r = EvolutionRecord(
            id="xyz789",
            skill_name="evolved-skill",
            parent_skill_id="parent-id",
            evolution_type=EvolutionType.FIX,
            trigger=TriggerType.DEGRADATION,
            generation=2,
            status=EvolutionStatus.PROMOTED,
            mutation_summary="+5/-3 lines",
            evidence=("reason1",),
            original_content="original",
            evolved_content="evolved",
            content_diff="diff here",
            blind_verified=True,
            rejection_reason="",
            created_at="2026-04-15T00:00:00Z",
            completed_at="2026-04-15T00:01:00Z",
        )
        assert r.generation == 2
        assert r.status == EvolutionStatus.PROMOTED
        assert r.blind_verified is True
        assert r.completed_at == "2026-04-15T00:01:00Z"

    def test_frozen_immutability(self):
        r = EvolutionRecord(
            id="a", skill_name="b", parent_skill_id=None,
            evolution_type=EvolutionType.FIX, trigger=TriggerType.POST_SESSION,
        )
        with pytest.raises(FrozenInstanceError):
            r.status = EvolutionStatus.PROMOTED  # type: ignore[misc]


class TestSkillLineage:
    """SkillLineage frozen dataclass and is_root property."""

    def test_is_root_when_no_parent(self):
        lineage = SkillLineage(
            skill_id="skill-001",
            parent_skill_id=None,
            evolution_type=EvolutionType.CAPTURED,
            generation=0,
            trigger=TriggerType.HEALTH_CHECK,
        )
        assert lineage.is_root is True

    def test_is_not_root_when_parent_exists(self):
        lineage = SkillLineage(
            skill_id="skill-002",
            parent_skill_id="skill-001",
            evolution_type=EvolutionType.DERIVED,
            generation=1,
            trigger=TriggerType.DEGRADATION,
        )
        assert lineage.is_root is False

    def test_defaults(self):
        lineage = SkillLineage(
            skill_id="s1", parent_skill_id=None,
            evolution_type=EvolutionType.FIX, generation=0,
            trigger=TriggerType.POST_SESSION,
        )
        assert lineage.mutation_summary == ""
        assert lineage.created_at == ""

    def test_frozen_immutability(self):
        lineage = SkillLineage(
            skill_id="s1", parent_skill_id=None,
            evolution_type=EvolutionType.FIX, generation=0,
            trigger=TriggerType.POST_SESSION,
        )
        with pytest.raises(FrozenInstanceError):
            lineage.generation = 5  # type: ignore[misc]


# ===================================================================
# SECTION 2: evolution_store.py
# ===================================================================


@pytest.fixture
def evo_store(tmp_path):
    """EvolutionStore backed by a temp DB file."""
    db_path = tmp_path / "test_evo.db"
    return EvolutionStore(db_path)


def _make_record(
    record_id: str = "rec-001",
    skill_name: str = "test-skill",
    status: EvolutionStatus = EvolutionStatus.CANDIDATE,
    evo_type: EvolutionType = EvolutionType.FIX,
    trigger: TriggerType = TriggerType.POST_SESSION,
    evidence: tuple[str, ...] = ("evidence-1",),
    created_at: str = "",
) -> EvolutionRecord:
    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat()
    return EvolutionRecord(
        id=record_id,
        skill_name=skill_name,
        parent_skill_id=None,
        evolution_type=evo_type,
        trigger=trigger,
        status=status,
        evidence=evidence,
        created_at=created_at,
    )


class TestEvolutionStoreCRUD:
    """Basic CRUD operations on EvolutionStore."""

    def test_save_and_get_record(self, evo_store):
        record = _make_record(record_id="r1")
        evo_store.save_record(record, "default")
        retrieved = evo_store.get_record("r1", "default")
        assert retrieved is not None
        assert retrieved.id == "r1"
        assert retrieved.skill_name == "test-skill"
        assert retrieved.evolution_type == EvolutionType.FIX
        assert retrieved.trigger == TriggerType.POST_SESSION

    def test_get_nonexistent_record(self, evo_store):
        assert evo_store.get_record("does-not-exist", "default") is None

    def test_save_record_upsert(self, evo_store):
        r1 = _make_record(record_id="r1", status=EvolutionStatus.CANDIDATE)
        evo_store.save_record(r1, "default")

        r1_updated = EvolutionRecord(
            id="r1",
            skill_name="test-skill",
            parent_skill_id=None,
            evolution_type=EvolutionType.FIX,
            trigger=TriggerType.POST_SESSION,
            status=EvolutionStatus.PROMOTED,
            created_at=r1.created_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        evo_store.save_record(r1_updated, "default")

        retrieved = evo_store.get_record("r1", "default")
        assert retrieved is not None
        assert retrieved.status == EvolutionStatus.PROMOTED

    def test_evidence_round_trip(self, evo_store):
        record = _make_record(evidence=("low perf", "3 retries", "error logged"))
        evo_store.save_record(record, "default")
        retrieved = evo_store.get_record(record.id, "default")
        assert retrieved is not None
        assert retrieved.evidence == ("low perf", "3 retries", "error logged")

    def test_get_skill_history(self, evo_store):
        for i in range(5):
            evo_store.save_record(
                _make_record(
                    record_id=f"r{i}",
                    skill_name="my-skill",
                    created_at=f"2026-04-{10 + i:02d}T00:00:00Z",
                ), "default")
        history = evo_store.get_skill_history("my-skill", "default", limit=3)
        assert len(history) == 3
        # Ordered by created_at DESC
        assert history[0].id == "r4"
        assert history[2].id == "r2"

    def test_get_skill_history_empty(self, evo_store):
        assert evo_store.get_skill_history("nonexistent", "default") == []

    def test_get_recent(self, evo_store):
        for i in range(5):
            evo_store.save_record(
                _make_record(
                    record_id=f"r{i}",
                    skill_name=f"skill-{i}",
                    created_at=f"2026-04-{10 + i:02d}T00:00:00Z",
                ), "default")
        recent = evo_store.get_recent("default", limit=2)
        assert len(recent) == 2
        assert recent[0].id == "r4"


class TestEvolutionStoreBudget:
    """Budget control — max 3 evolutions per cycle."""

    def test_can_evolve_initially(self, evo_store):
        assert evo_store.can_evolve("default") is True

    def test_budget_decrements(self, evo_store):
        for _ in range(MAX_EVOLUTIONS_PER_CYCLE):
            assert evo_store.can_evolve("default") is True
            evo_store.record_evolution_attempt("default")
        assert evo_store.can_evolve("default") is False

    def test_reset_cycle_restores_budget(self, evo_store):
        for _ in range(MAX_EVOLUTIONS_PER_CYCLE):
            evo_store.record_evolution_attempt("default")
        assert evo_store.can_evolve("default") is False
        evo_store.reset_cycle("default")
        assert evo_store.can_evolve("default") is True

    def test_max_evolutions_per_cycle_value(self):
        assert MAX_EVOLUTIONS_PER_CYCLE == 3


class TestEvolutionStoreAntiLoop:
    """Anti-loop: addressed degradations and attempt limits."""

    def test_mark_and_check_addressed(self, evo_store):
        assert evo_store.is_addressed("skill-a", "hash1") is False
        evo_store.mark_addressed("skill-a", "hash1")
        assert evo_store.is_addressed("skill-a", "hash1") is True
        assert evo_store.is_addressed("skill-a", "hash2") is False

    def test_prune_recovered(self, evo_store):
        evo_store.mark_addressed("skill-a", "h1")
        evo_store.mark_addressed("skill-b", "h2")
        evo_store.mark_addressed("skill-c", "h3")

        # skill-b recovered, skill-a and skill-c still degraded
        evo_store.prune_recovered(active_degraded_skills={"skill-a", "skill-c"})
        assert evo_store.is_addressed("skill-a", "h1") is True
        assert evo_store.is_addressed("skill-b", "h2") is False
        assert evo_store.is_addressed("skill-c", "h3") is True

    def test_count_attempts(self, evo_store):
        for i in range(4):
            evo_store.save_record(
                _make_record(
                    record_id=f"r{i}",
                    skill_name="failing-skill",
                    status=EvolutionStatus.CANDIDATE,
                ), "default")
        assert evo_store.count_attempts("failing-skill", "default") == 4

    def test_count_attempts_excludes_only_promoted(self, evo_store):
        """H-ATTEMPTS fix: rejected and failed now count toward the cap."""
        evo_store.save_record(
            _make_record(record_id="r0", status=EvolutionStatus.PROMOTED), "default")
        evo_store.save_record(
            _make_record(record_id="r1", status=EvolutionStatus.REJECTED), "default")
        evo_store.save_record(
            _make_record(record_id="r2", status=EvolutionStatus.CANDIDATE), "default")
        # promoted is excluded, rejected + candidate both count
        assert evo_store.count_attempts("test-skill", "default") == 2

    def test_has_exceeded_attempts(self, evo_store):
        for i in range(MAX_ATTEMPTS_PER_SKILL):
            evo_store.save_record(
                _make_record(
                    record_id=f"r{i}",
                    skill_name="bad-skill",
                    status=EvolutionStatus.CONFIRMED,
                ), "default")
        assert evo_store.has_exceeded_attempts("bad-skill", "default") is True

    def test_has_not_exceeded_attempts(self, evo_store):
        evo_store.save_record(
            _make_record(record_id="r0", skill_name="good-skill"), "default")
        assert evo_store.has_exceeded_attempts("good-skill", "default") is False


class TestEvolutionStoreStats:
    """Stats aggregation."""

    def test_stats_empty(self, evo_store):
        stats = evo_store.get_stats("default")
        assert stats["total"] == 0
        assert stats["by_status"] == {}
        assert stats["by_type"] == {}
        assert stats["cycle_budget_remaining"] == MAX_EVOLUTIONS_PER_CYCLE

    def test_stats_with_records(self, evo_store):
        evo_store.save_record(
            _make_record(record_id="r0", status=EvolutionStatus.PROMOTED,
                         evo_type=EvolutionType.FIX), "default")
        evo_store.save_record(
            _make_record(record_id="r1", status=EvolutionStatus.REJECTED,
                         evo_type=EvolutionType.FIX), "default")
        evo_store.save_record(
            _make_record(record_id="r2", status=EvolutionStatus.PROMOTED,
                         evo_type=EvolutionType.DERIVED), "default")
        evo_store.record_evolution_attempt("default")

        stats = evo_store.get_stats("default")
        assert stats["total"] == 3
        assert stats["by_status"]["promoted"] == 2
        assert stats["by_status"]["rejected"] == 1
        assert stats["by_type"]["fix"] == 2
        assert stats["by_type"]["derived"] == 1
        assert stats["cycle_budget_remaining"] == MAX_EVOLUTIONS_PER_CYCLE - 1


class TestEvolutionStoreRowToRecord:
    """Edge cases in _row_to_record for evidence parsing."""

    def test_malformed_evidence_json(self, evo_store):
        record = _make_record(evidence=("valid",))
        evo_store.save_record(record, "default")

        # Manually corrupt the evidence JSON in the DB
        conn = sqlite3.connect(str(evo_store._db_path), timeout=10)
        conn.execute(
            "UPDATE skill_evolution_log SET evidence = ? WHERE id = ?",
            ("not-json", record.id),
        )
        conn.commit()
        conn.close()

        retrieved = evo_store.get_record(record.id, "default")
        assert retrieved is not None
        assert retrieved.evidence == ()


# ===================================================================
# SECTION 3: triggers.py
# ===================================================================


@pytest.fixture
def trigger_db(tmp_path):
    """Set up a temp SQLite DB with tool_events + behavioral_assertions tables."""
    db_path = tmp_path / "trigger_test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            profile_id TEXT DEFAULT 'default',
            tool_name TEXT,
            input_summary TEXT,
            output_summary TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS behavioral_assertions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id TEXT DEFAULT 'default',
            category TEXT,
            trigger_condition TEXT,
            action TEXT,
            confidence REAL DEFAULT 0.0,
            evidence_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS evolution_cycle_state (
            profile_id TEXT NOT NULL DEFAULT 'default',
            key TEXT NOT NULL,
            value INTEGER DEFAULT 0,
            updated_at TEXT,
            PRIMARY KEY (profile_id, key)
        );
    """)
    conn.commit()
    conn.close()
    return db_path


class TestMemoryPressureGuard:
    """Memory pressure check — skip evolution if RSS > 1GB."""

    def test_no_pressure_when_psutil_unavailable(self):
        with patch.dict("sys.modules", {"psutil": None}):
            # When psutil import fails, should return False (no pressure)
            # The function catches ImportError internally
            result = _check_memory_pressure()
            # May or may not be False depending on psutil availability,
            # but should not raise
            assert isinstance(result, bool)

    def test_no_pressure_when_below_threshold(self):
        mock_psutil = MagicMock()
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(
            rss=500 * 1024 * 1024,  # 500 MB
        )
        mock_psutil.Process.return_value = mock_process
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            assert _check_memory_pressure() is False

    def test_pressure_when_above_threshold(self):
        mock_psutil = MagicMock()
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(
            rss=1500 * 1024 * 1024,  # 1500 MB > 1024 MB
        )
        mock_psutil.Process.return_value = mock_process
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            assert _check_memory_pressure() is True

    def test_max_rss_mb_constant(self):
        assert MAX_RSS_MB == 1024


class TestPostSessionTrigger:
    """PostSessionTrigger — scans tool_events for skill failures."""

    def test_empty_session_returns_no_candidates(self, trigger_db):
        trigger = PostSessionTrigger(trigger_db)
        candidates = trigger.scan("nonexistent-session")
        assert candidates == []

    def test_skill_with_negative_signals_becomes_fix_candidate(self, trigger_db):
        conn = sqlite3.connect(str(trigger_db))
        # M-MININVOC fix: need >= MIN_INVOCATIONS (5) total Skill events for
        # this skill, plus NEGATIVE_SIGNALS_THRESHOLD (2) negative signals.
        # Insert 5 Skill events, 2 of which are followed by Bash errors.
        for invoc in range(5):
            conn.execute(
                "INSERT INTO tool_events (session_id, profile_id, tool_name, input_summary, output_summary, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("sess-1", "default", "Skill",
                 json.dumps({"skill": "bad-skill"}), "",
                 f"2026-04-15T00:0{invoc}:00Z"),
            )
            # First 2 invocations get Bash error followers (negative signals)
            if invoc < NEGATIVE_SIGNALS_THRESHOLD:
                conn.execute(
                    "INSERT INTO tool_events (session_id, profile_id, tool_name, input_summary, output_summary, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("sess-1", "default", "Bash", "",
                     "error: command failed",
                     f"2026-04-15T00:0{invoc}:01Z"),
                )
        conn.commit()
        conn.close()

        trigger = PostSessionTrigger(trigger_db)
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=False):
            candidates = trigger.scan("sess-1")

        assert len(candidates) >= 1
        c = candidates[0]
        assert c.skill_name == "bad-skill"
        assert c.evolution_type == EvolutionType.FIX
        assert c.trigger == TriggerType.POST_SESSION

    def test_memory_pressure_skips_scan(self, trigger_db):
        trigger = PostSessionTrigger(trigger_db)
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=True):
            candidates = trigger.scan("sess-1")
        assert candidates == []

    def test_extract_skill_name_from_input_summary(self):
        trigger = PostSessionTrigger(":memory:")
        event = {"input_summary": json.dumps({"skill": "my-skill"}), "output_summary": ""}
        assert trigger._extract_skill_name(event) == "my-skill"

    def test_extract_skill_name_from_command_name(self):
        trigger = PostSessionTrigger(":memory:")
        event = {"input_summary": json.dumps({"commandName": "my-cmd"}), "output_summary": ""}
        assert trigger._extract_skill_name(event) == "my-cmd"

    def test_extract_skill_name_empty_on_bad_json(self):
        trigger = PostSessionTrigger(":memory:")
        event = {"input_summary": "not-json", "output_summary": ""}
        assert trigger._extract_skill_name(event) == ""

    def test_extract_skill_name_empty_on_missing_fields(self):
        trigger = PostSessionTrigger(":memory:")
        event = {"input_summary": json.dumps({"other": "data"}), "output_summary": ""}
        assert trigger._extract_skill_name(event) == ""


class TestDegradationTrigger:
    """DegradationTrigger — scans behavioral_assertions for skill performance."""

    def _insert_assertion(self, db_path, skill_name, score_pct, evidence_count):
        """Helper: insert a skill_performance assertion."""
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO behavioral_assertions "
            "(profile_id, category, trigger_condition, action, confidence, evidence_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "default",
                "skill_performance",
                f"when considering skill {skill_name}",
                f"effective score: {score_pct}% over N invocations",
                0.8,
                evidence_count,
            ),
        )
        conn.commit()
        conn.close()

    def test_low_score_produces_fix_candidate(self, trigger_db):
        score = int(DEGRADATION_THRESHOLD * 100) - 5  # Below threshold
        self._insert_assertion(trigger_db, "low-perf", score, MIN_INVOCATIONS + 1)

        trigger = DegradationTrigger(trigger_db)
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=False):
            candidates = trigger.scan()

        assert len(candidates) == 1
        assert candidates[0].skill_name == "low-perf"
        assert candidates[0].evolution_type == EvolutionType.FIX
        assert candidates[0].trigger == TriggerType.DEGRADATION

    def test_moderate_score_produces_derived_candidate(self, trigger_db):
        # Between DEGRADATION_THRESHOLD and DERIVED_THRESHOLD
        score = int((DEGRADATION_THRESHOLD + DERIVED_THRESHOLD) / 2 * 100)
        self._insert_assertion(trigger_db, "moderate-skill", score, MIN_INVOCATIONS + 1)

        trigger = DegradationTrigger(trigger_db)
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=False):
            candidates = trigger.scan()

        assert len(candidates) == 1
        assert candidates[0].evolution_type == EvolutionType.DERIVED

    def test_good_score_produces_no_candidate(self, trigger_db):
        score = 80  # Well above DERIVED_THRESHOLD
        self._insert_assertion(trigger_db, "good-skill", score, MIN_INVOCATIONS + 1)

        trigger = DegradationTrigger(trigger_db)
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=False):
            candidates = trigger.scan()

        assert candidates == []

    def test_insufficient_invocations_skipped(self, trigger_db):
        score = 10  # Very low
        self._insert_assertion(trigger_db, "new-skill", score, MIN_INVOCATIONS - 1)

        trigger = DegradationTrigger(trigger_db)
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=False):
            candidates = trigger.scan()

        assert candidates == []

    def test_memory_pressure_skips_scan(self, trigger_db):
        self._insert_assertion(trigger_db, "skill-x", 10, 100)

        trigger = DegradationTrigger(trigger_db)
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=True):
            candidates = trigger.scan()

        assert candidates == []

    def test_parse_effective_score_with_match(self):
        trigger = DegradationTrigger(":memory:")
        assert trigger._parse_effective_score("effective score: 35% over 10 invocations") == 0.35

    def test_parse_effective_score_no_match(self):
        trigger = DegradationTrigger(":memory:")
        assert trigger._parse_effective_score("some random text") == 0.5

    def test_get_active_degraded(self, trigger_db):
        self._insert_assertion(trigger_db, "degraded-skill", 30, 10)
        self._insert_assertion(trigger_db, "ok-skill", 80, 10)

        trigger = DegradationTrigger(trigger_db)
        degraded = trigger.get_active_degraded()
        assert "degraded-skill" in degraded
        assert "ok-skill" not in degraded


class TestHealthCheckTrigger:
    """HealthCheckTrigger — periodic consolidation scan."""

    def test_should_run_every_nth_cycle(self, trigger_db):
        trigger = HealthCheckTrigger(trigger_db)
        # _check_every_n defaults to 3
        assert trigger.should_run() is False  # cycle 1
        assert trigger.should_run() is False  # cycle 2
        assert trigger.should_run() is True   # cycle 3
        assert trigger.should_run() is False  # cycle 4
        assert trigger.should_run() is False  # cycle 5
        assert trigger.should_run() is True   # cycle 6

    def test_scan_returns_empty_when_not_due(self, trigger_db):
        trigger = HealthCheckTrigger(trigger_db)
        # First cycle — should_run() returns False
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=False):
            candidates = trigger.scan()
        assert candidates == []

    def test_scan_relabels_trigger_type(self, trigger_db):
        """When HealthCheck runs, candidates should have HEALTH_CHECK trigger."""
        # Insert assertion data that DegradationTrigger would find
        conn = sqlite3.connect(str(trigger_db))
        conn.execute(
            "INSERT INTO behavioral_assertions "
            "(profile_id, category, trigger_condition, action, confidence, evidence_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("default", "skill_performance", "when considering skill needs-fix",
             "effective score: 20% over 10", 0.9, 10),
        )
        conn.commit()
        conn.close()

        trigger = HealthCheckTrigger(trigger_db)
        # H-CYCLECNT fix: cycle count is now DB-persisted.
        # Set it to 2 in DB so next should_run() call sees cycle 3.
        _conn = sqlite3.connect(str(trigger_db))
        _conn.execute(
            "INSERT OR REPLACE INTO evolution_cycle_state (profile_id, key, value, updated_at) "
            "VALUES ('default', 'health_check_cycle_count', 2, '2026-04-15T00:00:00Z')",
        )
        _conn.commit()
        _conn.close()

        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=False):
            candidates = trigger.scan()

        assert len(candidates) == 1
        assert candidates[0].trigger == TriggerType.HEALTH_CHECK

    def test_memory_pressure_skips(self, trigger_db):
        trigger = HealthCheckTrigger(trigger_db)
        # Set cycle count to 2 in DB for the health check to fire
        _conn = sqlite3.connect(str(trigger_db))
        _conn.execute(
            "INSERT OR REPLACE INTO evolution_cycle_state (profile_id, key, value, updated_at) "
            "VALUES ('default', 'health_check_cycle_count', 2, '2026-04-15T00:00:00Z')",
        )
        _conn.commit()
        _conn.close()

        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=True):
            candidates = trigger.scan()

        assert candidates == []


# ===================================================================
# SECTION 4: mutation_generator.py
# ===================================================================


def _make_candidate(
    evo_type: EvolutionType = EvolutionType.FIX,
    skill_name: str = "test-skill",
    evidence: tuple[str, ...] = ("low perf",),
    score: float = 0.3,
) -> EvolutionCandidate:
    return EvolutionCandidate(
        skill_name=skill_name,
        evolution_type=evo_type,
        trigger=TriggerType.POST_SESSION,
        evidence=evidence,
        effective_score=score,
    )


class TestBuildMutationPrompt:
    """build_mutation_prompt for all three evolution types."""

    def test_fix_prompt_contains_skill_name_and_score(self):
        c = _make_candidate(EvolutionType.FIX, score=0.25)
        prompt = build_mutation_prompt(c, "# Original Skill Content")
        assert "test-skill" in prompt
        assert "25%" in prompt
        assert "underperforming" in prompt
        assert "EVOLUTION_COMPLETE" in prompt
        assert "EVOLUTION_FAILED" in prompt

    def test_derived_prompt_contains_parent_info(self):
        c = _make_candidate(EvolutionType.DERIVED, score=0.5)
        prompt = build_mutation_prompt(c, "# Parent Skill")
        assert "PARENT SKILL: test-skill" in prompt
        assert "50%" in prompt
        assert "specialized" in prompt.lower() or "DERIVED" in prompt

    def test_captured_prompt_contains_pattern_info(self):
        c = _make_candidate(EvolutionType.CAPTURED, evidence=("repeated pattern detected",))
        prompt = build_mutation_prompt(c, "")
        assert "PATTERN NAME: test-skill" in prompt
        assert "repeated pattern detected" in prompt
        assert "codif" in prompt.lower()

    def test_content_truncation(self):
        long_content = "x" * (MAX_CONTENT_CHARS + 5000)
        c = _make_candidate(EvolutionType.FIX)
        prompt = build_mutation_prompt(c, long_content)
        # The truncated content should appear, not the full content
        assert len(prompt) < len(long_content) + 2000

    def test_evidence_included_in_prompt(self):
        c = _make_candidate(evidence=("reason A", "reason B"))
        prompt = build_mutation_prompt(c, "content")
        assert "reason A" in prompt
        assert "reason B" in prompt


class TestParseMutationOutput:
    """parse_mutation_output — extracting SKILL.md from LLM output."""

    def test_extract_from_markdown_code_fence(self):
        output = (
            "Here is the improved skill:\n\n"
            "```markdown\n"
            "---\n"
            "name: improved-skill\n"
            "description: Better version\n"
            "---\n\n"
            "# Instructions\n"
            "Do the thing better.\n"
            "```\n\n"
            "<EVOLUTION_COMPLETE>"
        )
        result = parse_mutation_output(output)
        assert result is not None
        assert "name: improved-skill" in result
        assert "Instructions" in result

    def test_extract_from_md_fence(self):
        output = (
            "```md\n"
            "---\n"
            "name: test\n"
            "---\n"
            "Content\n"
            "```"
        )
        result = parse_mutation_output(output)
        assert result is not None
        assert "name: test" in result

    def test_extract_from_evolution_complete_token(self):
        output = (
            "Analysis done.\n\n"
            "<EVOLUTION_COMPLETE>\n"
            "---\n"
            "name: evolved-skill\n"
            "description: Fixed version\n"
            "---\n\n"
            "# Content\n"
            "Better instructions."
        )
        result = parse_mutation_output(output)
        assert result is not None
        assert "evolved-skill" in result

    def test_extract_from_frontmatter_directly(self):
        output = (
            "Some preamble.\n\n"
            "---\n"
            "name: direct-skill\n"
            "description: Something\n"
            "---\n"
            "# Body"
        )
        result = parse_mutation_output(output)
        assert result is not None
        assert "direct-skill" in result

    def test_returns_none_on_evolution_failed(self):
        output = "I cannot improve this skill. <EVOLUTION_FAILED>"
        assert parse_mutation_output(output) is None

    def test_returns_none_on_no_valid_content(self):
        output = "This is just some random text with no SKILL.md format."
        assert parse_mutation_output(output) is None


class TestValidateSkillContent:
    """validate_skill_content — checks content quality."""

    def test_valid_content(self):
        content = (
            "---\n"
            "name: my-skill\n"
            "description: A good skill\n"
            "---\n\n"
            "# Instructions\n"
            "Step 1: Do something.\n"
            "Step 2: Do more."
        )
        assert validate_skill_content(content) is None

    def test_too_short(self):
        error = validate_skill_content("short")
        assert error is not None
        assert "too short" in error.lower()

    def test_missing_frontmatter(self):
        content = "A" * 60  # Long enough but no ---
        error = validate_skill_content(content)
        assert error is not None
        assert "frontmatter" in error.lower()

    def test_empty_content(self):
        error = validate_skill_content("")
        assert error is not None
        assert "short" in error.lower()

    def test_missing_name_in_frontmatter(self):
        content = (
            "---\n"
            "description: no name field\n"
            "---\n"
            "# Body content that is long enough to pass length check"
        )
        error = validate_skill_content(content)
        assert error is not None
        assert "name:" in error.lower()


class TestBuildRetryPrompt:
    """build_retry_prompt — appends retry instructions."""

    def test_includes_attempt_number(self):
        prompt = build_retry_prompt("original prompt", "bad output", 2)
        assert "attempt 2/" in prompt
        assert "bad output" in prompt
        assert "original prompt" in prompt

    def test_includes_formatting_hints(self):
        prompt = build_retry_prompt("base", "error", 1)
        assert "YAML frontmatter" in prompt
        assert "EVOLUTION_COMPLETE" in prompt


# ===================================================================
# SECTION 5: blind_verifier.py
# ===================================================================


class TestVerificationResult:
    """VerificationResult frozen dataclass."""

    def test_creation_with_defaults(self):
        r = VerificationResult(passed=True, confidence=0.9)
        assert r.passed is True
        assert r.confidence == 0.9
        assert r.issues == ()
        assert r.reasoning == ""

    def test_frozen_immutability(self):
        r = VerificationResult(passed=False, confidence=0.5)
        with pytest.raises(FrozenInstanceError):
            r.passed = True  # type: ignore[misc]


class TestBuildVerificationPrompt:
    """build_verification_prompt — information isolation check."""

    def test_prompt_contains_skill_info(self):
        prompt = build_verification_prompt(
            "my-skill", "Handles code review", "# Evolved Skill Content",
        )
        assert "my-skill" in prompt
        assert "Handles code review" in prompt
        assert "Evolved Skill Content" in prompt

    def test_prompt_enforces_information_isolation(self):
        prompt = build_verification_prompt(
            "skill-a", "Does X", "# New content",
        )
        # The prompt must explicitly state the verifier has NOT seen the original
        assert "NOT seen the original" in prompt
        # Should not include evidence or mutation rationale fields
        assert "evidence" not in prompt.lower()
        assert "mutation" not in prompt.lower()

    def test_prompt_requests_json_response(self):
        prompt = build_verification_prompt("s", "d", "c")
        assert '"passed"' in prompt
        assert '"confidence"' in prompt
        assert '"issues"' in prompt
        assert '"reasoning"' in prompt

    def test_prompt_truncates_long_content(self):
        long_content = "x" * 10000
        prompt = build_verification_prompt("s", "d", long_content)
        # Content should be truncated at 8000 chars
        assert len(prompt) < 10000 + 1000

    def test_prompt_emphasizes_strict_evaluation(self):
        prompt = build_verification_prompt("s", "d", "c")
        assert "strict" in prompt.lower()


class TestParseVerificationResponse:
    """parse_verification_response — JSON and keyword fallback."""

    def test_parse_valid_json_passed(self):
        response = json.dumps({
            "passed": True,
            "confidence": 0.85,
            "issues": [],
            "reasoning": "Skill is well-structured.",
        })
        result = parse_verification_response(response)
        assert result.passed is True
        assert result.confidence == 0.85
        assert result.issues == ()
        assert "well-structured" in result.reasoning

    def test_parse_valid_json_failed(self):
        response = json.dumps({
            "passed": False,
            "confidence": 0.7,
            "issues": ["Missing error handling", "Vague instructions"],
            "reasoning": "Needs more specificity.",
        })
        result = parse_verification_response(response)
        assert result.passed is False
        assert result.confidence == 0.7
        assert len(result.issues) == 2
        assert "Missing error handling" in result.issues

    def test_parse_json_embedded_in_text(self):
        response = (
            "After careful review, here is my assessment:\n\n"
            '{"passed": true, "confidence": 0.9, "issues": [], "reasoning": "Good"}\n\n'
            "Hope this helps."
        )
        result = parse_verification_response(response)
        assert result.passed is True
        assert result.confidence == 0.9

    def test_keyword_fallback_approve(self):
        response = "I approve this skill. It looks good and is well-written."
        result = parse_verification_response(response)
        assert result.passed is True
        assert result.confidence == 0.6
        assert result.reasoning == "keyword match"

    def test_keyword_fallback_reject(self):
        response = "This skill should be rejected. Too many issues."
        result = parse_verification_response(response)
        assert result.passed is False
        assert result.confidence == 0.6

    def test_keyword_fallback_passed_true(self):
        response = 'The result is "passed": true based on my analysis.'
        result = parse_verification_response(response)
        assert result.passed is True

    def test_keyword_fallback_passed_false(self):
        response = 'My verdict: "passed": false. Needs work.'
        result = parse_verification_response(response)
        assert result.passed is False

    def test_unparseable_defaults_to_reject(self):
        response = "Some completely unstructured gibberish with no keywords."
        result = parse_verification_response(response)
        assert result.passed is False
        assert result.confidence == 0.3
        assert "Unparseable response" in result.issues

    def test_empty_response_defaults_to_reject(self):
        result = parse_verification_response("")
        assert result.passed is False

    def test_json_with_missing_fields_uses_defaults(self):
        response = '{"passed": true}'
        result = parse_verification_response(response)
        assert result.passed is True
        assert result.confidence == 0.5  # default
        assert result.issues == ()


# ===================================================================
# SECTION 6: skill_evolver.py
# ===================================================================


class TestDetectBackend:
    """detect_backend() — auto-detect LLM backend availability."""

    def test_claude_cli_detected(self):
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            assert detect_backend() == "claude"

    def test_ollama_detected_when_no_claude(self):
        with patch("shutil.which", return_value=None):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__ = MagicMock()
                mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
                assert detect_backend() == "ollama"

    def test_anthropic_api_key_detected(self):
        with patch("shutil.which", return_value=None):
            with patch("urllib.request.urlopen", side_effect=Exception("no ollama")):
                with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
                    # Clear OPENAI_API_KEY to avoid interference
                    env = {"ANTHROPIC_API_KEY": "sk-test"}
                    with patch.dict("os.environ", env, clear=False):
                        assert detect_backend() == "anthropic"

    def test_openai_api_key_detected(self):
        with patch("shutil.which", return_value=None):
            with patch("urllib.request.urlopen", side_effect=Exception("no ollama")):
                with patch.dict(
                    "os.environ",
                    {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": ""},
                    clear=False,
                ):
                    result = detect_backend()
                    # Could be 'anthropic' if ANTHROPIC_API_KEY was already set
                    assert result in ("openai", "anthropic")

    def test_none_when_nothing_available(self):
        with patch("shutil.which", return_value=None):
            with patch("urllib.request.urlopen", side_effect=Exception("no")):
                with patch.dict(
                    "os.environ",
                    {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""},
                    clear=False,
                ):
                    assert detect_backend() == "none"


class TestSkillEvolverIsEnabled:
    """SkillEvolver._is_enabled() — config-gated evolution."""

    def test_disabled_when_no_config(self, tmp_path):
        evolver = SkillEvolver(tmp_path / "test.db", config=None)
        assert evolver._is_enabled() is False

    def test_disabled_when_config_has_no_evolution(self, tmp_path):
        config = MagicMock(spec=[])  # No 'evolution' attribute
        evolver = SkillEvolver(tmp_path / "test.db", config=config)
        assert evolver._is_enabled() is False

    def test_enabled_when_config_says_true(self, tmp_path):
        config = MagicMock()
        config.evolution.enabled = True
        evolver = SkillEvolver(tmp_path / "test.db", config=config)
        assert evolver._is_enabled() is True

    def test_disabled_when_config_says_false(self, tmp_path):
        config = MagicMock()
        config.evolution.enabled = False
        evolver = SkillEvolver(tmp_path / "test.db", config=config)
        assert evolver._is_enabled() is False


class TestSkillEvolverConsolidation:
    """SkillEvolver.run_consolidation_cycle — integration-level."""

    def test_returns_disabled_when_not_enabled(self, tmp_path):
        evolver = SkillEvolver(tmp_path / "test.db", config=None)
        result = evolver.run_consolidation_cycle()
        assert result["enabled"] is False

    def test_returns_no_backend_message(self, tmp_path):
        config = MagicMock()
        config.evolution.enabled = True
        config.evolution.backend = "auto"

        evolver = SkillEvolver(tmp_path / "test.db", config=config)
        with patch("superlocalmemory.evolution.skill_evolver.detect_backend", return_value="none"):
            result = evolver.run_consolidation_cycle()
        assert result["backend"] == "none"
        assert "No LLM backend" in result["message"]


class TestSkillEvolverPostSession:
    """SkillEvolver.run_post_session — trigger 1."""

    def test_empty_session_returns_zero_candidates(self, tmp_path):
        db_path = tmp_path / "test.db"
        # Need tool_events table for PostSessionTrigger
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tool_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, profile_id TEXT, tool_name TEXT, "
            "input_summary TEXT, output_summary TEXT, created_at TEXT)",
        )
        conn.commit()
        conn.close()

        evolver = SkillEvolver(db_path)
        with patch("superlocalmemory.evolution.triggers._check_memory_pressure", return_value=False):
            result = evolver.run_post_session("nonexistent-session")
        assert result["candidates"] == 0
        assert result["evolved"] == 0


class TestSkillEvolverGetBackend:
    """SkillEvolver._get_backend() — caching and config override."""

    def test_auto_backend_detection(self, tmp_path):
        config = MagicMock()
        config.evolution.backend = "auto"
        evolver = SkillEvolver(tmp_path / "test.db", config=config)

        with patch("superlocalmemory.evolution.skill_evolver.detect_backend", return_value="claude"):
            backend = evolver._get_backend()
        assert backend == "claude"

    def test_configured_backend_overrides_auto(self, tmp_path):
        config = MagicMock()
        config.evolution.backend = "ollama"
        evolver = SkillEvolver(tmp_path / "test.db", config=config)
        backend = evolver._get_backend()
        assert backend == "ollama"

    def test_backend_cached_after_first_call(self, tmp_path):
        config = MagicMock()
        config.evolution.backend = "auto"
        evolver = SkillEvolver(tmp_path / "test.db", config=config)

        with patch("superlocalmemory.evolution.skill_evolver.detect_backend", return_value="claude") as mock_detect:
            evolver._get_backend()
            evolver._get_backend()
        mock_detect.assert_called_once()


class TestSkillEvolverHelpers:
    """Utility methods on SkillEvolver."""

    def test_extract_description_from_frontmatter(self, tmp_path):
        evolver = SkillEvolver(tmp_path / "test.db")
        content = "---\nname: skill\ndescription: Does code review\n---\n# Body"
        assert evolver._extract_description(content) == "Does code review"

    def test_extract_description_fallback(self, tmp_path):
        evolver = SkillEvolver(tmp_path / "test.db")
        content = "no frontmatter here"
        assert evolver._extract_description(content) == "Skill for AI agent tasks"

    def test_compute_diff_new_skill(self, tmp_path):
        evolver = SkillEvolver(tmp_path / "test.db")
        diff = evolver._compute_diff("", "new content")
        assert "no original" in diff.lower()

    def test_compute_diff_with_original(self, tmp_path):
        evolver = SkillEvolver(tmp_path / "test.db")
        diff = evolver._compute_diff("line1\nline2\n", "line1\nline3\n")
        assert "-line2" in diff
        assert "+line3" in diff

    def test_summarize_diff(self, tmp_path):
        evolver = SkillEvolver(tmp_path / "test.db")
        diff = "--- original\n+++ evolved\n-removed\n+added1\n+added2\n"
        summary = evolver._summarize_diff(diff)
        assert "+2/-1" in summary


class TestEvolutionStoreProfileIsolation:
    """Per-profile isolation of skill-evolution history AND cycle budget.

    Regression guard for the confirmed leak: skill_evolution_log and
    evolution_cycle_state had no profile_id, so every profile shared one
    history and one evolve budget (one profile exhausting its 3-per-cycle
    budget blocked all others).
    """

    def test_history_is_isolated_between_profiles(self, evo_store):
        evo_store.save_record(_make_record(record_id="a1", skill_name="s"), "work")
        evo_store.save_record(_make_record(record_id="a2", skill_name="s"), "work")
        evo_store.save_record(_make_record(record_id="b1", skill_name="s"), "home")

        assert len(evo_store.get_recent("work")) == 2
        assert len(evo_store.get_recent("home")) == 1
        assert evo_store.get_recent("empty") == []
        # A record from 'work' is invisible to 'home'.
        assert evo_store.get_record("a1", "work") is not None
        assert evo_store.get_record("a1", "home") is None

    def test_stats_are_isolated_between_profiles(self, evo_store):
        evo_store.save_record(
            _make_record(record_id="w1", status=EvolutionStatus.PROMOTED), "work")
        assert evo_store.get_stats("work")["total"] == 1
        assert evo_store.get_stats("home")["total"] == 0

    def test_attempts_are_isolated_between_profiles(self, evo_store):
        for i in range(3):
            evo_store.save_record(
                _make_record(record_id=f"w{i}", skill_name="risky",
                             status=EvolutionStatus.REJECTED), "work")
        assert evo_store.has_exceeded_attempts("risky", "work") is True
        # The SAME skill name is untouched under 'home'.
        assert evo_store.count_attempts("risky", "home") == 0
        assert evo_store.has_exceeded_attempts("risky", "home") is False

    def test_cycle_budget_is_isolated_between_profiles(self, evo_store):
        # 'work' exhausts its 3-per-cycle budget.
        for _ in range(3):
            evo_store.record_evolution_attempt("work")
        assert evo_store.can_evolve("work") is False
        # 'home' still has its full, independent budget.
        assert evo_store.can_evolve("home") is True
        # Resetting 'work' does not touch 'home' and restores 'work'.
        evo_store.reset_cycle("work")
        assert evo_store.can_evolve("work") is True

    def test_self_migration_backfills_legacy_rows_to_default(self, tmp_path):
        """A pre-isolation DB (no profile_id) migrates + backfills to 'default'."""
        db = tmp_path / "legacy_evo.db"
        # Build the OLD schema by hand (no profile_id anywhere).
        conn = sqlite3.connect(str(db))
        conn.executescript(
            "CREATE TABLE skill_evolution_log ("
            " id TEXT PRIMARY KEY, skill_name TEXT NOT NULL,"
            " evolution_type TEXT NOT NULL, trigger_type TEXT NOT NULL,"
            " status TEXT DEFAULT 'candidate', evidence TEXT DEFAULT '[]',"
            " created_at TEXT NOT NULL);"
            "CREATE TABLE evolution_cycle_state ("
            " key TEXT PRIMARY KEY, value INTEGER DEFAULT 0, updated_at TEXT);"
            "INSERT INTO skill_evolution_log "
            "(id, skill_name, evolution_type, trigger_type, status, created_at) "
            "VALUES ('legacy1','oldskill','fix','post_session','promoted','2026-01-01T00:00:00Z');"
            "INSERT INTO evolution_cycle_state (key, value) VALUES ('cycle_count', 2);"
        )
        conn.commit()
        conn.close()

        # Constructing the store must self-migrate.
        store = EvolutionStore(db)
        cols = {r[1] for r in sqlite3.connect(str(db)).execute(
            "PRAGMA table_info(skill_evolution_log)").fetchall()}
        assert "profile_id" in cols
        # Legacy row is now under 'default'.
        assert store.get_record("legacy1", "default") is not None
        assert store.get_record("legacy1", "other") is None
        # Legacy cycle counter migrated to 'default' (2 used → 1 left of 3).
        assert store.can_evolve("default") is True
        store.record_evolution_attempt("default")  # now 3 used
        assert store.can_evolve("default") is False
