# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file

"""Phase 2 Governance Tests — TDD RED → GREEN.

Groups:
  0 - @admits decorator lock (1 test — locks Phase 1 gating)
  2 - Strict structured decision parse
  3 - Append-only transition log (insert_record, append_transition, hash linkage)
  4 - State machine in _process_candidate
  5 - SkillActivator (atomic activation + rollback)
  6 - Audit emission (AuditChain integration)

Group 1 (RBAC/admission) is owned by Phase 1. Not duplicated here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.compliance.audit import AuditChain
from superlocalmemory.evolution.blind_verifier import VerificationResult
from superlocalmemory.evolution.evolution_store import EvolutionStore
from superlocalmemory.evolution.skill_evolver import SkillEvolver
from superlocalmemory.evolution.types import (
    EvolutionCandidate,
    EvolutionRecord,
    EvolutionStatus,
    EvolutionType,
    TriggerType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    record_id: str = "rec-test",
    skill_name: str = "brainstorming",
    status: EvolutionStatus = EvolutionStatus.CANDIDATE,
) -> EvolutionRecord:
    return EvolutionRecord(
        id=record_id,
        skill_name=skill_name,
        parent_skill_id=None,
        evolution_type=EvolutionType.FIX,
        trigger=TriggerType.HEALTH_CHECK,
        status=status,
        evidence=("low score",),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_candidate(skill_name: str = "brainstorming") -> EvolutionCandidate:
    return EvolutionCandidate(
        skill_name=skill_name,
        evolution_type=EvolutionType.FIX,
        trigger=TriggerType.HEALTH_CHECK,
        evidence=("evidence-1",),
        effective_score=0.4,
        invocation_count=10,
    )


def _make_evolver(tmp_path: Path, audit_chain: AuditChain | None = None) -> SkillEvolver:
    """Return a SkillEvolver backed by a real tmp_path DB, no real LLM."""
    db = tmp_path / "test_evo.db"
    if audit_chain is None:
        audit_chain = AuditChain(":memory:")
    evolver = SkillEvolver(str(db), profile_id="default", audit_chain=audit_chain)
    evolver._backend = "none"  # prevent actual LLM backend detection
    return evolver


# ---------------------------------------------------------------------------
# Group 0: @admits decorator lock — one test to lock Phase 1 gating
# ---------------------------------------------------------------------------

class TestAdmitsDecoratorPresence:
    """Assert that the @admits(EVOLVE_SKILL) decorator is wired on evolve_skill."""

    def test_evolve_skill_has_admits_decorator(self):
        """The evolve_skill MCP tool must be decorated with @admits(EVOLVE_SKILL)."""
        import superlocalmemory.mcp.tools_evolution as evo_mod
        # The module registers the tool inside register_evolution_tools().
        # We verify the admission check is imported and applied at module level.
        import superlocalmemory.core.admission as adm_mod
        assert hasattr(adm_mod, "admits"), "admits() factory must exist in core.admission"

        from superlocalmemory.core.operation_request import OperationKind
        assert hasattr(OperationKind, "EVOLVE_SKILL"), (
            "OperationKind.EVOLVE_SKILL must exist (Phase 1 delivered this)"
        )
        # Verify the @admits import is present in tools_evolution source
        source = inspect.getsource(evo_mod)
        assert "admits" in source, "@admits must be imported in tools_evolution"
        assert "EVOLVE_SKILL" in source, "@admits(EVOLVE_SKILL) must appear in tools_evolution"


# ---------------------------------------------------------------------------
# Group 2: Strict structured decision parse
# ---------------------------------------------------------------------------

class TestParseApprovalDecision:
    """Unit tests for SkillEvolver._parse_approval_decision."""

    @pytest.fixture
    def evolver(self, tmp_path):
        return _make_evolver(tmp_path)

    def test_approve_returns_true(self, evolver):
        assert evolver._parse_approval_decision('{"decision": "approve"}') is True

    def test_approve_case_insensitive_value(self, evolver):
        assert evolver._parse_approval_decision('{"decision": "APPROVE"}') is True

    def test_reject_returns_false(self, evolver):
        assert evolver._parse_approval_decision('{"decision": "reject"}') is False

    def test_yes_text_returns_false(self, evolver):
        """Plain 'yes' must no longer be accepted."""
        assert evolver._parse_approval_decision("yes") is False

    def test_yes_ish_returns_false(self, evolver):
        assert evolver._parse_approval_decision("yes-ish, probably") is False

    def test_malformed_json_returns_false(self, evolver):
        assert evolver._parse_approval_decision("{decision: approve}") is False

    def test_empty_string_returns_false(self, evolver):
        assert evolver._parse_approval_decision("") is False

    def test_approved_wrong_value_returns_false(self, evolver):
        """'approved' is NOT accepted — only exactly 'approve'."""
        assert evolver._parse_approval_decision('{"decision": "approved"}') is False

    def test_with_leading_text_returns_true(self, evolver):
        """LLM may emit leading text before the JSON object — still parses."""
        response = 'Sure, here is my decision: {"decision": "approve"}'
        assert evolver._parse_approval_decision(response) is True

    def test_maybe_returns_false(self, evolver):
        assert evolver._parse_approval_decision('{"decision": "maybe"}') is False

    def test_none_value_returns_false(self, evolver):
        assert evolver._parse_approval_decision('{"decision": null}') is False

    def test_missing_decision_key_returns_false(self, evolver):
        assert evolver._parse_approval_decision('{"result": "approve"}') is False


# ---------------------------------------------------------------------------
# Group 3: Append-only transition log
# ---------------------------------------------------------------------------

class TestAppendOnlyTransitionLog:
    """insert_record, append_transition, get_latest_status, get_transitions."""

    @pytest.fixture
    def store(self, tmp_path):
        return EvolutionStore(str(tmp_path / "test.db"))

    @pytest.fixture
    def store_with_record(self, store):
        record = _make_record("r1")
        store.insert_record(record, "default")
        return store

    def test_insert_record_creates_row(self, store):
        record = _make_record("r1")
        store.insert_record(record, "default")
        retrieved = store.get_record("r1", "default")
        assert retrieved is not None
        assert retrieved.id == "r1"
        assert retrieved.status == EvolutionStatus.CANDIDATE

    def test_insert_record_duplicate_raises(self, store):
        record = _make_record("r1")
        store.insert_record(record, "default")
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_record(record, "default")

    def test_append_transition_creates_row(self, store, store_with_record):
        result_hash = store.append_transition(
            "r1", "default",
            EvolutionStatus.CANDIDATE, EvolutionStatus.REJECTED,
            reason="llm_confirmation_rejected",
        )
        assert isinstance(result_hash, str)
        assert len(result_hash) == 64  # SHA-256 hex

    def test_append_transition_hash_linkage(self, store, store_with_record):
        """Second transition's prev_hash equals first transition's transition_hash."""
        h1 = store.append_transition(
            "r1", "default",
            EvolutionStatus.CANDIDATE, EvolutionStatus.VERIFIED_QUARANTINED,
            reason="blind_verified",
        )
        store.append_transition(
            "r1", "default",
            EvolutionStatus.VERIFIED_QUARANTINED, EvolutionStatus.APPROVED,
            reason="auto_approved",
        )
        rows = store.get_transitions("r1", "default")
        assert len(rows) == 2
        assert rows[1]["prev_hash"] == h1

    def test_append_transition_no_update_on_existing_log_row(self, store, store_with_record, tmp_path):
        """skill_evolution_log row must NOT be modified after append_transition."""
        original = store.get_record("r1", "default")
        store.append_transition(
            "r1", "default",
            EvolutionStatus.CANDIDATE, EvolutionStatus.REJECTED,
        )
        after = store.get_record("r1", "default")
        # The main log row's status should be unchanged (still CANDIDATE)
        assert after.status == original.status
        assert after.id == original.id

    def test_get_latest_status_returns_most_recent(self, store, store_with_record):
        store.append_transition(
            "r1", "default",
            EvolutionStatus.CANDIDATE, EvolutionStatus.VERIFIED_QUARANTINED,
        )
        store.append_transition(
            "r1", "default",
            EvolutionStatus.VERIFIED_QUARANTINED, EvolutionStatus.APPROVED,
        )
        status = store.get_latest_status("r1", "default")
        assert status == EvolutionStatus.APPROVED

    def test_get_transitions_ordered_by_seq(self, store, store_with_record):
        store.append_transition(
            "r1", "default",
            EvolutionStatus.CANDIDATE, EvolutionStatus.VERIFIED_QUARANTINED,
        )
        store.append_transition(
            "r1", "default",
            EvolutionStatus.VERIFIED_QUARANTINED, EvolutionStatus.ACTIVE,
        )
        rows = store.get_transitions("r1", "default")
        assert len(rows) == 2
        assert rows[0]["from_status"] == "candidate"
        assert rows[0]["to_status"] == "verified_quarantined"
        assert rows[1]["from_status"] == "verified_quarantined"
        assert rows[1]["to_status"] == "active"
        seqs = [r["seq"] for r in rows]
        assert seqs == sorted(seqs)

    def test_cannot_update_transition_row(self, tmp_path):
        """BEFORE UPDATE trigger must raise a DB error on any UPDATE attempt.

        Python 3.13 + SQLite 3.49 raises IntegrityError for RAISE(ABORT, …);
        older versions raise OperationalError.  The test accepts either —
        what matters is that the trigger fires and the message is preserved.
        """
        db_path = str(tmp_path / "test.db")
        store = EvolutionStore(db_path)
        store.insert_record(_make_record("r1"), "default")
        store.append_transition(
            "r1", "default",
            EvolutionStatus.CANDIDATE, EvolutionStatus.REJECTED,
        )
        conn = sqlite3.connect(db_path)
        try:
            # The trigger fires regardless of which subclass is raised.
            with pytest.raises(
                (sqlite3.OperationalError, sqlite3.IntegrityError),
                match="append-only",
            ):
                conn.execute(
                    "UPDATE skill_evolution_transitions "
                    "SET to_status = 'approved' WHERE seq = 1"
                )
                conn.commit()
        finally:
            conn.close()

    def test_genesis_prev_hash_for_first_transition(self, store, store_with_record):
        """First transition for a record must have prev_hash = 'genesis'."""
        store.append_transition(
            "r1", "default",
            EvolutionStatus.CANDIDATE, EvolutionStatus.REJECTED,
        )
        rows = store.get_transitions("r1", "default")
        assert rows[0]["prev_hash"] == "genesis"

    def test_get_latest_status_no_transitions_returns_none(self, store, store_with_record):
        result = store.get_latest_status("r1", "default")
        assert result is None

    def test_get_latest_status_unknown_record_returns_none(self, store):
        result = store.get_latest_status("does-not-exist", "default")
        assert result is None


# ---------------------------------------------------------------------------
# Group 4: State machine in _process_candidate
# ---------------------------------------------------------------------------

class TestStateMachineProcessCandidate:
    """_process_candidate uses insert_record once + append_transition per state."""

    def _mock_evolver(self, tmp_path, audit_chain=None):
        evolver = _make_evolver(tmp_path, audit_chain)
        # Prevent any actual LLM dispatch
        evolver._backend = "none"
        return evolver

    def test_candidate_to_verified_quarantined_on_success(self, tmp_path):
        audit = AuditChain(":memory:")
        evolver = self._mock_evolver(tmp_path, audit)
        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: Better brainstorming skill\n\nContent."

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(passed=True, confidence=0.9, reasoning="ok"),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
            patch.object(evolver, "_write_evolved_skill",
                         return_value=(tmp_path / "qdir" / "SKILL.md", "brainstorming-vabc12")),
        ):
            outcome = evolver._process_candidate(candidate, "default")

        assert outcome == "quarantined"

        recent = evolver._store.get_recent("default", limit=1)
        assert len(recent) == 1
        record_id = recent[0].id
        transitions = evolver._store.get_transitions(record_id, "default")
        assert len(transitions) == 1
        assert transitions[0]["from_status"] == "candidate"
        assert transitions[0]["to_status"] == "verified_quarantined"

    def test_candidate_to_rejected_on_confirm_fail(self, tmp_path):
        evolver = self._mock_evolver(tmp_path)
        candidate = _make_candidate()

        with (
            patch.object(evolver, "_llm_confirm", return_value=False),
            patch.object(evolver, "_read_skill_content", return_value="original"),
        ):
            outcome = evolver._process_candidate(candidate, "default")

        assert outcome == "rejected"

        recent = evolver._store.get_recent("default", limit=1)
        record_id = recent[0].id
        transitions = evolver._store.get_transitions(record_id, "default")
        assert len(transitions) == 1
        t = transitions[0]
        assert t["from_status"] == "candidate"
        assert t["to_status"] == "rejected"
        assert "llm_confirmation" in t["reason"].lower()

    def test_candidate_to_failed_on_mutation_fail(self, tmp_path):
        evolver = self._mock_evolver(tmp_path)
        candidate = _make_candidate()

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=None),
            patch.object(evolver, "_read_skill_content", return_value="original"),
        ):
            outcome = evolver._process_candidate(candidate, "default")

        assert outcome == "rejected"

        recent = evolver._store.get_recent("default", limit=1)
        record_id = recent[0].id
        transitions = evolver._store.get_transitions(record_id, "default")
        assert len(transitions) == 1
        assert transitions[0]["to_status"] == "failed"

    def test_candidate_to_rejected_on_blind_verify_fail(self, tmp_path):
        evolver = self._mock_evolver(tmp_path)
        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: skill\n\nContent."

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(
                    passed=False, confidence=0.2, reasoning="quality too low"
                ),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
        ):
            outcome = evolver._process_candidate(candidate, "default")

        assert outcome == "rejected"

        recent = evolver._store.get_recent("default", limit=1)
        record_id = recent[0].id
        transitions = evolver._store.get_transitions(record_id, "default")
        assert len(transitions) == 1
        assert transitions[0]["to_status"] == "rejected"
        assert "blind_verification" in transitions[0]["reason"].lower()

    def test_no_auto_approve_stops_at_quarantined(self, tmp_path):
        """Without auto-approve, _process_candidate returns 'quarantined'."""
        evolver = self._mock_evolver(tmp_path)
        evolver._config = None  # no config → auto_approve defaults False
        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: skill\n\nContent."

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(passed=True, confidence=0.9, reasoning="ok"),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
            patch.object(evolver, "_write_evolved_skill",
                         return_value=(tmp_path / "qdir" / "SKILL.md", "brainstorming-vabc12")),
        ):
            outcome = evolver._process_candidate(candidate, "default")

        assert outcome == "quarantined"

    def test_auto_approve_mode_activates_immediately(self, tmp_path):
        """With auto_approve=True, _process_candidate reaches ACTIVE status."""
        audit = AuditChain(":memory:")
        evolver = self._mock_evolver(tmp_path, audit)

        # Config with auto_approve=True
        mock_cfg = MagicMock()
        mock_cfg.evolution.auto_approve = True
        evolver._config = mock_cfg

        # Mock activator
        mock_activator = MagicMock()
        mock_activator.activate.return_value = {
            "skill_name": "brainstorming",
            "live_path": str(tmp_path / "live" / "brainstorming" / "SKILL.md"),
            "backup_path": None,
            "content_hash": "abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "actor_id": "auto",
        }
        evolver._activator = mock_activator

        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: skill\n\nContent."

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(passed=True, confidence=0.9, reasoning="ok"),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
            patch.object(evolver, "_write_evolved_skill",
                         return_value=(tmp_path / "qdir" / "SKILL.md", "brainstorming-vabc12")),
        ):
            outcome = evolver._process_candidate(candidate, "default")

        assert outcome == "evolved"

        recent = evolver._store.get_recent("default", limit=1)
        record_id = recent[0].id
        transitions = evolver._store.get_transitions(record_id, "default")
        # Should have: CANDIDATE→VQ, VQ→APPROVED, APPROVED→ACTIVE
        statuses = [(t["from_status"], t["to_status"]) for t in transitions]
        assert ("candidate", "verified_quarantined") in statuses
        assert ("verified_quarantined", "approved") in statuses
        assert ("approved", "active") in statuses

    def test_insert_record_called_once_per_candidate(self, tmp_path):
        """insert_record is called exactly once per candidate (CRIT-1)."""
        evolver = self._mock_evolver(tmp_path)
        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: skill\n\nContent."

        original_insert = evolver._store.insert_record
        insert_calls = []

        def counting_insert(record, profile_id):
            insert_calls.append(record.id)
            return original_insert(record, profile_id)

        evolver._store.insert_record = counting_insert

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(passed=True, confidence=0.9, reasoning="ok"),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
            patch.object(evolver, "_write_evolved_skill",
                         return_value=(tmp_path / "qdir" / "SKILL.md", "brainstorming-vabc12")),
        ):
            evolver._process_candidate(candidate, "default")

        assert len(insert_calls) == 1, "insert_record must be called exactly once"


# ---------------------------------------------------------------------------
# Group 5: SkillActivator
# ---------------------------------------------------------------------------

class TestSkillActivator:
    """Atomic activation, backup, rollback, path traversal guard."""

    @pytest.fixture
    def activator(self, tmp_path):
        from superlocalmemory.evolution.skill_activator import SkillActivator
        return SkillActivator(
            live_root=tmp_path / "live",
            backup_root=tmp_path / "backup",
            quarantine_root=tmp_path / "quarantine",
        )

    @pytest.fixture
    def quarantine_skill(self, tmp_path):
        """Create a quarantine artifact."""
        q_dir = tmp_path / "quarantine" / "brainstorming-vabc12"
        q_dir.mkdir(parents=True)
        skill_file = q_dir / "SKILL.md"
        skill_file.write_text("# Evolved skill content", encoding="utf-8")
        return "brainstorming-vabc12"

    def test_activate_writes_live_skill(self, activator, quarantine_skill, tmp_path):
        result = activator.activate("brainstorming", quarantine_skill)
        live_path = tmp_path / "live" / "brainstorming" / "SKILL.md"
        assert live_path.exists()
        assert live_path.read_text() == "# Evolved skill content"
        assert result["skill_name"] == "brainstorming"
        assert result["content_hash"] is not None

    def test_activate_creates_backup(self, activator, quarantine_skill, tmp_path):
        """When a live skill already exists, activate() backs it up."""
        live_dir = tmp_path / "live" / "brainstorming"
        live_dir.mkdir(parents=True)
        live_path = live_dir / "SKILL.md"
        live_path.write_text("# Original skill content", encoding="utf-8")

        activator.activate("brainstorming", quarantine_skill)

        backup = tmp_path / "backup" / "brainstorming" / "SKILL.md.bak"
        assert backup.exists()
        assert backup.read_text() == "# Original skill content"

    def test_activate_atomic_via_tmp_rename(self, activator, quarantine_skill, tmp_path):
        """After activate(), no .tmp file must remain."""
        activator.activate("brainstorming", quarantine_skill)
        live_dir = tmp_path / "live" / "brainstorming"
        tmp_files = list(live_dir.glob("*.tmp"))
        assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"

    def test_rollback_restores_backup(self, activator, quarantine_skill, tmp_path):
        """rollback() after activate() restores the original content."""
        live_dir = tmp_path / "live" / "brainstorming"
        live_dir.mkdir(parents=True)
        original_content = "# Original skill content"
        (live_dir / "SKILL.md").write_text(original_content, encoding="utf-8")

        activator.activate("brainstorming", quarantine_skill)
        result = activator.rollback("brainstorming")

        assert result["rolled_back"] is True
        live_path = tmp_path / "live" / "brainstorming" / "SKILL.md"
        assert live_path.read_text() == original_content

    def test_rollback_with_no_backup_removes_live(self, activator, quarantine_skill, tmp_path):
        """rollback() with no backup (new skill) removes the live file."""
        activator.activate("brainstorming", quarantine_skill)
        result = activator.rollback("brainstorming")

        assert result["rolled_back"] is True
        live_path = tmp_path / "live" / "brainstorming" / "SKILL.md"
        assert not live_path.exists()

    def test_activate_path_traversal_raises(self, activator, tmp_path):
        """skill_name with traversal sequences raises ValueError."""
        from superlocalmemory.evolution.skill_activator import SkillActivator
        activator_strict = SkillActivator(
            live_root=tmp_path / "live",
            backup_root=tmp_path / "backup",
            quarantine_root=tmp_path / "quarantine",
        )
        # Path traversal in skill_name → should raise ValueError
        with pytest.raises(ValueError):
            activator_strict.activate("../../../evil", "brainstorming-vabc12")

    def test_activate_missing_quarantine_raises(self, activator, tmp_path):
        """activate() when quarantine artifact missing → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            activator.activate("brainstorming", "does-not-exist")

    def test_full_activate_then_rollback_restores(self, activator, quarantine_skill, tmp_path):
        """activate() then rollback() → live file matches the original content."""
        live_dir = tmp_path / "live" / "brainstorming"
        live_dir.mkdir(parents=True)
        original = "# Original content to restore"
        (live_dir / "SKILL.md").write_text(original, encoding="utf-8")

        activator.activate("brainstorming", quarantine_skill)
        activator.rollback("brainstorming")

        live_path = live_dir / "SKILL.md"
        assert live_path.read_text() == original


# ---------------------------------------------------------------------------
# Group 6: Audit emission
# ---------------------------------------------------------------------------

class TestAuditEmission:
    """AuditChain.log() is called with correct operation strings."""

    def test_promotion_emits_audit_event(self, tmp_path):
        """After blind verify passes, audit.log called with operation='skill_promotion'."""
        audit = AuditChain(":memory:")
        evolver = _make_evolver(tmp_path, audit)
        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: skill\n\nContent."

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(passed=True, confidence=0.9, reasoning="ok"),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
            patch.object(evolver, "_write_evolved_skill",
                         return_value=(tmp_path / "qdir" / "SKILL.md", "brainstorming-vabc12")),
        ):
            evolver._process_candidate(candidate, "default")

        # Verify audit chain contains skill_promotion event
        conn = audit._get_conn()
        rows = conn.execute(
            "SELECT operation FROM audit_chain ORDER BY id"
        ).fetchall()
        audit._release_conn(conn)
        operations = [r["operation"] for r in rows]
        assert "skill_promotion" in operations

    def test_activation_emits_audit_event(self, tmp_path):
        """After activate() succeeds, audit.log called with 'skill_activation'."""
        audit = AuditChain(":memory:")
        evolver = _make_evolver(tmp_path, audit)

        mock_cfg = MagicMock()
        mock_cfg.evolution.auto_approve = True
        evolver._config = mock_cfg

        mock_activator = MagicMock()
        mock_activator.activate.return_value = {
            "skill_name": "brainstorming",
            "live_path": str(tmp_path / "live" / "brainstorming" / "SKILL.md"),
            "backup_path": None,
            "content_hash": "abc123" * 10 + "ab12",
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "actor_id": "auto",
        }
        evolver._activator = mock_activator

        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: skill\n\nContent."

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(passed=True, confidence=0.9, reasoning="ok"),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
            patch.object(evolver, "_write_evolved_skill",
                         return_value=(tmp_path / "qdir" / "SKILL.md", "brainstorming-vabc12")),
        ):
            evolver._process_candidate(candidate, "default")

        conn = audit._get_conn()
        rows = conn.execute(
            "SELECT operation FROM audit_chain ORDER BY id"
        ).fetchall()
        audit._release_conn(conn)
        operations = [r["operation"] for r in rows]
        assert "skill_activation" in operations

    def test_failed_activation_emits_audit_event(self, tmp_path):
        """SkillActivationError → audit.log called with 'skill_activation_failed'."""
        from superlocalmemory.evolution.skill_activator import SkillActivationError

        audit = AuditChain(":memory:")
        evolver = _make_evolver(tmp_path, audit)

        mock_cfg = MagicMock()
        mock_cfg.evolution.auto_approve = True
        evolver._config = mock_cfg

        mock_activator = MagicMock()
        mock_activator.activate.side_effect = SkillActivationError("write failed")
        evolver._activator = mock_activator

        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: skill\n\nContent."

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(passed=True, confidence=0.9, reasoning="ok"),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
            patch.object(evolver, "_write_evolved_skill",
                         return_value=(tmp_path / "qdir" / "SKILL.md", "brainstorming-vabc12")),
        ):
            outcome = evolver._process_candidate(candidate, "default")

        assert outcome == "rejected"

        conn = audit._get_conn()
        rows = conn.execute(
            "SELECT operation FROM audit_chain ORDER BY id"
        ).fetchall()
        audit._release_conn(conn)
        operations = [r["operation"] for r in rows]
        assert "skill_activation_failed" in operations

        # Check the transition is ROLLED_BACK
        recent = evolver._store.get_recent("default", limit=1)
        record_id = recent[0].id
        transitions = evolver._store.get_transitions(record_id, "default")
        statuses = [t["to_status"] for t in transitions]
        assert "rolled_back" in statuses

    def test_audit_chain_integrity_after_evolution(self, tmp_path):
        """After promotion + activation, AuditChain.verify_integrity() returns True."""
        audit = AuditChain(":memory:")
        evolver = _make_evolver(tmp_path, audit)

        mock_cfg = MagicMock()
        mock_cfg.evolution.auto_approve = True
        evolver._config = mock_cfg

        mock_activator = MagicMock()
        mock_activator.activate.return_value = {
            "skill_name": "brainstorming",
            "live_path": str(tmp_path / "live" / "brainstorming" / "SKILL.md"),
            "backup_path": None,
            "content_hash": "a" * 64,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "actor_id": "auto",
        }
        evolver._activator = mock_activator

        candidate = _make_candidate()
        evolved_content = "# Evolved\ndescription: skill\n\nContent."

        with (
            patch.object(evolver, "_llm_confirm", return_value=True),
            patch.object(evolver, "_generate_mutation", return_value=evolved_content),
            patch.object(
                evolver, "_blind_verify",
                return_value=VerificationResult(passed=True, confidence=0.9, reasoning="ok"),
            ),
            patch.object(evolver, "_read_skill_content", return_value="original"),
            patch.object(evolver, "_write_evolved_skill",
                         return_value=(tmp_path / "qdir" / "SKILL.md", "brainstorming-vabc12")),
        ):
            evolver._process_candidate(candidate, "default")

        assert audit.verify_integrity() is True
