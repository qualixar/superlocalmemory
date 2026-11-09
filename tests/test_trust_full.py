# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Tests for V3 Trust System — Task 5 of V3 build.

Covers TrustScorer (Beta distribution) via record_signal/propagate_recall_trust/
set_trust, SignalRecorder (burst detection), TrustGate (pre-operation checks),
and TrustError, all against a hand-rolled MockDB.

update_on_confirmation / update_on_contradiction / update_on_access and the
default-lookup paths (get_fact_trust, get_entity_trust, generic get_trust,
empty get_all_scores) are NOT re-tested here — they are covered against a
real DatabaseManager in tests/test_trust/test_scorer.py, which is the
canonical copy for that behavior.
"""
from __future__ import annotations

import pytest

from superlocalmemory.trust.gate import TrustError, TrustGate
from superlocalmemory.trust.scorer import TrustScorer
from superlocalmemory.trust.signals import VALID_SIGNAL_TYPES, SignalRecorder


class _DictRow:
    """Wraps a dict to support dict(row) like sqlite3.Row."""
    def __init__(self, data: dict) -> None:
        self._data = dict(data)
    def __getitem__(self, key: str):
        return self._data[key]
    def keys(self):
        return self._data.keys()
    def __iter__(self):
        return iter(self._data)
    def __len__(self):
        return len(self._data)


class MockDB:
    """In-memory mock of DatabaseManager.execute() for trust_scores."""
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str], dict] = {}
        self._override: list | None = None

    def set_query_result(self, rows: list) -> None:
        self._override = rows

    def execute(self, sql: str, params: tuple | None = None) -> list:
        up = sql.strip().upper()
        if up.startswith("SELECT") and self._override is not None:
            result, self._override = self._override, None
            return [_DictRow(r) for r in result]
        if params is None:
            params = ()
        if up.startswith("SELECT") and "WHERE TARGET_TYPE" in up:
            key = (params[0], params[1], params[2])
            row = self._rows.get(key)
            return [_DictRow(row)] if row else []
        if up.startswith("SELECT") and "WHERE PROFILE_ID" in up:
            return [_DictRow(r) for r in self._rows.values() if r["profile_id"] == params[0]]
        if up.startswith("SELECT") and "WHERE TRUST_ID" in up:
            tid = params[0] if params else None
            for r in self._rows.values():
                if r.get("trust_id") == tid:
                    return [_DictRow(r)]
            return []
        if up.startswith("INSERT"):
            tid, pid, tt, ti, score, ev, lu = params
            self._rows[(tt, ti, pid)] = {"trust_id": tid, "profile_id": pid,
                "target_type": tt, "target_id": ti, "trust_score": score,
                "evidence_count": ev, "last_updated": lu}
            return []
        if up.startswith("UPDATE"):
            score, ev, lu, tid = params
            for row in self._rows.values():
                if row["trust_id"] == tid:
                    row.update(trust_score=score, evidence_count=ev, last_updated=lu)
                    break
            return []
        return []


@pytest.fixture()
def db() -> MockDB:
    return MockDB()

@pytest.fixture()
def scorer(db: MockDB) -> TrustScorer:
    return TrustScorer(db)

@pytest.fixture()
def signals(db: MockDB) -> SignalRecorder:
    return SignalRecorder(db)


# -- TrustScorer: defaults --
#
# Defaults for get_fact_trust / get_entity_trust / the generic get_trust are
# covered by tests/test_trust/test_scorer.py (TestConvenience, TestGetTrust)
# against a real DatabaseManager, which is the more faithful double. Only
# get_agent_trust's default stays here: it is the one lookup with its own
# code path (the is_anonymous short-circuit in scorer.py), not exercised by
# that file.

class TestScorerDefaults:
    def test_new_agent_default(self, scorer: TrustScorer) -> None:
        assert scorer.get_agent_trust("agent-1", "p1") == pytest.approx(0.5, abs=0.01)


# -- TrustScorer: signal recording --

class TestScorerSignals:
    def test_positive_increases(self, scorer: TrustScorer) -> None:
        assert scorer.record_signal("a1", "p1", "store_success") > 0.5

    def test_negative_decreases(self, scorer: TrustScorer) -> None:
        assert scorer.record_signal("a1", "p1", "contradiction") < 0.5

    def test_recall_hit_boosts(self, scorer: TrustScorer) -> None:
        assert scorer.record_signal("a1", "p1", "recall_hit") > 0.5

    def test_deletion_decreases(self, scorer: TrustScorer) -> None:
        assert scorer.record_signal("a1", "p1", "deletion") < 0.5

    def test_store_rejected_decreases(self, scorer: TrustScorer) -> None:
        assert scorer.record_signal("a1", "p1", "store_rejected") < 0.5

    def test_multiple_accumulate(self, scorer: TrustScorer) -> None:
        for _ in range(10):
            score = scorer.record_signal("good", "p1", "store_success")
        assert score > 0.9

    def test_mixed_signals(self, scorer: TrustScorer) -> None:
        scorer.record_signal("mx", "p1", "store_success")
        scorer.record_signal("mx", "p1", "store_success")
        score = scorer.record_signal("mx", "p1", "contradiction")
        assert 0.3 < score < 0.6


# -- TrustScorer: propagation --

class TestPropagation:
    def test_propagate_recall_trust(self, scorer: TrustScorer) -> None:
        assert scorer.propagate_recall_trust("a1", "p1") > 0.5


# -- TrustScorer: backward compat --
#
# update_on_confirmation / update_on_contradiction / update_on_access are
# fully covered against a real DatabaseManager by tests/test_trust/
# test_scorer.py (TestConfirmation, TestContradiction, TestAccess) — same
# methods, same bounds (single-update direction, convergence toward 0/1,
# clamping at the 0.0/1.0 edges). That suite is the better double (real
# schema, not a hand-rolled MockDB) so it is the canonical copy; this class
# duplicated it line for line and is removed rather than kept as a second
# copy of the same assertions.


# -- TrustScorer: set_trust --

class TestSetTrust:
    def test_set_directly(self, scorer: TrustScorer) -> None:
        assert scorer.set_trust("a1", "p1", alpha=8.0, beta_param=2.0) == pytest.approx(0.8, abs=0.01)

    def test_reflected_in_get(self, scorer: TrustScorer) -> None:
        scorer.set_trust("a1", "p1", alpha=8.0, beta_param=2.0)
        assert scorer.get_agent_trust("a1", "p1") == pytest.approx(0.8, abs=0.05)

    def test_clamps_negative(self, scorer: TrustScorer) -> None:
        assert scorer.set_trust("a1", "p1", alpha=-5.0, beta_param=-3.0) == pytest.approx(0.5, abs=0.01)


# -- TrustScorer: get_all_scores --
#
# The empty-profile case is covered by test_trust/test_scorer.py::
# TestGetAllScores.test_empty_profile_returns_empty_list against a real
# DatabaseManager — same call, same assertion, removed here as a duplicate.

class TestGetAllScores:
    def test_returns_dicts_with_beta(self, scorer: TrustScorer) -> None:
        scorer.record_signal("a1", "p1", "store_success")
        scores = scorer.get_all_scores("p1")
        assert len(scores) == 1
        assert all(k in scores[0] for k in ("alpha", "beta_param", "trust_score"))
        assert scores[0]["target_type"] == "agent"


# -- SignalRecorder: burst detection --

class TestBurstDetection:
    def test_over_threshold(self) -> None:
        rec = SignalRecorder(MockDB(), burst_window_seconds=60, burst_threshold=3)
        for _ in range(3):
            rec.record("a1", "p1", "store_success")
        assert rec.is_burst_detected("a1", "p1") is True

    def test_under_threshold(self) -> None:
        rec = SignalRecorder(MockDB(), burst_window_seconds=60, burst_threshold=10)
        rec.record("a1", "p1", "store_success")
        rec.record("a1", "p1", "store_success")
        assert rec.is_burst_detected("a1", "p1") is False

    def test_unknown_agent(self) -> None:
        rec = SignalRecorder(MockDB(), burst_window_seconds=60, burst_threshold=3)
        assert rec.is_burst_detected("nobody", "p1") is False

    def test_per_agent_isolation(self) -> None:
        rec = SignalRecorder(MockDB(), burst_window_seconds=60, burst_threshold=3)
        for _ in range(3):
            rec.record("a", "p1", "store_success")
        rec.record("b", "p1", "store_success")
        assert rec.is_burst_detected("a", "p1") is True
        assert rec.is_burst_detected("b", "p1") is False


# -- SignalRecorder: recording --

class TestSignalRecording:
    def test_valid_returns_true(self, signals: SignalRecorder) -> None:
        assert signals.record("a1", "p1", "store_success") is True

    def test_invalid_returns_false(self, signals: SignalRecorder) -> None:
        assert signals.record("a1", "p1", "invalid_type") is False

    def test_recent_signals(self, signals: SignalRecorder) -> None:
        signals.record("a1", "p1", "store_success")
        signals.record("a1", "p1", "recall_hit")
        recent = signals.get_recent_signals("a1", "p1")
        assert len(recent) == 2
        assert recent[0]["signal_type"] == "recall_hit"

    def test_recent_limited(self, signals: SignalRecorder) -> None:
        for _ in range(10):
            signals.record("a1", "p1", "store_success")
        assert len(signals.get_recent_signals("a1", "p1", limit=3)) == 3

    def test_burst_status(self) -> None:
        rec = SignalRecorder(MockDB(), burst_window_seconds=60, burst_threshold=2)
        rec.record("fast", "p1", "store_success")
        rec.record("fast", "p1", "store_success")
        rec.record("slow", "p1", "store_success")
        status = rec.get_burst_status("p1")
        assert status["fast"] is True
        assert status["slow"] is False

    def test_valid_types_exposed(self) -> None:
        for t in ("store_success", "contradiction", "recall_hit"):
            assert t in VALID_SIGNAL_TYPES


# -- TrustGate: write --

def _mock_row(tid, pid, target, score, ev):
    return {"trust_id": tid, "profile_id": pid, "target_type": "agent",
            "target_id": target, "trust_score": score,
            "evidence_count": ev, "last_updated": "2026-03-16"}


class TestTrustGateWrite:
    def test_allows_high_trust(self, scorer: TrustScorer, db: MockDB) -> None:
        db.set_query_result([_mock_row("t1", "p1", "good", 0.8, 10)])
        TrustGate(scorer).check_write("good", "p1")

    def test_rejects_low_trust(self, scorer: TrustScorer, db: MockDB) -> None:
        db.set_query_result([_mock_row("t1", "p1", "bad", 0.1, 20)])
        with pytest.raises(TrustError) as exc:
            TrustGate(scorer).check_write("bad", "p1")
        assert exc.value.operation == "write"

    def test_unknown_agent_passes(self, scorer: TrustScorer) -> None:
        TrustGate(scorer).check_write("new-agent", "p1")


# -- TrustGate: delete --

class TestTrustGateDelete:
    def test_mid_trust_write_ok_delete_fail(self, scorer: TrustScorer, db: MockDB) -> None:
        db.set_query_result([_mock_row("t1", "p1", "mid", 0.4, 5)])
        gate = TrustGate(scorer)
        gate.check_write("mid", "p1")
        db.set_query_result([_mock_row("t1", "p1", "mid", 0.4, 5)])
        with pytest.raises(TrustError) as exc:
            gate.check_delete("mid", "p1")
        assert exc.value.operation == "delete"

    def test_high_trust_allows(self, scorer: TrustScorer, db: MockDB) -> None:
        db.set_query_result([_mock_row("t1", "p1", "ok", 0.9, 50)])
        TrustGate(scorer).check_delete("ok", "p1")


# -- TrustGate: read + config --

class TestTrustGateReadConfig:
    def test_read_always_passes(self, scorer: TrustScorer, db: MockDB) -> None:
        db.set_query_result([_mock_row("t1", "p1", "untrusted", 0.01, 100)])
        TrustGate(scorer).check_read("untrusted", "p1")

    def test_custom_thresholds(self, scorer: TrustScorer) -> None:
        gate = TrustGate(scorer, write_threshold=0.8, delete_threshold=0.95)
        assert gate.write_threshold == 0.8
        assert gate.delete_threshold == 0.95

    def test_invalid_threshold(self, scorer: TrustScorer) -> None:
        with pytest.raises(ValueError):
            TrustGate(scorer, write_threshold=-0.1)
        with pytest.raises(ValueError):
            TrustGate(scorer, delete_threshold=1.5)

    def test_trust_error_is_permission_error(self) -> None:
        err = TrustError("x", 0.1, 0.3, "write")
        assert isinstance(err, PermissionError)
        assert err.agent_id == "x"
        assert err.trust_score == 0.1
