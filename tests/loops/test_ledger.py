"""Tests for the loop ledger: entry serialization, in-memory + SLM-backed."""

import pytest

from superlocalmemory.loops.engine import run_bounded_loop
from superlocalmemory.loops.ledger import (
    InMemoryLedger,
    LedgerEntry,
    SLMMemoryLedger,
)
from superlocalmemory.loops.models import Bounds, LapResult, Verdict


# ---------------------------------------------------------------------------
# LedgerEntry serialization
# ---------------------------------------------------------------------------


def test_entry_json_roundtrip():
    entry = LedgerEntry(
        run_id="r1", name="fix", lap=2, ts="2026-07-23T00:00:00+00:00",
        decision="done", passed=True, detail="pytest passed",
        budget={"tokens": 10, "wallclock_s": 0.5},
    )
    restored = LedgerEntry.from_json(entry.to_json())
    assert restored == entry


def test_entry_from_json_rejects_garbage():
    assert LedgerEntry.from_json("not json") is None
    assert LedgerEntry.from_json("[1,2,3]") is None
    assert LedgerEntry.from_json('{"name":"x"}') is None  # missing run_id/lap


# ---------------------------------------------------------------------------
# InMemoryLedger
# ---------------------------------------------------------------------------


def test_in_memory_records_and_reads():
    led = InMemoryLedger()
    led.record(LedgerEntry("r1", "fix", 1, "t", "continue", False, "d"))
    led.record(LedgerEntry("r1", "fix", 2, "t", "done", True, "d"))
    laps = led.laps("r1")
    assert [e.lap for e in laps] == [1, 2]
    assert led.runs("fix") == ["r1"]
    assert led.laps("missing") == []


def test_in_memory_runs_filters_by_name():
    led = InMemoryLedger()
    led.record(LedgerEntry("r1", "alpha", 1, "t", "done", True, "d"))
    led.record(LedgerEntry("r2", "beta", 1, "t", "done", True, "d"))
    assert led.runs("alpha") == ["r1"]
    assert led.runs("beta") == ["r2"]


# ---------------------------------------------------------------------------
# SLMMemoryLedger with a fake store (translation logic, no engine)
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal add/list_session/list_prefix store for unit-testing translation."""

    def __init__(self):
        self.rows = []  # list of (session_id, content)

    def add(self, content, *, session_id, metadata):
        self.rows.append((session_id, content))

    def list_session(self, session_id):
        return [{"content": c} for s, c in self.rows if s == session_id]

    def list_prefix(self, prefix):
        return [{"content": c} for s, c in self.rows if s.startswith(prefix)]


def test_slm_ledger_translation_roundtrip():
    led = SLMMemoryLedger(_FakeStore())
    led.record(LedgerEntry("run-1", "fix", 1, "t", "continue", False, "d1"))
    led.record(LedgerEntry("run-1", "fix", 2, "t", "done", True, "d2"))
    laps = led.laps("run-1")
    assert [e.lap for e in laps] == [1, 2]
    assert laps[-1].decision == "done" and laps[-1].passed is True
    assert led.runs("fix") == ["run-1"]


def test_slm_ledger_runs_isolates_by_name():
    store = _FakeStore()
    led = SLMMemoryLedger(store)
    led.record(LedgerEntry("run-a", "alpha", 1, "t", "done", True, "d"))
    led.record(LedgerEntry("run-b", "beta", 1, "t", "done", True, "d"))
    assert led.runs("alpha") == ["run-a"]
    assert led.runs("beta") == ["run-b"]
    assert led.laps("run-a") and not led.laps("run-a")[0].run_id == "run-b"


def test_engine_ledger_uses_nonblocking_write_through_path():
    from superlocalmemory.loops.ledger import _EngineLedgerStore

    class _Engine:
        profile_id = "default"

        def __init__(self):
            self.calls = []

        def store(self, *_args, **_kwargs):
            raise AssertionError("ledger must not run synchronous enrichment")

        def store_fast(self, content, *, metadata, index_external):
            self.calls.append((content, metadata, index_external))
            return ["fact"]

    engine = _Engine()
    store = _EngineLedgerStore(engine, owns_engine=False)
    store.add("lap", session_id="slm-loop:run", metadata={"loop": "demo"})

    assert engine.calls == [
        (
            "lap",
            {"loop": "demo", "session_id": "slm-loop:run"},
            False,
        ),
    ]


def test_engine_ledger_falls_back_for_legacy_adapter_engine():
    from superlocalmemory.loops.ledger import _EngineLedgerStore

    class _LegacyEngine:
        profile_id = "default"

        def __init__(self):
            self.calls = []

        def store(self, content, *, session_id, metadata):
            self.calls.append((content, session_id, metadata))

    engine = _LegacyEngine()
    store = _EngineLedgerStore(engine, owns_engine=False)
    store.add("lap", session_id="slm-loop:run", metadata={"loop": "demo"})

    assert engine.calls == [
        ("lap", "slm-loop:run", {"loop": "demo"}),
    ]


# ---------------------------------------------------------------------------
# SLMMemoryLedger against a REAL engine on a temp DB (value-add path)
# ---------------------------------------------------------------------------


def test_slm_ledger_persists_to_real_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    from superlocalmemory.loops.ledger import open_engine_store

    db = tmp_path / "loops.db"
    store = open_engine_store(db)
    try:
        ledger = SLMMemoryLedger(store)
        out = run_bounded_loop(
            "demo-fix",
            bounds=Bounds(max_iterations=5),
            runner=lambda lap: LapResult(changed=True, tokens=5),
            gate=lambda lap: Verdict(lap >= 2, f"gate {lap}"),
            ledger=ledger,
            run_id="demo-fix-xyz",
        )
        assert out.status.value == "DONE" and out.laps == 2

        laps = ledger.laps("demo-fix-xyz")
        assert [e.decision for e in laps] == ["continue", "done"]
        assert laps[-1].budget.get("tokens") == 10
        assert ledger.runs("demo-fix") == ["demo-fix-xyz"]
    finally:
        store.close()
