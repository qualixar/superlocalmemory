# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Phase 4b — Bi-temporal completion test suite.

Groups:
  1. normalize_as_of() unit tests (UTC normalisation, format, round-trip)
  2. get_invalidated_fact_ids with as_of (transaction-time supersession)
  3. valid_until boundary fix (half-open interval <= )
  4. TemporalValidityFilter.filter() end-to-end with as_of (CRIT-2: score check)
  5. recall_trace MCP tool with as_of
  6. recall_trace HTTP endpoint with as_of
  7. prestage_context with as_of (CRIT-3: 3-arg fallback)
  8. UTC normalization at HTTP / MCP boundaries

TDD: Every test in this file was written BEFORE the implementation.
Run these first; confirm RED; implement; confirm GREEN.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# DB / schema helpers (guarded import so tests can be collected even if the
# storage module has import issues)
# ---------------------------------------------------------------------------

try:
    from superlocalmemory.storage import schema as real_schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

pytestmark_db = pytest.mark.skipif(not _DB_AVAILABLE, reason="storage module unavailable")


def _make_db(tmp_path: Path) -> "DatabaseManager":
    mgr = DatabaseManager(tmp_path / "test_p4.db")
    mgr.initialize(real_schema)
    return mgr


def _seed_facts(db: "DatabaseManager", *fact_ids: str) -> None:
    """Insert minimal atomic_facts rows so temporal functions can run."""
    db.store_memory(MemoryRecord(memory_id="m0", content="parent"))
    for fid in fact_ids:
        db.store_fact(AtomicFact(
            fact_id=fid, memory_id="m0",
            content=f"fact {fid}", fact_type=FactType.SEMANTIC,
        ))


# ---------------------------------------------------------------------------
# MockServer — same pattern as test_mcp_recall_tool.py
# ---------------------------------------------------------------------------

class _MockServer:
    """Capture @server.tool() decorated functions by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


# ============================================================================
# Group 1: normalize_as_of() unit tests
# ============================================================================

class TestNormalizeAsOf:
    """G1 — UTC normalization helper."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        self.normalize = normalize_as_of

    def test_utc_z_suffix(self):
        """'2024-01-01T12:00:00Z' → '2024-01-01T12:00:00+00:00'."""
        assert self.normalize("2024-01-01T12:00:00Z") == "2024-01-01T12:00:00+00:00"

    def test_positive_offset(self):
        """'2024-01-01T17:30:00+05:30' → '2024-01-01T12:00:00+00:00'."""
        assert self.normalize("2024-01-01T17:30:00+05:30") == "2024-01-01T12:00:00+00:00"

    def test_date_only(self):
        """'2024-01-01' → '2024-01-01T00:00:00+00:00'."""
        assert self.normalize("2024-01-01") == "2024-01-01T00:00:00+00:00"

    def test_naive_datetime_assumed_utc(self):
        """'2024-01-01T12:00:00' (no tz) → assumed UTC."""
        assert self.normalize("2024-01-01T12:00:00") == "2024-01-01T12:00:00+00:00"

    def test_invalid_returns_none(self):
        """'not-a-date' → None."""
        assert self.normalize("not-a-date") is None

    def test_empty_string_returns_none(self):
        """'' → None."""
        assert self.normalize("") is None

    def test_none_returns_none(self):
        """None → None."""
        assert self.normalize(None) is None  # type: ignore[arg-type]

    def test_whitespace_only_returns_none(self):
        """'   ' → None."""
        assert self.normalize("   ") is None

    def test_already_utc_plus_zero_passthrough(self):
        """'2024-06-15T10:30:00+00:00' → same but normalised format."""
        result = self.normalize("2024-06-15T10:30:00+00:00")
        assert result == "2024-06-15T10:30:00+00:00"

    def test_lexicographic_sort_order_vs_sqlite(self):
        """CRIT-1: normalized output date+time portion matches SQLite strftime.

        This test proves the first 19 characters (YYYY-MM-DDTHH:MM:SS) are
        identical between normalize_as_of output and SQLite's strftime output,
        confirming that timezone conversion is correct. The suffix (+00:00) is
        the Python isoformat format — matching stored system_expired_at values
        written by datetime.now(UTC).isoformat().
        """
        import sqlite3
        conn = sqlite3.connect(":memory:")
        ts = "2024-06-15T10:30:00Z"
        norm = self.normalize(ts)
        assert norm is not None
        # SQLite strftime produces YYYY-MM-DDTHH:MM:SSZ
        sqlite_ts = conn.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%SZ', ?)", (ts.replace("Z", ""),)
        ).fetchone()[0]
        # Both must have identical date+time portion (first 19 chars).
        assert norm[:19] == sqlite_ts[:19], (
            f"Timezone mismatch: norm={norm!r}, sqlite={sqlite_ts!r}"
        )
        conn.close()


# ============================================================================
# Group 2: get_invalidated_fact_ids with as_of (transaction-time supersession)
# ============================================================================

@pytestmark_db
class TestGetInvalidatedFactIdsWithAsOf:
    """G2 — as_of parameter gates transaction-time supersession correctly."""

    @pytest.fixture
    def db_superseded(self, tmp_path: Path):
        """DB with fact F1 superseded on 2026-03-01T00:00:00+00:00."""
        db = _make_db(tmp_path)
        _seed_facts(db, "F1")
        db.store_temporal_validity("F1", "default")
        # Directly set system_expired_at to a known past date
        db.execute(
            "UPDATE fact_temporal_validity "
            "SET system_expired_at = '2026-03-01T00:00:00+00:00' "
            "WHERE fact_id = 'F1' AND profile_id = 'default'",
        )
        return db

    def test_no_as_of_returns_superseded(self, db_superseded):
        """Without as_of: fact with system_expired_at IS returned (existing behaviour)."""
        result = db_superseded.get_invalidated_fact_ids(["F1"], "default", as_of=None)
        assert "F1" in result

    def test_as_of_before_supersession_excludes_fact(self, db_superseded):
        """as_of before system_expired_at: fact NOT returned — still valid at that time."""
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        as_of = normalize_as_of("2024-01-01T00:00:00Z")
        result = db_superseded.get_invalidated_fact_ids(["F1"], "default", as_of=as_of)
        assert "F1" not in result, (
            "Fact was not yet superseded at 2024-01-01; should not be demoted."
        )

    def test_as_of_after_supersession_includes_fact(self, db_superseded):
        """as_of after system_expired_at: fact IS returned — supersession already happened."""
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        as_of = normalize_as_of("2026-06-01T00:00:00Z")
        result = db_superseded.get_invalidated_fact_ids(["F1"], "default", as_of=as_of)
        assert "F1" in result, (
            "Fact was superseded on 2026-03-01; should be demoted at 2026-06-01."
        )

    def test_as_of_exactly_at_supersession_includes_fact(self, db_superseded):
        """as_of == system_expired_at (boundary inclusive): fact IS returned."""
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        as_of = normalize_as_of("2026-03-01T00:00:00Z")
        result = db_superseded.get_invalidated_fact_ids(["F1"], "default", as_of=as_of)
        assert "F1" in result, (
            "Boundary is inclusive (<=): fact superseded AT as_of should be demoted."
        )

    def test_crit1_roundtrip_format_via_real_write_path(self, tmp_path: Path):
        """CRIT-1 round-trip: normalize_as_of output compares correctly against
        system_expired_at values stored by invalidate_fact_temporal() which uses
        datetime.now(UTC).isoformat() → 'YYYY-MM-DDTHH:MM:SS.microseconds+00:00'.

        Verifies:
        (a) Stored format ends with '+00:00' (empirical fact).
        (b) Query with as_of BEFORE supersession → fact NOT demoted.
        (c) Query with as_of AFTER supersession → fact IS demoted.
        """
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        db = _make_db(tmp_path)
        _seed_facts(db, "F_rt")
        db.store_temporal_validity("F_rt", "default")
        # Use real write path: invalidate_fact_temporal uses isoformat()
        db.invalidate_fact_temporal("F_rt", "newer_fact", "contradiction")
        # (a) Verify stored format
        tv = db.get_temporal_validity("F_rt", "default")
        assert tv is not None and tv["system_expired_at"] is not None
        stored_ts = tv["system_expired_at"]
        assert stored_ts.endswith("+00:00"), (
            f"CRIT-1: system_expired_at stored as {stored_ts!r}; expected +00:00 suffix. "
            "normalize_as_of must output the SAME suffix format."
        )
        # (b) Query in the far past → fact was valid then
        as_of_before = normalize_as_of("2000-01-01T00:00:00Z")
        r_before = db.get_invalidated_fact_ids(["F_rt"], "default", as_of=as_of_before)
        assert "F_rt" not in r_before, (
            "Fact was not superseded in 2000; should NOT appear in invalidated set."
        )
        # (c) Query in the far future → supersession already happened
        as_of_after = normalize_as_of("2099-01-01T00:00:00Z")
        r_after = db.get_invalidated_fact_ids(["F_rt"], "default", as_of=as_of_after)
        assert "F_rt" in r_after, (
            "Fact IS superseded (now); should appear in invalidated set at 2099."
        )
        # (d) AUDIT P0/CRIT-1 — the killer case the far past/future checks CANNOT
        # catch: round-trip the EXACT microsecond-bearing stored value as as_of.
        # The inclusive boundary (system_expired_at <= as_of) only holds if
        # normalize_as_of PRESERVES microseconds. This is the test that fails if
        # microseconds are stripped.
        assert "." in stored_ts.split("+")[0], (
            "real write path must produce microsecond precision"
        )
        exact = normalize_as_of(stored_ts)
        assert exact == stored_ts, (
            "normalize_as_of must preserve sub-second precision so a real stored "
            "supersession instant round-trips to the inclusive boundary (P0)."
        )
        assert "F_rt" in db.get_invalidated_fact_ids(["F_rt"], "default", as_of=exact), (
            "as_of == exact stored supersession must be inclusive (fact demoted)."
        )
        # ±1s around the real (microsecond-bearing) supersession instant
        from datetime import datetime as _dt, timedelta as _td
        base = _dt.fromisoformat(stored_ts)
        before = normalize_as_of((base - _td(seconds=1)).isoformat())
        after = normalize_as_of((base + _td(seconds=1)).isoformat())
        assert "F_rt" not in db.get_invalidated_fact_ids(["F_rt"], "default", as_of=before), (
            "1s before the supersession → fact still valid → not demoted."
        )
        assert "F_rt" in db.get_invalidated_fact_ids(["F_rt"], "default", as_of=after), (
            "1s after the supersession → fact demoted."
        )


# ============================================================================
# Group 3: valid_until boundary fix (half-open interval <= )
# ============================================================================

@pytestmark_db
class TestValidUntilBoundaryFix:
    """G3 — valid_until <= as_of (not <) correctly implements half-open interval."""

    @pytest.fixture
    def db_boundary(self, tmp_path: Path):
        """Fact F2 with valid_until='2024-06-01T00:00:00+00:00'."""
        db = _make_db(tmp_path)
        _seed_facts(db, "F2")
        db.store_temporal_validity(
            "F2", "default",
            valid_until="2024-06-01T00:00:00+00:00",
        )
        return db

    def test_valid_until_strictly_before_as_of_is_expired(self, db_boundary):
        """valid_until < as_of → fact IS expired. Unambiguous case."""
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        as_of = normalize_as_of("2024-06-02T00:00:00Z")
        result = db_boundary.get_event_time_expired_fact_ids(["F2"], "default", as_of=as_of)
        assert "F2" in result

    def test_valid_until_equals_as_of_is_expired_half_open(self, db_boundary):
        """valid_until == as_of → fact IS expired (half-open interval [valid_from, valid_until)).

        This is the BOUNDARY FIX test. With the old 'valid_until < ?' predicate,
        this returned an empty set. With the fixed '<= ?', it returns {'F2'}.
        """
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        as_of = normalize_as_of("2024-06-01T00:00:00Z")
        result = db_boundary.get_event_time_expired_fact_ids(["F2"], "default", as_of=as_of)
        assert "F2" in result, (
            "BOUNDARY FIX: fact with valid_until == as_of must be expired "
            "(half-open interval excludes the endpoint)."
        )

    def test_valid_until_after_as_of_is_not_expired(self, db_boundary):
        """valid_until > as_of → fact is NOT expired. Unambiguous case."""
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        as_of = normalize_as_of("2024-05-31T00:00:00Z")
        result = db_boundary.get_event_time_expired_fact_ids(["F2"], "default", as_of=as_of)
        assert "F2" not in result


# ============================================================================
# Group 4: TemporalValidityFilter.filter() end-to-end with as_of
# ============================================================================

@pytestmark_db
class TestTemporalValidityFilterWithAsOf:
    """G4 — filter() correctly demotes or preserves based on as_of.

    CRIT-2: tests verify the SCORE of the fact, not just its presence.
    """

    @pytest.fixture
    def db_with_future_supersession(self, tmp_path: Path):
        """F3 superseded on 2026-01-01; as_of=2024-01-01 is BEFORE that."""
        db = _make_db(tmp_path)
        _seed_facts(db, "F3", "F_valid")
        db.store_temporal_validity("F3", "default")
        db.execute(
            "UPDATE fact_temporal_validity "
            "SET system_expired_at = '2026-01-01T00:00:00+00:00' "
            "WHERE fact_id = 'F3' AND profile_id = 'default'",
        )
        db.store_temporal_validity("F_valid", "default")
        return db

    def test_as_of_before_supersession_score_unchanged(
        self, db_with_future_supersession
    ):
        """CRIT-2: as_of BEFORE supersession → F3 score is the ORIGINAL score (not × 0.25).

        This specifically checks the score (not just that F3 is in the results).
        """
        from superlocalmemory.retrieval.temporal_validity_filter import TemporalValidityFilter

        filt = TemporalValidityFilter(db_with_future_supersession, demotion_factor=0.25)
        all_results = {
            "semantic": [("F3", 0.8), ("F_valid", 0.9)],
        }
        context = {"as_of": "2024-01-01T00:00:00+00:00"}
        filtered = filt.filter(all_results, "default", context)

        scores = dict(filtered["semantic"])
        # F3 was NOT superseded at as_of=2024-01-01; score must be UNCHANGED.
        assert scores["F3"] == pytest.approx(0.8), (
            "CRIT-2: F3 was valid at as_of; score must not be multiplied by demotion factor."
        )
        assert scores["F_valid"] == pytest.approx(0.9)

    def test_as_of_after_supersession_score_demoted(
        self, db_with_future_supersession
    ):
        """as_of AFTER supersession → F3 score is multiplied by demotion factor."""
        from superlocalmemory.retrieval.temporal_validity_filter import TemporalValidityFilter

        filt = TemporalValidityFilter(db_with_future_supersession, demotion_factor=0.25)
        all_results = {
            "semantic": [("F3", 0.8), ("F_valid", 0.9)],
        }
        context = {"as_of": "2026-06-01T00:00:00+00:00"}
        filtered = filt.filter(all_results, "default", context)

        scores = dict(filtered["semantic"])
        assert scores["F3"] == pytest.approx(0.8 * 0.25), (
            "F3 was superseded at as_of; score must be multiplied by demotion factor."
        )

    def test_no_as_of_superseded_fact_still_demoted(
        self, db_with_future_supersession
    ):
        """Regression: without as_of, existing supersession behaviour is unchanged."""
        from superlocalmemory.retrieval.temporal_validity_filter import TemporalValidityFilter

        filt = TemporalValidityFilter(db_with_future_supersession, demotion_factor=0.25)
        all_results = {"semantic": [("F3", 0.8), ("F_valid", 0.9)]}
        # context=None means no as_of
        filtered = filt.filter(all_results, "default", None)
        scores = dict(filtered["semantic"])
        # F3 has system_expired_at set → always demoted when as_of=None
        assert scores["F3"] == pytest.approx(0.8 * 0.25)


# ============================================================================
# Group 5: recall_trace MCP tool with as_of
# ============================================================================

class TestRecallTraceMcpWithAsOf:
    """G5 — recall_trace MCP: as_of normalisation and error handling."""

    @pytest.fixture(autouse=True)
    def _inline_thread(self, monkeypatch):
        """Run asyncio.to_thread inline so tests don't spawn threads."""
        async def _run_inline(fn, *args, **kwargs):
            return fn(*args, **kwargs)
        monkeypatch.setattr(asyncio, "to_thread", _run_inline)

    def _get_recall_trace(self):
        from superlocalmemory.mcp.tools_v3 import register_v3_tools
        srv = _MockServer()
        get_engine = MagicMock()
        register_v3_tools(srv, get_engine)
        return srv._tools["recall_trace"]

    async def test_recall_trace_valid_as_of_no_error(self):
        """Valid as_of → success=True, no error field."""
        recall_trace = self._get_recall_trace()
        pool = MagicMock()
        pool.recall.return_value = {"results": [], "ok": True}
        # choose_pool is imported inline inside recall_trace's lambda; patch the
        # module where it lives so the inner lambda picks up the mock.
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            return_value=pool,
        ):
            result = await recall_trace(
                query="test", limit=5, as_of="2024-01-01T00:00:00Z"
            )
        assert result.get("error") != "invalid_as_of"
        assert result.get("success") is True

    async def test_recall_trace_invalid_as_of_returns_error(self):
        """Invalid as_of → {'success': False, 'error': 'invalid_as_of'}."""
        recall_trace = self._get_recall_trace()
        result = await recall_trace(query="test", limit=5, as_of="not-a-date")
        assert result.get("success") is False
        assert result.get("error") == "invalid_as_of"

    async def test_recall_trace_no_as_of_unchanged(self):
        """No as_of → same behaviour as before Phase 4b (regression guard)."""
        recall_trace = self._get_recall_trace()
        pool = MagicMock()
        pool.recall.return_value = {
            "results": [{"fact_id": "f1", "content": "c", "score": 0.9}],
            "ok": True,
        }
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool
        ):
            result = await recall_trace(query="hello world", limit=3)
        assert result.get("success") is True

    async def test_recall_trace_forwards_strict_two_clock_boundaries(self):
        recall_trace = self._get_recall_trace()
        pool = MagicMock()
        pool.recall.return_value = {"results": [], "ok": True}
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool,
        ):
            result = await recall_trace(
                query="historical", known_as_of="2026-01-01T00:00:00Z",
                valid_at="2025-01-01T00:00:00Z", include_unknown=True,
            )
        assert result["success"] is True
        assert pool.recall.call_args.kwargs["known_as_of"] == "2026-01-01T00:00:00+00:00"
        assert pool.recall.call_args.kwargs["valid_at"] == "2025-01-01T00:00:00+00:00"
        assert pool.recall.call_args.kwargs["include_unknown"] is True


# ============================================================================
# Group 6: recall_trace HTTP endpoint with as_of
# ============================================================================

class TestRecallTraceHttpWithAsOf:
    """G6 — POST /recall/trace: as_of from request body normalised and validated."""

    def _make_request(self, body: dict):
        """Minimal starlette-like mock request."""
        class _MockApp:
            class state:
                pass

        class _MockRequest:
            app = _MockApp()

            async def json(self_inner):
                return body

        return _MockRequest()

    async def test_invalid_as_of_returns_400(self):
        """POST /recall/trace with invalid as_of → JSONResponse(400)."""
        from superlocalmemory.server.routes.v3_api import recall_trace
        from starlette.responses import JSONResponse
        request = self._make_request({"query": "hello", "as_of": "not-a-date"})
        response = await recall_trace(request)
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        data = json.loads(response.body)
        assert data.get("error") == "invalid_as_of"

    async def test_no_as_of_proceeds_normally(self):
        """POST /recall/trace without as_of continues to engine (no 400)."""
        from superlocalmemory.server.routes.v3_api import recall_trace
        from starlette.responses import JSONResponse
        request = self._make_request({"query": "hello"})
        # get_engine_lazy is imported locally inside the route function from
        # .helpers; patch it there.  A None engine → 503 (not 400), confirming
        # the as_of validation path was not triggered.
        with patch(
            "superlocalmemory.server.routes.helpers.get_engine_lazy",
            return_value=None,
        ):
            response = await recall_trace(request)
        # None engine → 503, not 400
        if isinstance(response, JSONResponse):
            assert response.status_code != 400
        else:
            assert response.get("error") != "invalid_as_of"

    async def test_invalid_strict_json_types_return_400(self):
        from superlocalmemory.server.routes.v3_api import recall_trace
        from starlette.responses import JSONResponse
        request = self._make_request({"query": "hello", "known_as_of": 42})
        response = await recall_trace(request)
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        assert json.loads(response.body)["error"] == "invalid_known_as_of"

    async def test_include_unknown_requires_json_boolean(self):
        from superlocalmemory.server.routes.v3_api import recall_trace
        from starlette.responses import JSONResponse
        request = self._make_request({"query": "hello", "include_unknown": "false"})
        response = await recall_trace(request)
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        assert json.loads(response.body)["error"] == "invalid_include_unknown"


# ============================================================================
# Group 7: prestage_context with as_of (including CRIT-3: 3-arg fallback)
# ============================================================================

class TestPrestageContextWithAsOf:
    """G7 — prestage_context: as_of threaded, normalised, and 3-arg fallback safe."""

    def test_as_of_forwarded_to_4arg_recall_fn(self):
        """prestage_context with valid as_of calls recall_fn with normalized as_of."""
        from superlocalmemory.mcp.tools_context import prestage_context

        received: list = []

        def recall_fn_4arg(q, lim, pid, as_of=None):
            received.append({"q": q, "lim": lim, "pid": pid, "as_of": as_of})
            return []

        result = prestage_context(
            "hello world",
            limit=3,
            profile_id="default",
            session_id="s1",
            recall_fn=recall_fn_4arg,
            as_of="2024-01-01T00:00:00Z",
        )
        assert result.get("error") is None
        assert len(received) == 1
        # as_of must be normalized to +00:00 format
        assert received[0]["as_of"] == "2024-01-01T00:00:00+00:00"

    def test_invalid_as_of_silently_ignored_warns(self):
        """prestage_context with invalid as_of: recall_fn called with as_of=None.

        prestage is background, so invalid as_of is silently ignored (just logged).
        """
        from superlocalmemory.mcp.tools_context import prestage_context

        received: list = []

        def recall_fn_4arg(q, lim, pid, as_of=None):
            received.append(as_of)
            return []

        result = prestage_context(
            "hello",
            limit=2,
            profile_id="default",
            session_id="s1",
            recall_fn=recall_fn_4arg,
            as_of="garbage",
        )
        # No crash; recall_fn was still called
        assert result.get("error") is None
        assert len(received) == 1
        # invalid as_of → None passed to recall_fn
        assert received[0] is None

    def test_no_as_of_unchanged(self):
        """prestage_context without as_of: identical to pre-Phase4 behaviour."""
        from superlocalmemory.mcp.tools_context import prestage_context

        received: list = []

        def recall_fn_3arg(q, lim, pid):
            received.append((q, lim, pid))
            return [{"text": "mem", "score": 0.9, "id": "1", "source": "recall"}]

        result = prestage_context(
            "hello",
            limit=2,
            profile_id="default",
            session_id="s1",
            recall_fn=recall_fn_3arg,
        )
        assert len(received) == 1
        assert result["memories"] != [] or result.get("error") is None

    def test_crit3_3arg_recall_fn_with_as_of_fallback(self):
        """CRIT-3: a 3-arg recall_fn does NOT crash when as_of is supplied.

        Option A2 try/except TypeError fallback: when recall_fn only accepts 3
        positional args, the 4-arg call raises TypeError. The fallback silently
        drops as_of and calls the 3-arg form. No crash allowed.
        """
        from superlocalmemory.mcp.tools_context import prestage_context

        call_count = [0]

        def recall_fn_3arg(q, lim, pid):
            # This is a legacy 3-arg recall_fn
            call_count[0] += 1
            return []

        # Must NOT raise TypeError; must call recall_fn successfully
        result = prestage_context(
            "query",
            limit=2,
            profile_id="default",
            session_id="s1",
            recall_fn=recall_fn_3arg,
            as_of="2024-01-01T00:00:00Z",  # as_of present but fn only takes 3 args
        )
        assert call_count[0] == 1, "recall_fn must be called exactly once (fallback)"
        assert result.get("error") is None

    def test_4arg_internal_typeerror_not_swallowed_as_3arg(self):
        """AUDIT P2/CRIT-3: a genuine TypeError raised INSIDE a 4-arg recall_fn
        must NOT be silently swallowed and retried as a 3-arg call (which would
        drop as_of and give a wrong point-in-time view). With arity inspection
        the fn is invoked exactly once and the error surfaces honestly.
        """
        from superlocalmemory.mcp.tools_context import prestage_context

        call_count = [0]

        def recall_fn_4arg(q, lim, pid, as_of=None):
            call_count[0] += 1
            raise TypeError("internal bug, NOT an arity mismatch")

        result = prestage_context(
            "query",
            limit=2,
            profile_id="default",
            session_id="s1",
            recall_fn=recall_fn_4arg,
            as_of="2024-01-01T00:00:00Z",
        )
        assert call_count[0] == 1, "4-arg fn called exactly once (no silent 3-arg retry)"
        assert result.get("error") == "recall_error"


# ============================================================================
# Group 8: UTC normalization at HTTP / MCP recall boundaries
# ============================================================================

class TestBoundaryNormalization:
    """G8 — invalid as_of is rejected at HTTP and MCP recall boundaries."""

    @pytest.fixture(autouse=True)
    def _inline_thread(self, monkeypatch):
        """Run asyncio.to_thread inline."""
        async def _run_inline(fn, *args, **kwargs):
            return fn(*args, **kwargs)
        monkeypatch.setattr(asyncio, "to_thread", _run_inline)

    def _get_recall_tool(self):
        from superlocalmemory.mcp.tools_core import register_core_tools
        srv = _MockServer()
        get_engine = MagicMock()
        register_core_tools(srv, get_engine)
        return srv._tools["recall"]

    async def test_mcp_recall_invalid_as_of_returns_error(self):
        """MCP recall tool with as_of='not-a-date' → {'success': False, 'error': 'invalid_as_of'}."""
        recall = self._get_recall_tool()
        result = await recall(query="test", as_of="not-a-date")
        assert result.get("success") is False
        assert result.get("error") == "invalid_as_of"

    async def test_mcp_recall_valid_as_of_not_rejected(self):
        """MCP recall tool with valid as_of passes normalization check."""
        recall = self._get_recall_tool()
        pool = MagicMock()
        pool.recall.return_value = {"ok": True, "results": []}
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool
        ):
            result = await recall(query="test", as_of="2024-01-01T00:00:00Z")
        # Should not have error="invalid_as_of"
        assert result.get("error") != "invalid_as_of"

    async def test_http_recall_invalid_as_of_returns_error(self):
        """Verify normalize_as_of rejects 'baddate' at boundary.

        We test the helper directly because the HTTP recall endpoint wires
        normalize_as_of in a large handler. The specific rejection behavior
        is verified through the helper function.
        """
        from superlocalmemory.retrieval.temporal_utils import normalize_as_of
        result = normalize_as_of("baddate")
        assert result is None, (
            "normalize_as_of must return None for invalid input so the "
            "HTTP handler can return an error response."
        )
