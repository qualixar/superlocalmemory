# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Unit tests for the Google Calendar ingestion adapter.

Covers tier detection, the ICS file importer (Tier 1) via both the icalendar
library path and the dependency-free basic parser fallback, the OAuth
Calendar-API poller (Tier 2) including sync-token handling and 410 re-sync,
attendee entity propagation, and dedup_key generation.

External I/O is mocked: google-api-python-client / google-auth are injected as
fake modules via sys.modules, and the icalendar import is forced to fail (by
setting sys.modules['icalendar'] = None) to exercise the basic parser branch.
No real credentials, sockets, or files outside tmp_path are touched.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.ingestion import calendar_adapter
from superlocalmemory.ingestion.base_adapter import AdapterConfig
from superlocalmemory.ingestion.calendar_adapter import CalendarAdapter


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

_ICS_ONE_EVENT = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//SLM Test//EN
BEGIN:VEVENT
UID:evt-1@test
SUMMARY:Design Review
DTSTART:20260720T100000Z
DTEND:20260720T110000Z
LOCATION:Room 5
DESCRIPTION:Discuss the ingestion architecture
ATTENDEE:mailto:alice@example.com
ATTENDEE:mailto:bob@example.com
END:VEVENT
END:VCALENDAR
"""


def _write_ics(path: Path, text: str = _ICS_ONE_EVENT) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _install_fake_google(monkeypatch, service: MagicMock) -> MagicMock:
    """Inject fake google-auth + googleapiclient modules returning ``service``."""
    google_mod = types.ModuleType("google")
    oauth2_mod = types.ModuleType("google.oauth2")
    creds_mod = types.ModuleType("google.oauth2.credentials")
    creds_mod.Credentials = MagicMock(name="Credentials")
    gac_mod = types.ModuleType("googleapiclient")
    disc_mod = types.ModuleType("googleapiclient.discovery")
    disc_mod.build = MagicMock(name="build", return_value=service)

    for name, mod in (
        ("google", google_mod),
        ("google.oauth2", oauth2_mod),
        ("google.oauth2.credentials", creds_mod),
        ("googleapiclient", gac_mod),
        ("googleapiclient.discovery", disc_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return disc_mod.build


def _patch_credentials(monkeypatch, values: dict) -> None:
    monkeypatch.setattr(
        "superlocalmemory.ingestion.credentials.load_credential",
        lambda service, key: values.get(key),
    )


# ---------------------------------------------------------------------------
# _detect_tier
# ---------------------------------------------------------------------------

class TestDetectTier:
    def test_non_auto_tier_is_left_untouched(self):
        adapter = CalendarAdapter(tier="oauth")
        adapter._detect_tier()
        assert adapter._tier == "oauth"

    def test_explicit_ics_tier_from_config(self, monkeypatch, tmp_path):
        cfg = tmp_path / "adapters.json"
        cfg.write_text(json.dumps(
            {"calendar": {"tier": "ics", "ics_path": "/cal/export.ics"}}))
        monkeypatch.setattr(calendar_adapter, "_adapters_config_path", lambda: cfg)

        adapter = CalendarAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "ics"
        assert adapter._ics_path == "/cal/export.ics"

    def test_oauth_selected_when_gmail_refresh_token_present(self, monkeypatch, tmp_path):
        cfg = tmp_path / "adapters.json"
        cfg.write_text(json.dumps({"calendar": {}}))
        monkeypatch.setattr(calendar_adapter, "_adapters_config_path", lambda: cfg)
        monkeypatch.setattr(
            "superlocalmemory.ingestion.credentials.has_credential",
            lambda service, key: service == "gmail" and key == "refresh_token",
        )

        adapter = CalendarAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "oauth"

    def test_ics_file_discovered_in_import_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            calendar_adapter, "_adapters_config_path", lambda: tmp_path / "absent.json")
        monkeypatch.setattr(
            "superlocalmemory.ingestion.credentials.has_credential",
            lambda service, key: False,
        )
        import_dir = tmp_path / "import"
        import_dir.mkdir()
        found = _write_ics(import_dir / "cal.ics")
        monkeypatch.setattr(calendar_adapter, "_import_dir", lambda: import_dir)

        adapter = CalendarAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "ics"
        assert adapter._ics_path == str(found)

    def test_default_falls_back_to_ics(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            calendar_adapter, "_adapters_config_path", lambda: tmp_path / "absent.json")
        monkeypatch.setattr(
            "superlocalmemory.ingestion.credentials.has_credential",
            lambda service, key: False,
        )
        empty_dir = tmp_path / "import"
        empty_dir.mkdir()
        monkeypatch.setattr(calendar_adapter, "_import_dir", lambda: empty_dir)

        adapter = CalendarAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "ics"
        assert adapter._ics_path is None


# ---------------------------------------------------------------------------
# fetch_items dispatch
# ---------------------------------------------------------------------------

class TestFetchItemsDispatch:
    def test_dispatches_to_active_tier(self, monkeypatch):
        adapter = CalendarAdapter(tier="ics")
        monkeypatch.setattr(adapter, "_fetch_ics", lambda: ["ICS"])
        monkeypatch.setattr(adapter, "_fetch_oauth", lambda: ["OAUTH"])

        assert adapter.fetch_items() == ["ICS"]
        adapter._tier = "oauth"
        assert adapter.fetch_items() == ["OAUTH"]

    def test_unknown_tier_returns_empty(self):
        adapter = CalendarAdapter(tier="nonsense")
        assert adapter.fetch_items() == []


class TestWaitForNextCycle:
    def test_ics_import_self_terminates(self):
        adapter = CalendarAdapter(tier="ics")
        adapter._ics_processed = True
        adapter.wait_for_next_cycle()
        assert adapter._stop_event.is_set()

    def test_polling_tier_waits_poll_interval(self, monkeypatch):
        adapter = CalendarAdapter(tier="oauth")
        waited: list[float] = []
        monkeypatch.setattr(adapter._stop_event, "wait", waited.append)

        adapter.wait_for_next_cycle()

        assert waited == [adapter._poll_interval]
        assert not adapter._stop_event.is_set()


# ---------------------------------------------------------------------------
# _fetch_ics — icalendar library path
# ---------------------------------------------------------------------------

class TestFetchIcsIcalendar:
    def test_single_attendee_is_normalized_to_list(self, tmp_path):
        pytest.importorskip("icalendar")
        ics = _write_ics(tmp_path / "solo.ics", text=(
            "BEGIN:VCALENDAR\n"
            "VERSION:2.0\n"
            "PRODID:-//T//EN\n"
            "BEGIN:VEVENT\n"
            "UID:solo-1\n"
            "SUMMARY:One on One\n"
            "DTSTART:20260720T100000Z\n"
            "ATTENDEE:mailto:solo@example.com\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        ))
        adapter = CalendarAdapter(tier="ics")
        adapter._ics_path = str(ics)

        items = adapter._fetch_ics()

        # 1 event + 1 attendee-propagation item (single ATTENDEE, not a list).
        assert len(items) == 2
        assert items[0].metadata["attendees"] == ["solo"]
        assert items[1].metadata["entity_name"] == "solo"

    def test_parses_event_with_attendee_propagation(self, tmp_path):
        pytest.importorskip("icalendar")
        ics = _write_ics(tmp_path / "cal.ics")
        adapter = CalendarAdapter(tier="ics")
        adapter._ics_path = str(ics)

        items = adapter._fetch_ics()

        # 1 event + 2 attendee entity-propagation items.
        assert len(items) == 3
        event = items[0]
        assert "Calendar Event: Design Review" in event.content
        assert "Where: Room 5" in event.content
        assert "Discuss the ingestion architecture" in event.content
        # dedup_key = "{uid}-{start_iso}"
        assert event.dedup_key == "evt-1@test-2026-07-20T10:00:00+00:00"
        assert event.metadata["source"] == "ics_import"
        assert event.metadata["attendees"] == ["alice", "bob"]

        propagated = items[1:]
        names = {i.metadata["entity_name"] for i in propagated}
        assert names == {"alice", "bob"}
        for prop in propagated:
            assert prop.metadata["source"] == "entity_propagation"
            assert prop.dedup_key.startswith("attendee-")
            assert "Design Review" in prop.content

    def test_marks_processed_and_is_idempotent(self, tmp_path):
        pytest.importorskip("icalendar")
        ics = _write_ics(tmp_path / "cal.ics")
        adapter = CalendarAdapter(tier="ics")
        adapter._ics_path = str(ics)

        assert len(adapter._fetch_ics()) == 3
        assert adapter._ics_processed is True
        assert adapter._fetch_ics() == []

    def test_no_path_returns_empty(self):
        adapter = CalendarAdapter(tier="ics")
        adapter._ics_path = None
        assert adapter._fetch_ics() == []

    def test_missing_file_returns_empty_and_marks_processed(self, tmp_path):
        adapter = CalendarAdapter(tier="ics")
        adapter._ics_path = str(tmp_path / "nope.ics")

        assert adapter._fetch_ics() == []
        assert adapter._ics_processed is True


# ---------------------------------------------------------------------------
# _fetch_ics — basic parser fallback (icalendar unavailable)
# ---------------------------------------------------------------------------

class TestFetchIcsBasicFallback:
    def test_import_error_falls_back_to_basic_parser(self, monkeypatch, tmp_path):
        ics = _write_ics(tmp_path / "cal.ics")
        # Force `from icalendar import Calendar` to raise ImportError.
        monkeypatch.setitem(sys.modules, "icalendar", None)

        adapter = CalendarAdapter(tier="ics")
        adapter._ics_path = str(ics)
        items = adapter._fetch_ics()

        # Basic parser does not expand attendees -> exactly one item.
        assert len(items) == 1
        assert items[0].metadata["source"] == "ics_basic"
        assert items[0].dedup_key == "evt-1@test-20260720T100000Z"
        assert "Calendar Event: Design Review" in items[0].content
        assert adapter._ics_processed is True

    def test_basic_parser_directly(self, tmp_path):
        ics = _write_ics(tmp_path / "cal.ics", text=(
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:x-1\n"
            "SUMMARY:Standup\n"
            "DTSTART;TZID=UTC:20260721T090000\n"
            "END:VEVENT\n"
            "BEGIN:VEVENT\n"
            "SUMMARY:No UID Event\n"
            "DTSTART:20260722T090000Z\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        ))
        adapter = CalendarAdapter(tier="ics")

        items = adapter._parse_ics_basic(ics)

        assert len(items) == 2
        # DTSTART;TZID=UTC:... -> the ";" params are stripped from the key.
        assert items[0].dedup_key == "x-1-20260721T090000"
        assert items[0].metadata["summary"] == "Standup"
        assert items[0].metadata["source"] == "ics_basic"
        # Missing UID -> synthetic "basic-{index}" key.
        assert items[1].dedup_key == "basic-1-20260722T090000Z"


# ---------------------------------------------------------------------------
# _fetch_oauth (Tier 2)
# ---------------------------------------------------------------------------

def _calendar_event(**overrides) -> dict:
    event = {
        "summary": "Sprint Planning",
        "start": {"dateTime": "2026-07-20T10:00:00Z"},
        "end": {"dateTime": "2026-07-20T11:00:00Z"},
        "description": "plan the sprint",
        "location": "Zoom",
        "id": "ev1",
        "updated": "2026-07-19T00:00:00Z",
        "attendees": [{"displayName": "Alice"}, {"email": "bob@corp.com"}],
    }
    event.update(overrides)
    return event


class TestFetchOauth:
    def test_incomplete_credentials_returns_empty(self, monkeypatch):
        _patch_credentials(monkeypatch, {"refresh_token": "rt"})
        adapter = CalendarAdapter(tier="oauth")
        assert adapter._fetch_oauth() == []

    def test_import_error_returns_empty(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        monkeypatch.setitem(sys.modules, "googleapiclient.discovery", None)
        adapter = CalendarAdapter(tier="oauth")
        assert adapter._fetch_oauth() == []

    def test_initial_sync_parses_events_and_attendees(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        service.events.return_value.list.return_value.execute.return_value = {
            "items": [_calendar_event()],
            "nextSyncToken": "sync-token-1",
        }
        _install_fake_google(monkeypatch, service)

        adapter = CalendarAdapter(tier="oauth")
        items = adapter._fetch_oauth()

        # 1 event + 2 attendee-propagation items.
        assert len(items) == 3
        event = items[0]
        assert event.dedup_key == "ev1-2026-07-19T00:00:00Z"
        assert "Calendar Event: Sprint Planning" in event.content
        assert "Where: Zoom" in event.content
        assert event.metadata["attendees"] == ["Alice", "bob"]
        assert event.metadata["source"] == "oauth"

        names = {i.metadata["entity_name"] for i in items[1:]}
        assert names == {"Alice", "bob"}
        assert adapter._sync_token == "sync-token-1"

    def test_incremental_sync_passes_sync_token(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        list_call = service.events.return_value.list
        list_call.return_value.execute.return_value = {
            "items": [], "nextSyncToken": "sync-token-2",
        }
        _install_fake_google(monkeypatch, service)

        adapter = CalendarAdapter(tier="oauth")
        adapter._sync_token = "prev-token"
        items = adapter._fetch_oauth()

        assert items == []
        # The stored sync token drives an incremental query (no timeMin).
        _, kwargs = list_call.call_args
        assert kwargs["syncToken"] == "prev-token"
        assert "timeMin" not in kwargs
        assert adapter._sync_token == "sync-token-2"

    def test_expired_sync_token_triggers_full_resync(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        list_call = service.events.return_value.list
        # First call: 410 Gone. Second (recursive full re-sync): succeeds.
        list_call.return_value.execute.side_effect = [
            Exception("<HttpError 410 when requesting ... Sync token is no longer valid>"),
            {"items": [_calendar_event(id="ev2", updated="u2", attendees=[])],
             "nextSyncToken": "fresh-token"},
        ]
        _install_fake_google(monkeypatch, service)

        adapter = CalendarAdapter(tier="oauth")
        adapter._sync_token = "stale-token"
        items = adapter._fetch_oauth()

        assert list_call.return_value.execute.call_count == 2
        assert len(items) == 1
        assert items[0].dedup_key == "ev2-u2"
        assert adapter._sync_token == "fresh-token"

    def test_non_410_api_error_returns_empty(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        (service.events.return_value.list.return_value
         .execute.side_effect) = RuntimeError("<HttpError 500 server error>")
        _install_fake_google(monkeypatch, service)

        adapter = CalendarAdapter(tier="oauth")
        # A non-410 error propagates to the outer guard and yields no items.
        assert adapter._fetch_oauth() == []

    def test_malformed_event_is_skipped(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        good = _calendar_event(id="good", updated="u", attendees=[])
        bad = _calendar_event(id="bad", updated="u2", attendees=["not-a-dict"])
        service.events.return_value.list.return_value.execute.return_value = {
            "items": [bad, good], "nextSyncToken": "tok",
        }
        _install_fake_google(monkeypatch, service)

        adapter = CalendarAdapter(tier="oauth")
        items = adapter._fetch_oauth()

        keys = [i.dedup_key for i in items]
        assert "good-u" in keys
        assert "bad-u2" not in keys

    def test_stop_event_halts_oauth_poll(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        service.events.return_value.list.return_value.execute.return_value = {
            "items": [_calendar_event()], "nextSyncToken": "tok",
        }
        _install_fake_google(monkeypatch, service)

        adapter = CalendarAdapter(tier="oauth")
        adapter.stop()  # shutdown before the per-event loop runs
        assert adapter._fetch_oauth() == []


# ---------------------------------------------------------------------------
# Stop-event during ICS parse + full run() loop lifecycle
# ---------------------------------------------------------------------------

class TestIcsShutdownAndRunLoop:
    def test_stop_event_halts_ics_parsing(self, tmp_path):
        pytest.importorskip("icalendar")
        ics = _write_ics(tmp_path / "cal.ics")
        adapter = CalendarAdapter(tier="ics")
        adapter._ics_path = str(ics)
        adapter.stop()  # set before parsing begins

        assert adapter._fetch_ics() == []
        assert adapter._ics_processed is True

    def test_run_loop_ingests_ics_then_self_stops(self, monkeypatch, tmp_path):
        """Full ingestion path: run() fetches the ICS, ingests every item
        (event + attendee propagation), then the ICS tier stops itself."""
        pytest.importorskip("icalendar")
        ics = _write_ics(tmp_path / "run.ics")
        daemon_request = MagicMock(return_value={"ingested": True})
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request", daemon_request)
        monkeypatch.setattr("psutil.pid_exists", lambda pid: True)

        adapter = CalendarAdapter(
            config=AdapterConfig(batch_delay_sec=0, rate_limit_per_hour=100),
            tier="ics",
        )
        adapter._ics_path = str(ics)

        adapter.run()  # terminates: ICS tier calls stop() after processing

        # 1 event + 2 attendee-propagation items.
        assert adapter._total_ingested == 3
        assert daemon_request.call_count == 3
        assert adapter._stop_event.is_set()
