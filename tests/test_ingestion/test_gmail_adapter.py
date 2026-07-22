# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Unit tests for the Gmail ingestion adapter.

Covers tier detection, the .mbox file importer (Tier 1), the IMAP poller
(Tier 1.5), the OAuth Gmail-API poller (Tier 2), Gmail-API body extraction,
and the BaseAdapter rate limiter as exercised through the ingestion path.

All external I/O is mocked:
  - imaplib.IMAP4_SSL is replaced with an in-memory fake connection.
  - google-api-python-client / google-auth are injected as fake modules via
    sys.modules so the tests run whether or not the optional deps are present.
  - the daemon /ingest call is patched at its source module.

No real credentials, sockets, or files outside tmp_path are touched.
"""

from __future__ import annotations

import base64
import json
import mailbox
import sys
import types
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import MagicMock

from superlocalmemory.ingestion import gmail_adapter
from superlocalmemory.ingestion.base_adapter import AdapterConfig, IngestItem
from superlocalmemory.ingestion.gmail_adapter import GmailAdapter


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _write_mbox(path: Path, messages: list[dict]) -> Path:
    """Generate a real .mbox file (as Google Takeout exports)."""
    box = mailbox.mbox(str(path))
    box.lock()
    try:
        for spec in messages:
            msg = EmailMessage()
            for header in ("Message-ID", "Subject", "From", "Date"):
                if spec.get(header) is not None:
                    msg[header] = spec[header]
            msg.set_content(spec.get("body", ""))
            if spec.get("html"):
                msg.add_alternative(spec["html"], subtype="html")
            box.add(msg)
        box.flush()
    finally:
        box.unlock()
        box.close()
    return path


def _raw_email_bytes(subject: str, from_addr: str, body: str,
                     message_id: str | None = None) -> bytes:
    """Build an RFC822 byte payload for the IMAP fetch fake."""
    msg = EmailMessage()
    if message_id is not None:
        msg["Message-ID"] = message_id
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg.set_content(body)
    return msg.as_bytes()


def _install_fake_google(monkeypatch, service: MagicMock) -> MagicMock:
    """Inject fake google-auth + googleapiclient modules returning ``service``.

    Works regardless of whether the real optional dependencies are installed,
    because monkeypatch.setitem overrides sys.modules for the test only.
    """
    credentials_cls = MagicMock(name="Credentials")

    google_mod = types.ModuleType("google")
    oauth2_mod = types.ModuleType("google.oauth2")
    creds_mod = types.ModuleType("google.oauth2.credentials")
    creds_mod.Credentials = credentials_cls
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
    """Patch load_credential at its source module (imported lazily)."""
    def _load(service: str, key: str):
        return values.get(key)

    monkeypatch.setattr(
        "superlocalmemory.ingestion.credentials.load_credential", _load,
    )


# ---------------------------------------------------------------------------
# _detect_tier
# ---------------------------------------------------------------------------

class TestDetectTier:
    def test_non_auto_tier_is_left_untouched(self):
        adapter = GmailAdapter(tier="imap")
        adapter._detect_tier()
        assert adapter._tier == "imap"

    def test_explicit_mbox_tier_from_config(self, monkeypatch, tmp_path):
        cfg = tmp_path / "adapters.json"
        cfg.write_text(json.dumps(
            {"gmail": {"tier": "mbox", "mbox_path": "/data/takeout.mbox"}}))
        monkeypatch.setattr(gmail_adapter, "_adapters_config_path", lambda: cfg)

        adapter = GmailAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "mbox"
        assert adapter._mbox_path == "/data/takeout.mbox"

    def test_mbox_path_alone_selects_mbox(self, monkeypatch, tmp_path):
        cfg = tmp_path / "adapters.json"
        cfg.write_text(json.dumps({"gmail": {"mbox_path": "/x/y.mbox"}}))
        monkeypatch.setattr(gmail_adapter, "_adapters_config_path", lambda: cfg)

        adapter = GmailAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "mbox"
        assert adapter._mbox_path == "/x/y.mbox"

    def test_explicit_imap_tier_from_config(self, monkeypatch, tmp_path):
        cfg = tmp_path / "adapters.json"
        cfg.write_text(json.dumps({"gmail": {"tier": "imap"}}))
        monkeypatch.setattr(gmail_adapter, "_adapters_config_path", lambda: cfg)

        adapter = GmailAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "imap"

    def test_oauth_selected_when_refresh_token_present(self, monkeypatch, tmp_path):
        cfg = tmp_path / "adapters.json"  # exists but empty gmail section
        cfg.write_text(json.dumps({"gmail": {}}))
        monkeypatch.setattr(gmail_adapter, "_adapters_config_path", lambda: cfg)
        monkeypatch.setattr(
            "superlocalmemory.ingestion.credentials.has_credential",
            lambda service, key: service == "gmail" and key == "refresh_token",
        )

        adapter = GmailAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "oauth"

    def test_mbox_file_discovered_in_import_dir(self, monkeypatch, tmp_path):
        # No config file, no credentials — fall through to import-dir glob.
        monkeypatch.setattr(
            gmail_adapter, "_adapters_config_path", lambda: tmp_path / "absent.json")
        monkeypatch.setattr(
            "superlocalmemory.ingestion.credentials.has_credential",
            lambda service, key: False,
        )
        import_dir = tmp_path / "import"
        import_dir.mkdir()
        found = import_dir / "archive.mbox"
        found.write_text("From dummy\n")
        monkeypatch.setattr(gmail_adapter, "_import_dir", lambda: import_dir)

        adapter = GmailAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "mbox"
        assert adapter._mbox_path == str(found)

    def test_default_falls_back_to_empty_mbox(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            gmail_adapter, "_adapters_config_path", lambda: tmp_path / "absent.json")
        monkeypatch.setattr(
            "superlocalmemory.ingestion.credentials.has_credential",
            lambda service, key: False,
        )
        empty_dir = tmp_path / "import"
        empty_dir.mkdir()
        monkeypatch.setattr(gmail_adapter, "_import_dir", lambda: empty_dir)

        adapter = GmailAdapter(tier="auto")
        adapter._detect_tier()

        assert adapter._tier == "mbox"
        assert adapter._mbox_path is None


# ---------------------------------------------------------------------------
# _fetch_mbox (Tier 1)
# ---------------------------------------------------------------------------

class TestFetchMbox:
    def test_parses_single_and_multipart_messages(self, tmp_path):
        mbox_path = _write_mbox(tmp_path / "t.mbox", [
            {"Message-ID": "<abc@mail>", "Subject": "Hello",
             "From": "alice@x.com", "Date": "Mon, 20 Jul 2026 10:00:00 +0000",
             "body": "plain body one"},
            {"Message-ID": "<def@mail>", "Subject": "Multi",
             "From": "bob@y.com", "body": "the text part",
             "html": "<html>ignored</html>"},
        ])
        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_path = str(mbox_path)

        items = adapter._fetch_mbox()

        assert len(items) == 2
        first, second = items
        # dedup_key strips the angle brackets off the Message-ID.
        assert first.dedup_key == "abc@mail"
        assert second.dedup_key == "def@mail"
        assert "Email: Hello" in first.content
        assert "From: alice@x.com" in first.content
        assert "plain body one" in first.content
        # multipart: the text/plain part is extracted, html is skipped.
        assert "the text part" in second.content
        assert "ignored" not in second.content
        assert first.metadata["source"] == "mbox_import"
        assert first.metadata["subject"] == "Hello"

    def test_message_without_id_gets_positional_dedup_key(self, tmp_path):
        mbox_path = _write_mbox(tmp_path / "noid.mbox", [
            {"Subject": "No id", "From": "x@x.com", "body": "hi"},
        ])
        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_path = str(mbox_path)

        items = adapter._fetch_mbox()

        assert len(items) == 1
        assert items[0].dedup_key == "mbox-0"

    def test_marks_processed_and_is_idempotent(self, tmp_path):
        mbox_path = _write_mbox(tmp_path / "t.mbox", [
            {"Message-ID": "<a@m>", "Subject": "S", "From": "a@x", "body": "b"},
        ])
        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_path = str(mbox_path)

        assert len(adapter._fetch_mbox()) == 1
        assert adapter._mbox_processed is True
        # Second call short-circuits — a completed import never re-emits.
        assert adapter._fetch_mbox() == []

    def test_no_path_returns_empty(self):
        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_path = None
        assert adapter._fetch_mbox() == []

    def test_missing_file_returns_empty_and_marks_processed(self, tmp_path):
        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_path = str(tmp_path / "does-not-exist.mbox")

        assert adapter._fetch_mbox() == []
        assert adapter._mbox_processed is True

    def test_stop_event_halts_parsing(self, tmp_path):
        mbox_path = _write_mbox(tmp_path / "t.mbox", [
            {"Message-ID": f"<m{i}@x>", "Subject": f"S{i}",
             "From": "a@x", "body": "b"} for i in range(5)
        ])
        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_path = str(mbox_path)
        adapter.stop()  # set stop event before parsing begins

        assert adapter._fetch_mbox() == []

    def test_corrupt_message_is_skipped_and_import_continues(self, monkeypatch, tmp_path):
        """A single unparseable message must not abort the whole mbox import."""
        good = EmailMessage()
        good["Message-ID"] = "<good@mail>"
        good["Subject"] = "Good"
        good["From"] = "a@x.com"
        good.set_content("good body")

        bad = MagicMock(name="corrupt-message")
        bad.get.side_effect = RuntimeError("unparseable header")

        class _FakeMbox:
            def __init__(self, messages):
                self._messages = messages

            def __len__(self):
                return len(self._messages)

            def __iter__(self):
                return iter(self._messages)

        path = tmp_path / "mixed.mbox"
        path.write_text("placeholder")  # existence check only
        monkeypatch.setattr("mailbox.mbox", lambda p: _FakeMbox([bad, good]))

        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_path = str(path)
        items = adapter._fetch_mbox()

        # The corrupt message is dropped; the good one is still ingested.
        assert [i.dedup_key for i in items] == ["good@mail"]


# ---------------------------------------------------------------------------
# fetch_items dispatch
# ---------------------------------------------------------------------------

class TestFetchItemsDispatch:
    def test_dispatches_to_active_tier(self, monkeypatch):
        adapter = GmailAdapter(tier="mbox")
        monkeypatch.setattr(adapter, "_fetch_mbox", lambda: ["MBOX"])
        monkeypatch.setattr(adapter, "_fetch_imap", lambda: ["IMAP"])
        monkeypatch.setattr(adapter, "_fetch_oauth", lambda: ["OAUTH"])

        assert adapter.fetch_items() == ["MBOX"]
        adapter._tier = "imap"
        assert adapter.fetch_items() == ["IMAP"]
        adapter._tier = "oauth"
        assert adapter.fetch_items() == ["OAUTH"]

    def test_unknown_tier_returns_empty(self):
        adapter = GmailAdapter(tier="nonsense")
        assert adapter.fetch_items() == []


class TestWaitForNextCycle:
    def test_mbox_import_self_terminates(self):
        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_processed = True
        adapter.wait_for_next_cycle()
        assert adapter._stop_event.is_set()

    def test_polling_tier_waits_poll_interval(self, monkeypatch):
        adapter = GmailAdapter(tier="imap")
        waited: list[float] = []
        monkeypatch.setattr(adapter._stop_event, "wait", waited.append)

        adapter.wait_for_next_cycle()

        assert waited == [adapter._poll_interval]
        assert not adapter._stop_event.is_set()


# ---------------------------------------------------------------------------
# _fetch_imap (Tier 1.5)
# ---------------------------------------------------------------------------

class TestFetchImap:
    def test_missing_credentials_returns_empty(self, monkeypatch):
        _patch_credentials(monkeypatch, {})  # no email / password
        adapter = GmailAdapter(tier="imap")
        assert adapter._fetch_imap() == []

    def test_multipart_body_is_extracted(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "email": "me@gmail.com", "password": "pw",
        })
        msg = EmailMessage()
        msg["Message-ID"] = "<mp@mail>"
        msg["Subject"] = "MP"
        msg["From"] = "s@x.com"
        msg.set_content("plain multipart body")
        msg.add_alternative("<html>skip me</html>", subtype="html")

        conn = MagicMock(name="IMAP4_SSL")
        conn.search.return_value = ("OK", [b"7"])
        conn.fetch.return_value = ("OK", [(b"7 (RFC822 {N}", msg.as_bytes())])
        monkeypatch.setattr("imaplib.IMAP4_SSL", MagicMock(return_value=conn))

        adapter = GmailAdapter(tier="imap")
        items = adapter._fetch_imap()

        assert len(items) == 1
        assert "plain multipart body" in items[0].content
        assert "skip me" not in items[0].content

    def test_polls_unseen_messages(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "email": "me@gmail.com", "password": "app-password",
        })

        raw = _raw_email_bytes("IMAP Subject", "sender@x.com", "imap body",
                               message_id="<imap-1@mail>")
        conn = MagicMock(name="IMAP4_SSL")
        conn.search.return_value = ("OK", [b"1 2"])
        conn.fetch.return_value = ("OK", [(b"1 (RFC822 {N}", raw)])

        factory = MagicMock(return_value=conn)
        monkeypatch.setattr("imaplib.IMAP4_SSL", factory)

        adapter = GmailAdapter(tier="imap")
        items = adapter._fetch_imap()

        factory.assert_called_once_with("imap.gmail.com")
        conn.login.assert_called_once_with("me@gmail.com", "app-password")
        conn.select.assert_called_once_with("INBOX")
        conn.logout.assert_called_once()
        # Two UNSEEN message numbers -> two items.
        assert len(items) == 2
        assert items[0].dedup_key == "imap-1@mail"
        assert "IMAP Subject" in items[0].content
        assert "imap body" in items[0].content
        assert items[0].metadata["source"] == "imap"

    def test_connection_error_is_swallowed(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "email": "me@gmail.com", "password": "pw",
        })
        monkeypatch.setattr(
            "imaplib.IMAP4_SSL",
            MagicMock(side_effect=OSError("network down")),
        )
        adapter = GmailAdapter(tier="imap")
        assert adapter._fetch_imap() == []


# ---------------------------------------------------------------------------
# _fetch_oauth (Tier 2)
# ---------------------------------------------------------------------------

def _gmail_message(subject: str, from_addr: str, body: str) -> dict:
    data = base64.urlsafe_b64encode(body.encode()).decode()
    return {
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": from_addr},
                {"name": "Date", "value": "Mon, 20 Jul 2026 10:00:00 +0000"},
            ],
            "body": {"data": data},
        }
    }


class TestFetchOauth:
    def test_incomplete_credentials_returns_empty(self, monkeypatch):
        _patch_credentials(monkeypatch, {"refresh_token": "rt"})  # missing client id/secret
        adapter = GmailAdapter(tier="oauth")
        assert adapter._fetch_oauth() == []

    def test_import_error_returns_empty(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        # Force the google import to fail even if it is installed.
        monkeypatch.setitem(sys.modules, "google.oauth2.credentials", None)
        adapter = GmailAdapter(tier="oauth")
        assert adapter._fetch_oauth() == []

    def test_initial_sync_lists_recent_messages(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        users = service.users.return_value
        users.messages.return_value.list.return_value.execute.return_value = {
            "messages": [{"id": "m1"}, {"id": "m2"}],
        }
        users.messages.return_value.get.return_value.execute.return_value = (
            _gmail_message("Quarterly plan", "boss@corp.com", "read this")
        )
        users.getProfile.return_value.execute.return_value = {"historyId": "999"}
        _install_fake_google(monkeypatch, service)

        adapter = GmailAdapter(tier="oauth")
        items = adapter._fetch_oauth()

        assert [i.dedup_key for i in items] == ["m1", "m2"]
        assert "Email: Quarterly plan" in items[0].content
        assert "read this" in items[0].content
        assert items[0].metadata["source"] == "oauth"
        # History id captured for the next incremental cycle.
        assert adapter._history_id == "999"

    def test_incremental_sync_uses_history(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        users = service.users.return_value
        users.history.return_value.list.return_value.execute.return_value = {
            "history": [{"messagesAdded": [{"message": {"id": "h1"}}]}],
        }
        users.messages.return_value.get.return_value.execute.return_value = (
            _gmail_message("Reply", "peer@corp.com", "new mail body")
        )
        users.getProfile.return_value.execute.return_value = {"historyId": "1001"}
        _install_fake_google(monkeypatch, service)

        adapter = GmailAdapter(tier="oauth")
        adapter._history_id = "500"  # trigger the incremental branch
        items = adapter._fetch_oauth()

        users.history.return_value.list.assert_called_once()
        assert [i.dedup_key for i in items] == ["h1"]
        assert adapter._history_id == "1001"


# ---------------------------------------------------------------------------
# _extract_gmail_body (static)
# ---------------------------------------------------------------------------

class TestExtractGmailBody:
    def test_direct_text_plain_payload(self):
        data = base64.urlsafe_b64encode(b"top level body").decode()
        payload = {"mimeType": "text/plain", "body": {"data": data}}
        assert GmailAdapter._extract_gmail_body(payload) == "top level body"

    def test_text_plain_inside_parts(self):
        data = base64.urlsafe_b64encode(b"part body").decode()
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/html", "body": {"data": "x"}},
                {"mimeType": "text/plain", "body": {"data": data}},
            ],
        }
        assert GmailAdapter._extract_gmail_body(payload) == "part body"

    def test_recurses_into_nested_parts(self):
        data = base64.urlsafe_b64encode(b"deep body").decode()
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": data}},
                    ],
                },
            ],
        }
        assert GmailAdapter._extract_gmail_body(payload) == "deep body"

    def test_returns_empty_when_no_text_part(self):
        payload = {"mimeType": "text/html", "body": {"data": "x"}}
        assert GmailAdapter._extract_gmail_body(payload) == ""


# ---------------------------------------------------------------------------
# BaseAdapter rate limiter — exercised through the ingestion path
# ---------------------------------------------------------------------------

class TestRateLimiterInIngestionPath:
    def test_under_limit_is_not_rate_limited(self):
        adapter = GmailAdapter(config=AdapterConfig(rate_limit_per_hour=3))
        adapter._items_this_hour = 2
        assert adapter._rate_limited() is False

    def test_at_limit_is_rate_limited(self):
        adapter = GmailAdapter(config=AdapterConfig(rate_limit_per_hour=3))
        adapter._items_this_hour = 3
        assert adapter._rate_limited() is True

    def test_hour_window_resets_counter(self):
        adapter = GmailAdapter(config=AdapterConfig(rate_limit_per_hour=1))
        adapter._items_this_hour = 5
        adapter._hour_start = 0.0  # far in the past -> window elapsed
        assert adapter._rate_limited() is False
        assert adapter._items_this_hour == 0

    def test_ingest_increments_hourly_counter(self, monkeypatch):
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            MagicMock(return_value={"ingested": True}),
        )
        adapter = GmailAdapter(config=AdapterConfig(rate_limit_per_hour=100))
        item = IngestItem(content="c", dedup_key="k", metadata={})

        assert adapter._ingest(item) is True
        assert adapter._items_this_hour == 1
        assert adapter._total_ingested == 1

    def test_run_loop_ingests_mbox_then_self_stops(self, monkeypatch, tmp_path):
        """Full ingestion path: run() fetches an mbox, ingests every item
        under the rate limit, then the mbox tier stops itself."""
        mbox_path = _write_mbox(tmp_path / "run.mbox", [
            {"Message-ID": "<r1@m>", "Subject": "One", "From": "a@x", "body": "b1"},
            {"Message-ID": "<r2@m>", "Subject": "Two", "From": "b@x", "body": "b2"},
        ])
        daemon_request = MagicMock(return_value={"ingested": True})
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request", daemon_request)
        # Keep the parent-watchdog satisfied deterministically.
        monkeypatch.setattr("psutil.pid_exists", lambda pid: True)

        adapter = GmailAdapter(
            config=AdapterConfig(batch_delay_sec=0, rate_limit_per_hour=100),
            tier="mbox",
        )
        adapter._mbox_path = str(mbox_path)

        adapter.run()  # terminates: mbox tier calls stop() after processing

        assert adapter._total_ingested == 2
        assert daemon_request.call_count == 2
        assert adapter._stop_event.is_set()

    def test_run_loop_stops_when_parent_daemon_dead(self, monkeypatch, tmp_path):
        monkeypatch.setattr("psutil.pid_exists", lambda pid: False)
        adapter = GmailAdapter(tier="mbox")
        adapter._mbox_path = str(tmp_path / "unused.mbox")
        # Should exit the loop immediately without raising.
        adapter.run()
        assert adapter._total_ingested == 0


# ---------------------------------------------------------------------------
# Resilience & graceful shutdown — one bad item must never abort ingestion,
# and a stop signal must halt mid-fetch cleanly.
# ---------------------------------------------------------------------------

class TestResilienceAndShutdown:
    def test_imap_fetch_error_skips_message_and_continues(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "email": "me@gmail.com", "password": "pw",
        })
        good = _raw_email_bytes("Good", "a@x.com", "good body", "<g@mail>")

        def _fetch(num, spec):
            if num == b"1":
                raise RuntimeError("imap boom")
            return ("OK", [(b"meta", good)])

        conn = MagicMock(name="IMAP4_SSL")
        conn.search.return_value = ("OK", [b"1 2"])
        conn.fetch.side_effect = _fetch
        monkeypatch.setattr("imaplib.IMAP4_SSL", MagicMock(return_value=conn))

        adapter = GmailAdapter(tier="imap")
        items = adapter._fetch_imap()

        # Message 1 raised and was skipped; message 2 still came through.
        assert [i.dedup_key for i in items] == ["g@mail"]

    def test_stop_event_halts_imap_poll(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "email": "me@gmail.com", "password": "pw",
        })
        conn = MagicMock(name="IMAP4_SSL")
        conn.search.return_value = ("OK", [b"1 2"])
        conn.fetch.return_value = ("OK", [(b"meta", _raw_email_bytes("s", "f", "b"))])
        monkeypatch.setattr("imaplib.IMAP4_SSL", MagicMock(return_value=conn))

        adapter = GmailAdapter(tier="imap")
        adapter.stop()  # shutdown before the fetch loop runs
        assert adapter._fetch_imap() == []
        conn.logout.assert_called_once()

    def test_oauth_generic_api_failure_returns_empty(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        (service.users.return_value.messages.return_value
         .list.return_value.execute.side_effect) = RuntimeError("API down")
        _install_fake_google(monkeypatch, service)

        adapter = GmailAdapter(tier="oauth")
        assert adapter._fetch_oauth() == []

    def test_oauth_malformed_message_is_skipped(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        users = service.users.return_value
        users.messages.return_value.list.return_value.execute.return_value = {
            "messages": [{"id": "bad"}, {"id": "good"}],
        }
        good_msg = _gmail_message("Good", "a@x.com", "ok body")

        def _get(userId, id, format):
            resp = MagicMock()
            if id == "bad":
                resp.execute.side_effect = RuntimeError("corrupt payload")
            else:
                resp.execute.return_value = good_msg
            return resp

        users.messages.return_value.get.side_effect = _get
        users.getProfile.return_value.execute.return_value = {"historyId": "1"}
        _install_fake_google(monkeypatch, service)

        adapter = GmailAdapter(tier="oauth")
        items = adapter._fetch_oauth()

        # The corrupt message is dropped; the batch is not aborted.
        assert [i.dedup_key for i in items] == ["good"]

    def test_stop_event_halts_oauth_poll(self, monkeypatch):
        _patch_credentials(monkeypatch, {
            "refresh_token": "rt", "client_id": "cid", "client_secret": "secret",
        })
        service = MagicMock(name="service")
        users = service.users.return_value
        users.messages.return_value.list.return_value.execute.return_value = {
            "messages": [{"id": "m1"}, {"id": "m2"}],
        }
        users.getProfile.return_value.execute.return_value = {"historyId": "1"}
        _install_fake_google(monkeypatch, service)

        adapter = GmailAdapter(tier="oauth")
        adapter.stop()  # shutdown before the per-message loop runs
        assert adapter._fetch_oauth() == []
