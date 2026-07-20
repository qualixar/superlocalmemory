# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Integration tests for Ingestion Pipeline (Phase E)."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Parser Tests
# ---------------------------------------------------------------------------

class TestParsers:

    def test_parse_srt(self, tmp_path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nAlice: Hello everyone\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nBob: Hi Alice\n\n"
            "3\n00:00:07,000 --> 00:00:10,000\nSome text without speaker\n"
        )
        from superlocalmemory.ingestion.parsers import parse_srt
        utterances = parse_srt(srt_file)
        assert len(utterances) == 3
        assert utterances[0].speaker == "Alice"
        assert utterances[1].speaker == "Bob"
        assert "Hello everyone" in utterances[0].text

    def test_parse_vtt(self, tmp_path):
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Alice: First utterance\n\n"
            "00:00:04.000 --> 00:00:06.000\n"
            "Bob: Second utterance\n"
        )
        from superlocalmemory.ingestion.parsers import parse_vtt
        utterances = parse_vtt(vtt_file)
        assert len(utterances) >= 2
        speakers = {u.speaker for u in utterances}
        assert "Alice" in speakers

    def test_parse_transcript_file_srt(self, tmp_path):
        srt_file = tmp_path / "meeting.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nAlice: Let's discuss the plan\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nBob: I agree\n"
        )
        from superlocalmemory.ingestion.parsers import parse_transcript_file
        text, speakers = parse_transcript_file(srt_file)
        assert "Alice" in speakers or "Bob" in speakers
        assert "plan" in text

    def test_content_hash_path_independent(self, tmp_path):
        """Same content in different paths → same hash."""
        f1 = tmp_path / "a" / "test.txt"
        f2 = tmp_path / "b" / "test.txt"
        f1.parent.mkdir()
        f2.parent.mkdir()
        content = "identical content for hashing"
        f1.write_text(content)
        f2.write_text(content)

        from superlocalmemory.ingestion.parsers import content_hash
        assert content_hash(f1) == content_hash(f2)

    def test_content_hash_differs_for_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")

        from superlocalmemory.ingestion.parsers import content_hash
        assert content_hash(f1) != content_hash(f2)


# ---------------------------------------------------------------------------
# Base Adapter Tests
# ---------------------------------------------------------------------------

class TestBaseAdapter:

    def test_rate_limiting(self):
        from superlocalmemory.ingestion.base_adapter import BaseAdapter, AdapterConfig
        config = AdapterConfig(rate_limit_per_hour=3)
        adapter = BaseAdapter(config)
        adapter._items_this_hour = 3
        assert adapter._rate_limited() is True

    def test_rate_limit_resets_after_hour(self):
        import time
        from superlocalmemory.ingestion.base_adapter import BaseAdapter, AdapterConfig
        config = AdapterConfig(rate_limit_per_hour=3)
        adapter = BaseAdapter(config)
        adapter._items_this_hour = 3
        adapter._hour_start = time.time() - 3601  # Over an hour ago
        assert adapter._rate_limited() is False

    def test_stop_event(self):
        from superlocalmemory.ingestion.base_adapter import BaseAdapter
        adapter = BaseAdapter()
        assert not adapter._stop_event.is_set()
        adapter.stop()
        assert adapter._stop_event.is_set()


# ---------------------------------------------------------------------------
# Adapter Manager Tests
# ---------------------------------------------------------------------------

class TestAdapterManager:

    def test_list_adapters(self, tmp_path):
        config_file = tmp_path / "adapters.json"
        config_file.write_text(json.dumps({
            "gmail": {"enabled": True},
            "calendar": {"enabled": False},
            "transcript": {"enabled": False},
        }))

        with patch("superlocalmemory.ingestion.adapter_manager._ADAPTERS_CONFIG", config_file):
            from superlocalmemory.ingestion.adapter_manager import list_adapters
            adapters = list_adapters()
            assert len(adapters) == 3
            gmail = [a for a in adapters if a["name"] == "gmail"][0]
            assert gmail["enabled"] is True

    def test_enable_adapter(self, tmp_path):
        config_file = tmp_path / "adapters.json"
        config_file.write_text(json.dumps({"gmail": {"enabled": False}}))

        with patch("superlocalmemory.ingestion.adapter_manager._ADAPTERS_CONFIG", config_file):
            from superlocalmemory.ingestion.adapter_manager import enable_adapter
            result = enable_adapter("gmail")
            assert result["ok"] is True

            data = json.loads(config_file.read_text())
            assert data["gmail"]["enabled"] is True

    def test_enable_invalid_adapter(self):
        from superlocalmemory.ingestion.adapter_manager import enable_adapter
        result = enable_adapter("invalid_adapter")
        assert result["ok"] is False

    def test_disable_adapter(self, tmp_path):
        config_file = tmp_path / "adapters.json"
        config_file.write_text(json.dumps({"gmail": {"enabled": True}}))

        with patch("superlocalmemory.ingestion.adapter_manager._ADAPTERS_CONFIG", config_file):
            with patch("superlocalmemory.ingestion.adapter_manager._is_running", return_value=(False, None)):
                from superlocalmemory.ingestion.adapter_manager import disable_adapter
                result = disable_adapter("gmail")
                assert result["ok"] is True


# ---------------------------------------------------------------------------
# Credential Tests
# ---------------------------------------------------------------------------

class TestCredentials:

    def test_store_and_load_file_fallback(self, tmp_path, monkeypatch):
        """Credential storage uses file fallback when keyring unavailable."""
        import superlocalmemory.ingestion.credentials as credentials

        cred_dir = tmp_path / "credentials"
        monkeypatch.setitem(sys.modules, "keyring", None)
        monkeypatch.setattr(credentials, "_CRED_DIR", cred_dir)

        result = credentials.store_credential("test-service", "api_key", "secret123")
        assert result is True

        value = credentials.load_credential("test-service", "api_key")
        assert value == "secret123"

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        import superlocalmemory.ingestion.credentials as credentials

        cred_dir = tmp_path / "credentials"
        monkeypatch.setitem(sys.modules, "keyring", None)
        monkeypatch.setattr(credentials, "_CRED_DIR", cred_dir)
        assert credentials.load_credential("nope", "nope") is None

    def test_file_fallback_creates_directory(self, tmp_path, monkeypatch):
        """When keyring raises, file fallback creates the cred directory."""
        import superlocalmemory.ingestion.credentials as credentials

        cred_dir = tmp_path / "new_creds_dir"
        assert not cred_dir.exists()
        monkeypatch.setitem(sys.modules, "keyring", None)
        monkeypatch.setattr(credentials, "_CRED_DIR", cred_dir)

        credentials.store_credential("svc", "key", "val")
        assert cred_dir.exists()
        val = credentials.load_credential("svc", "key")
        assert val == "val"
