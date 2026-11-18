# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-28: the measurement notice is shown until a person has seen this version of it."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.outcomes import disclosure


def test_a_fresh_home_is_pending(tmp_path) -> None:
    assert disclosure.pending(tmp_path) is True


def test_acknowledging_ends_pending_and_records_version_time_and_channel(tmp_path) -> None:
    entry = disclosure.acknowledge(tmp_path, "cli")
    assert disclosure.pending(tmp_path) is False
    saved = json.loads((tmp_path / "disclosures.json").read_text())["measurement"]
    assert saved["version"] == disclosure.NOTICE_VERSION == entry["version"]
    assert saved["channel"] == "cli" and saved["at"]


@pytest.mark.parametrize("channel", ["cli", "dashboard", "install"])
def test_each_surface_may_acknowledge(tmp_path, channel) -> None:
    disclosure.acknowledge(tmp_path, channel)
    assert disclosure.pending(tmp_path) is False


def test_an_unknown_channel_is_refused_and_nothing_is_written(tmp_path) -> None:
    with pytest.raises(ValueError):
        disclosure.acknowledge(tmp_path, "carrier-pigeon")
    assert disclosure.pending(tmp_path) is True
    assert not (tmp_path / "disclosures.json").exists()


def test_a_new_notice_version_is_pending_again(tmp_path) -> None:
    (tmp_path / "disclosures.json").write_text(json.dumps(
        {"measurement": {"version": "4.2.0-0", "at": "2026-01-01T00:00:00+00:00", "channel": "cli"}}))
    assert disclosure.pending(tmp_path) is True


@pytest.mark.parametrize("content", ["", "{not json", "[]", '{"measurement": 5}',
                                     '{"measurement": {"version": 7}}'])
def test_a_damaged_file_reads_as_pending_and_never_raises(tmp_path, content) -> None:
    (tmp_path / "disclosures.json").write_text(content)
    assert disclosure.pending(tmp_path) is True
    disclosure.acknowledge(tmp_path, "cli")  # and it heals
    assert disclosure.pending(tmp_path) is False


def test_an_unreadable_home_reads_as_pending(tmp_path) -> None:
    assert disclosure.pending(tmp_path / "does" / "not" / "exist") is True


def test_acknowledging_keeps_the_change_history(tmp_path) -> None:
    disclosure.record_change(tmp_path, {"action": "off", "channel": "cli"})
    disclosure.acknowledge(tmp_path, "dashboard")
    saved = json.loads((tmp_path / "disclosures.json").read_text())
    assert saved["changes"][0]["action"] == "off" and saved["measurement"]["channel"] == "dashboard"


def test_the_history_is_capped(tmp_path) -> None:
    for i in range(disclosure.HISTORY_MAX + 20):
        disclosure.record_change(tmp_path, {"action": "on", "n": i})
    changes = json.loads((tmp_path / "disclosures.json").read_text())["changes"]
    assert len(changes) == disclosure.HISTORY_MAX and changes[-1]["n"] == disclosure.HISTORY_MAX + 19


def test_the_notice_says_what_how_often_and_how_to_turn_it_off() -> None:
    words = disclosure.text()
    assert "slm measurement off" in words and "Settings" in words
    assert "1 in 10" in words and "1 in 20" in words
    for fact in ("pinned", "recall", "cue", "lease", "this computer"):
        assert fact in words.lower()
    for jargon in ("IPW", "propensity", "ε", "epsilon", "holdout"):
        assert jargon not in words


def test_the_notice_numbers_follow_the_design_constants(monkeypatch) -> None:
    from superlocalmemory.outcomes import randomiser, swaps

    monkeypatch.setattr(randomiser, "EPSILON", 0.2)
    monkeypatch.setattr(swaps, "SWAP_SHARE", 0.25)
    assert "1 in 5" in disclosure.text() and "1 in 4" in disclosure.text()


def test_announce_prints_to_the_stream_and_acknowledges_only_on_a_terminal(tmp_path) -> None:
    import io

    class Tty(io.StringIO):
        def isatty(self):
            return True

    quiet = io.StringIO()
    assert disclosure.announce(tmp_path, stream=quiet, channel="install") is True
    assert "slm measurement off" in quiet.getvalue()
    assert disclosure.pending(tmp_path) is True  # a pipe or a log is not a person reading it
    terminal = Tty()
    assert disclosure.announce(tmp_path, stream=terminal, channel="install") is True
    assert disclosure.pending(tmp_path) is False
    again = io.StringIO()
    assert disclosure.announce(tmp_path, stream=again, channel="install") is False
    assert again.getvalue() == ""


def test_announce_never_raises(tmp_path, monkeypatch) -> None:
    class Broken:
        def write(self, _):
            raise OSError("closed")

        def isatty(self):
            return True

    assert disclosure.announce(tmp_path, stream=Broken(), channel="install") is False
