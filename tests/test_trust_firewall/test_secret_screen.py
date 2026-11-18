# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The write-time secret screen (WS-5, ``trust.secret_screen``): typed placeholders, flag off unchanged."""

from __future__ import annotations

import logging

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, env_name
from superlocalmemory.memory_core import (
    ContentOrigin,
    prepare_for_save,
    prepare_metadata,
    prepare_user_text,
)
from superlocalmemory.trust import secret_screen as ss
from tests.test_trust_firewall import secrets as fake

FLAG = env_name("trust.secret_screen")


@pytest.fixture(autouse=True)
def _own_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
    monkeypatch.delenv(KILL_ENV, raising=False)
    monkeypatch.delenv(FLAG, raising=False)


def _on(monkeypatch, value="on"):
    monkeypatch.setenv(FLAG, value)


# -- each type becomes its placeholder ------------------------------------------------------------

@pytest.mark.parametrize("kind,make", [
    ("github", "GITHUB"), ("stripe", "STRIPE"), ("slack", "SLACK"), ("aws", "AWS"),
    ("anthropic", "ANTHROPIC"), ("openai", "OPENAI"), ("pem", "PRIVATE_KEY")])
def test_a_credential_becomes_a_typed_placeholder(monkeypatch, kind, make) -> None:
    _on(monkeypatch)
    secret = fake.ALL[kind]()
    got = prepare_user_text(None, f"Deploy notes: the key is {secret} until Friday.")
    assert secret not in got.text
    assert f"[REDACTED:{make}:" in got.text
    assert got.text.startswith("Deploy notes: the key is [REDACTED:") and got.text.endswith(
        " until Friday.")
    assert dict(got.secret_types) == {make: 1} and got.secret_count == 1


def test_the_placeholder_keeps_only_the_house_last_four(monkeypatch) -> None:
    _on(monkeypatch)
    secret = fake.github()
    out = prepare_user_text(None, f"token {secret} here").text
    assert out == f"token [REDACTED:GITHUB:{secret[-4:]}] here"


def test_two_credentials_are_counted_per_type(monkeypatch) -> None:
    _on(monkeypatch)
    got = prepare_user_text(None, f"{fake.github()} and {fake.aws()} and {fake.stripe()} again "
                                  f"{fake.stripe()[:-1]}Q")
    assert dict(got.secret_types) == {"GITHUB": 1, "AWS": 1, "STRIPE": 2}


def test_screening_twice_changes_nothing(monkeypatch) -> None:
    _on(monkeypatch)
    once = prepare_user_text(None, f"key {fake.slack()} ok").text
    assert prepare_user_text(None, once).text == once
    assert prepare_user_text(None, once).secret_count == 0


def test_a_placeholder_already_in_the_text_is_not_counted_as_new(monkeypatch) -> None:
    _on(monkeypatch)
    got = prepare_user_text(None, f"Old note [REDACTED:AWS:ab12] and a new key {fake.github()} now.")
    assert dict(got.secret_types) == {"GITHUB": 1} and got.secret_count == 1
    assert "[REDACTED:AWS:ab12]" in got.text


def test_ordinary_text_is_kept_byte_for_byte(monkeypatch) -> None:
    _on(monkeypatch)
    plain = ("The commit 9f2c1ab7e4d3b5a60718c9d2e3f4a5b6c7d8e9f0 fixed the cache; "
             "sha256 of the file is " + "ab12" * 16 + ". See /Users/alice/Documents/work/project/"
             "docs/CHANGELOG.md and https://example.com/a/very/long/path/segment/index.html. "
             "My wifi password is hunter2 and the printer is at 192.168.1.20.")
    got = prepare_user_text(None, plain)
    assert got.text == plain and got.secret_count == 0 and got.secret_types == ()


# -- the flag --------------------------------------------------------------------------------------

def test_off_is_byte_identical(monkeypatch) -> None:
    secret = fake.github()
    text = f"Deploy key {secret} stays."
    got = prepare_user_text(None, text)
    assert got.text == text and got.secret_count == 0 and got.secret_types == ()


def test_shadow_counts_and_logs_but_stores_the_text_as_it_is(monkeypatch, caplog) -> None:
    _on(monkeypatch, "shadow")
    secret = fake.github()
    with caplog.at_level(logging.INFO, logger=ss.logger.name):
        got = prepare_user_text(None, f"Deploy key {secret} stays.")
    assert got.text == f"Deploy key {secret} stays." and got.secret_types == ()
    assert "would replace 1 credential" in caplog.text and "GITHUB 1" in caplog.text


def test_the_log_line_has_counts_and_never_a_value(monkeypatch, caplog) -> None:
    _on(monkeypatch)
    secret = fake.stripe()
    with caplog.at_level(logging.DEBUG):
        prepare_user_text(None, f"billing key {secret} rotates monthly")
    assert secret not in caplog.text and secret[-12:] not in caplog.text
    assert "replaced 1 credential" in caplog.text


def test_the_kill_switch_reads_as_off(monkeypatch) -> None:
    _on(monkeypatch)
    monkeypatch.setenv(KILL_ENV, "1")
    secret = fake.github()
    assert secret in prepare_user_text(None, f"key {secret} here").text


def test_an_unreadable_flag_file_reads_as_off(monkeypatch, tmp_path) -> None:
    home = tmp_path / "slm-home"
    home.mkdir()
    (home / "features.json").write_text("{not json")
    secret = fake.github()
    assert secret in prepare_user_text(None, f"key {secret} here").text


def test_the_file_flag_turns_it_on(tmp_path) -> None:
    from superlocalmemory.core.feature_flags import FlagState, write_flag

    home = tmp_path / "slm-home"
    home.mkdir()
    write_flag(home, "trust.secret_screen", FlagState.ON)
    secret = fake.github()
    assert secret not in prepare_user_text(None, f"key {secret} here").text


# -- derived text and metadata ---------------------------------------------------------------------

def test_derived_text_is_still_always_stripped(monkeypatch) -> None:
    secret = fake.github()
    got = prepare_for_save(f"OCR says {secret}", origin=ContentOrigin.DERIVED_TEXT,
                           pii_redaction=False)
    assert secret not in got.text and got.secret_count == 1


def test_metadata_strings_and_keys_are_screened_when_on(monkeypatch) -> None:
    secret = fake.github()
    meta = {"tags": f"deploy,{secret}", "note": {"inner": [f"x {fake.aws()}", 7]}, "n": 3,
            f"key-{fake.stripe()}": "v", "_slm_hash": fake.slack()}
    off, _ = prepare_metadata(meta, pii_redaction=False)
    assert off == meta
    _on(monkeypatch)
    out, _ = prepare_metadata(meta, pii_redaction=False)
    flat = repr({k: v for k, v in out.items() if k != "_slm_hash"})
    assert secret not in flat and fake.aws() not in flat and fake.stripe() not in flat
    assert out["n"] == 3 and out["note"]["inner"][1] == 7
    assert out["_slm_hash"] == meta["_slm_hash"]  # the server's own keys are never touched
    assert meta["tags"] == f"deploy,{secret}"  # the input is never mutated


def test_metadata_screen_composes_with_pii_redaction(monkeypatch) -> None:
    _on(monkeypatch)
    out, count = prepare_metadata({"who": f"a@example.com {fake.github()}"}, pii_redaction=True)
    assert count == 1 and "a@example.com" not in out["who"] and fake.github() not in out["who"]


def test_a_non_string_and_short_strings_pass(monkeypatch) -> None:
    _on(monkeypatch)
    assert ss.screen("short").text == "short"
    assert ss.screen_tree(None) == (None, ss.Counter())
    assert ss.screen_tree(("a", 1)) == (("a", 1), ss.Counter())


# -- the hook spool (tool events carry what a tool printed) ----------------------------------------

def _tool_event(**over):
    base = {"kind": "tool_event", "event_id": "a" * 32, "profile_id": "default",
            "session_id": "s1", "tool_name": "Bash", "event_type": "complete",
            "input_summary": "env | grep TOKEN", "output_summary": "ok",
            "created_at": "2026-11-18T00:00:00Z"}
    return {**base, **over}


def test_the_hook_spool_screens_what_a_tool_printed(monkeypatch, tmp_path) -> None:
    from superlocalmemory.hooks import event_spool

    secret = fake.github()
    event = _tool_event(output_summary=f"GITHUB_TOKEN={secret}")
    off = event_spool._encode(event)
    assert secret.encode() in off  # off: written as before
    _on(monkeypatch)
    line = event_spool._encode(event)
    assert secret.encode() not in line and b"[REDACTED:GITHUB:" in line
    assert event["output_summary"] == f"GITHUB_TOKEN={secret}"  # the caller's dict is untouched
    result = event_spool.append_events([event], profile="default", root=tmp_path / "spool")
    assert result.written == 1
    assert all(secret.encode() not in p.read_bytes()
               for p in (tmp_path / "spool").rglob("*") if p.is_file())


def test_other_events_and_a_screen_that_cannot_run(monkeypatch) -> None:
    from superlocalmemory.hooks import event_spool

    _on(monkeypatch)
    other = {"kind": "session_finalize", "event_id": "b" * 32, "profile_id": "default",
             "session_id": "s", "observed_ms": 1}
    assert event_spool._encode(other) is not None
    odd = {"kind": "fact_signal", "event_id": "c" * 32, "output_summary": fake.github()}
    assert fake.github().encode() in event_spool._encode(odd)  # only a tool event carries a tool's text
    monkeypatch.setattr(ss, "screen", lambda _t: (_ for _ in ()).throw(RuntimeError("no")))
    assert event_spool._encode(_tool_event(output_summary=fake.github())) is None  # dropped, never clear
