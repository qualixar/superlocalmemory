# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""The shared save step: what happens to text before it is stored."""

from __future__ import annotations

import hashlib
import time

import pytest

from superlocalmemory.memory_core import (
    ContentOrigin,
    PreparedContent,
    pii_redaction_enabled,
    prepare_for_save,
)

PII_TEXT = (
    "Reach Dana at dana.ops@example.com or 415-555-0132, "
    "SSN 123-45-6789, card 4111 1111 1111 1111, host 192.168.10.20."
)


def _fake_key() -> str:
    return "sk-ant-" + "A1b2C3d4" * 4


def test_user_text_is_byte_identical_when_redaction_is_off() -> None:
    decomposed = "café " + PII_TEXT
    out = prepare_for_save(
        decomposed, origin=ContentOrigin.USER_TEXT, pii_redaction=False,
    )
    assert isinstance(out, PreparedContent)
    assert out.text == decomposed
    assert out.text.encode("utf-8") == decomposed.encode("utf-8")
    assert out.pii_count == 0 and out.secret_count == 0
    assert out.content_sha256 == hashlib.sha256(
        decomposed.encode("utf-8")
    ).hexdigest()


def test_user_text_keeps_credentials_when_redaction_is_off() -> None:
    text = f"my key is {_fake_key()}"
    out = prepare_for_save(text, origin=ContentOrigin.USER_TEXT, pii_redaction=False)
    assert out.text == text


def test_user_text_redacts_each_identifier_kind_when_on() -> None:
    out = prepare_for_save(PII_TEXT, origin=ContentOrigin.USER_TEXT, pii_redaction=True)
    for raw in (
        "dana.ops@example.com", "415-555-0132", "123-45-6789",
        "4111 1111 1111 1111", "192.168.10.20",
    ):
        assert raw not in out.text
    assert "[PII:EMAIL]" in out.text
    assert out.pii_count == 5
    assert out.content_sha256 == hashlib.sha256(out.text.encode()).hexdigest()


def test_redaction_is_idempotent() -> None:
    first = prepare_for_save(PII_TEXT, origin=ContentOrigin.USER_TEXT, pii_redaction=True)
    second = prepare_for_save(first.text, origin=ContentOrigin.USER_TEXT, pii_redaction=True)
    assert second.text == first.text
    assert second.pii_count == 0


def test_derived_text_is_nfc_and_always_strips_credentials() -> None:
    text = f"café token {_fake_key()}"
    out = prepare_for_save(text, origin=ContentOrigin.DERIVED_TEXT, pii_redaction=False)
    assert out.text.startswith("café ")
    assert _fake_key() not in out.text
    assert out.secret_count >= 1
    assert out.pii_count == 0


def test_non_str_input_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        prepare_for_save(b"bytes", origin=ContentOrigin.USER_TEXT, pii_redaction=True)  # type: ignore[arg-type]


def test_empty_text_is_returned_unchanged() -> None:
    out = prepare_for_save("", origin=ContentOrigin.DERIVED_TEXT, pii_redaction=True)
    assert out.text == "" and out.pii_count == 0 and out.secret_count == 0


def test_a_full_length_memory_prepares_quickly() -> None:
    text = ("Meeting notes about the release plan and owners. " * 600)[:24_000]
    started = time.perf_counter()
    prepare_for_save(text, origin=ContentOrigin.USER_TEXT, pii_redaction=True)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"prepare_for_save 24000 chars: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 50


class _Cfg:
    def __init__(self, value: bool) -> None:
        self.pii_redaction = value


def test_enabled_by_config(monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    assert pii_redaction_enabled(_Cfg(True)) is True
    assert pii_redaction_enabled(_Cfg(False)) is False
    assert pii_redaction_enabled(None) is False


@pytest.mark.parametrize("value", ["1", "on", "TRUE", "Yes", " true "])
def test_enabled_by_env(monkeypatch, value) -> None:
    monkeypatch.setenv("SLM_PII_REDACTION", value)
    assert pii_redaction_enabled(_Cfg(False)) is True


@pytest.mark.parametrize("value", ["", "0", "off", "no"])
def test_env_off_values(monkeypatch, value) -> None:
    monkeypatch.setenv("SLM_PII_REDACTION", value)
    assert pii_redaction_enabled(_Cfg(False)) is False
