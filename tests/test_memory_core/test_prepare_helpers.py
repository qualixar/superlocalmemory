# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Metadata, key and one-call text helpers of the shared save step."""

from __future__ import annotations

import copy
import hashlib
import logging

from superlocalmemory.memory_core import (
    prepare_key,
    prepare_metadata,
    prepare_user_text,
    same_after_redaction,
)

EMAIL = "bob.builder@example.com"


class _Cfg:
    def __init__(self, on: bool) -> None:
        self.pii_redaction = on


def test_metadata_is_redacted_recursively_without_mutating_input() -> None:
    value = {
        "from": f"Bob <{EMAIL}>",
        f"key {EMAIL}": ["x", {"deep": (f"mail {EMAIL}", 7)}],
        "n": 3, "flag": True, "none": None,
    }
    before = copy.deepcopy(value)
    out, count = prepare_metadata(value, pii_redaction=True)
    assert value == before
    assert EMAIL not in repr(out)
    assert count == 3
    assert out["n"] == 3 and out["flag"] is True and out["none"] is None
    assert isinstance(out[f"key [PII:EMAIL]"][1]["deep"], tuple)


def test_metadata_is_untouched_when_redaction_is_off() -> None:
    value = {"from": EMAIL, "list": [EMAIL]}
    out, count = prepare_metadata(value, pii_redaction=False)
    assert out == value and count == 0


def test_key_is_replaced_only_when_it_changes() -> None:
    key = f"event:{EMAIL}:42"
    out = prepare_key(key, pii_redaction=True)
    assert out == "redacted:" + hashlib.sha256(key.encode()).hexdigest()
    assert prepare_key("event:42", pii_redaction=True) == "event:42"
    assert prepare_key(key, pii_redaction=False) == key


def test_prepare_user_text_reads_config_and_logs_only_counts(caplog, monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    with caplog.at_level(logging.INFO):
        on = prepare_user_text(_Cfg(True), f"mail {EMAIL}")
    assert on.text == "mail [PII:EMAIL]" and on.pii_count == 1
    assert EMAIL not in caplog.text and "1" in caplog.text
    off = prepare_user_text(_Cfg(False), f"mail {EMAIL}")
    assert off.text == f"mail {EMAIL}"


def test_same_after_redaction() -> None:
    assert same_after_redaction(f"hi {EMAIL}", "hi [PII:EMAIL]", True) is True
    assert same_after_redaction(f"hi {EMAIL}", "hi [PII:EMAIL]", False) is False
    assert same_after_redaction("other", "hi [PII:EMAIL]", True) is False
