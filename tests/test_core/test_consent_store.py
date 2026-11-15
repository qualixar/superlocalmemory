# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Consent to send memories off the machine: explicit, fail-closed, never from the environment."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core import consent_store
from superlocalmemory.core.feature_flags import FLAGS


def test_no_file_means_no_consent(tmp_path) -> None:
    assert not consent_store.given(tmp_path, "judge_ladder_hosted")


def test_record_give_and_withdraw(tmp_path) -> None:
    consent_store.record(tmp_path, "judge_ladder_hosted", True, via="cli")
    assert consent_store.given(tmp_path, "judge_ladder_hosted")
    entry = json.loads((tmp_path / "consents.json").read_text())["judge_ladder_hosted"]
    assert entry["via"] == "cli" and entry["at"]
    consent_store.record(tmp_path, "judge_ladder_hosted", False, via="dashboard")
    assert not consent_store.given(tmp_path, "judge_ladder_hosted")


@pytest.mark.parametrize("content", ['{"judge_ladder_hosted": true}', '{"judge_ladder_hosted": '
                                     '{"consent": "yes"}}', "not json", "[]"])
def test_anything_but_an_explicit_entry_is_no_consent(tmp_path, content) -> None:
    (tmp_path / "consents.json").write_text(content)
    assert not consent_store.given(tmp_path, "judge_ladder_hosted")


def test_the_environment_cannot_give_consent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLM_FEATURE_JUDGE_LADDER_HOSTED", "on")
    monkeypatch.setenv("SLM_CONSENT_JUDGE_LADDER_HOSTED", "true")
    assert not consent_store.given(tmp_path, "judge_ladder_hosted")
    assert "judge.ladder.hosted" not in FLAGS


def test_unknown_purposes_are_refused(tmp_path) -> None:
    assert not consent_store.given(tmp_path, "anything_else")
    with pytest.raises(KeyError):
        consent_store.record(tmp_path, "anything_else", True, via="cli")
    with pytest.raises(ValueError):
        consent_store.record(tmp_path, "judge_ladder_hosted", True, via="")
