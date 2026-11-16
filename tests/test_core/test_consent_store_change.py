# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-20: a consent names its provider; changing it leaves a receipt first."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.core import consent_store as cs

P = "judge_ladder_hosted"


def _change(home, consent, provider="typesafe"):
    return cs.change(home, P, consent, via="test", actor="tester", provider=provider,
                     audit_path=home / "audit.db")


def test_a_consent_names_its_provider(tmp_path) -> None:
    assert _change(tmp_path, True) is True
    assert cs.given(tmp_path, P) and cs.given(tmp_path, P, provider="typesafe")
    assert not cs.given(tmp_path, P, provider="openrouter")
    assert cs.reference(tmp_path, P, provider="openrouter") is None
    assert cs.reference(tmp_path, P, provider="typesafe").startswith(P + "@")


def test_an_older_consent_without_a_provider_still_holds(tmp_path) -> None:
    cs.record(tmp_path, P, True, via="setup")
    assert cs.given(tmp_path, P, provider="openrouter")


def test_giving_twice_or_withdrawing_nothing_changes_nothing(tmp_path) -> None:
    assert _change(tmp_path, False) is False
    assert _change(tmp_path, True) is True
    assert _change(tmp_path, True) is False
    assert _change(tmp_path, True, provider="openrouter") is True  # a new provider is a new consent


def test_withdrawing_forgets_the_provider(tmp_path) -> None:
    _change(tmp_path, True)
    _change(tmp_path, False)
    entry = json.loads((tmp_path / cs.CONSENTS_FILE).read_text())[P]
    assert entry["consent"] is False and "provider" not in entry


def test_the_notice_names_the_provider(tmp_path) -> None:
    assert "OpenRouter" in cs.notice(P, "openrouter") and "{provider}" not in cs.notice(P, "x")


def test_an_unknown_purpose_is_refused(tmp_path) -> None:
    with pytest.raises(KeyError):
        cs.change(tmp_path, "everything", True, via="t", actor="t", provider="typesafe")


def test_no_receipt_no_change(tmp_path, monkeypatch) -> None:
    from superlocalmemory.compliance import audit

    monkeypatch.setattr(audit.AuditChain, "log", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    with pytest.raises(cs.ConsentNotRecorded):
        _change(tmp_path, True)
    assert not cs.given(tmp_path, P)


# --- slm consent ------------------------------------------------------------

def _cli(monkeypatch, tmp_path, *, tty, judge="jev", consent=True, answer="yes"):
    from superlocalmemory.cli import consent_cmd

    monkeypatch.setattr(consent_cmd, "_home", lambda: tmp_path)
    monkeypatch.setattr(consent_cmd, "_answer_check", lambda: (judge, "typesafe", consent))
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: tty, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    from superlocalmemory.infra import data_root

    monkeypatch.setattr(data_root, "state_path", lambda name: tmp_path / name)
    return consent_cmd


def test_giving_consent_needs_a_person_at_a_terminal(tmp_path, monkeypatch) -> None:
    cmd = _cli(monkeypatch, tmp_path, tty=False)
    assert cmd.cmd_consent(Namespace(action="give", purpose=P, json=False)) == 1
    assert not cs.given(tmp_path, P)


def test_a_person_who_types_yes_gives_it(tmp_path, monkeypatch) -> None:
    cmd = _cli(monkeypatch, tmp_path, tty=True)
    assert cmd.cmd_consent(Namespace(action="give", purpose=P, json=False)) == 0
    assert cs.given(tmp_path, P, provider="typesafe")


def test_anything_but_yes_changes_nothing(tmp_path, monkeypatch) -> None:
    cmd = _cli(monkeypatch, tmp_path, tty=True, answer="y")
    assert cmd.cmd_consent(Namespace(action="give", purpose=P, json=False)) == 0
    assert not cs.given(tmp_path, P)


def test_giving_needs_the_online_answer_check_with_jev(tmp_path, monkeypatch) -> None:
    cmd = _cli(monkeypatch, tmp_path, tty=True, judge="laya")
    assert cmd.cmd_consent(Namespace(action="give", purpose=P, json=False)) == 1
    cmd = _cli(monkeypatch, tmp_path, tty=True, consent=False)
    assert cmd.cmd_consent(Namespace(action="give", purpose=P, json=False)) == 1
    assert not cs.given(tmp_path, P)


def test_withdrawing_works_without_a_terminal(tmp_path, monkeypatch) -> None:
    _change(tmp_path, True)
    cmd = _cli(monkeypatch, tmp_path, tty=False)
    assert cmd.cmd_consent(Namespace(action="withdraw", purpose=P, json=True)) == 0
    assert not cs.given(tmp_path, P)


def test_an_unknown_purpose_is_refused_by_the_cli(tmp_path, monkeypatch) -> None:
    cmd = _cli(monkeypatch, tmp_path, tty=True)
    assert cmd.cmd_consent(Namespace(action="give", purpose="all", json=False)) == 1


def test_the_list_shows_each_purpose(tmp_path, monkeypatch, capsys) -> None:
    cmd = _cli(monkeypatch, tmp_path, tty=False)
    assert cmd.cmd_consent(Namespace(action="list", purpose="", json=False)) == 0
    assert f"{P}: not given" in capsys.readouterr().out
