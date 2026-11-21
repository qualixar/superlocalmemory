# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""4.1.25: the upgrade path offers the answer check (Laya) the way setup does.

Interactive: ask [Y/n], install on yes, remember a no. Unattended: start the
install in its own background process, so the command that triggered it is
never held for a ~1 GB download. Never on unsupported hardware, never over an
explicit opt-out, a remembered no, an install someone already has, or a
check the person set to off or to the hosted one.
"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

import superlocalmemory.core.laya_runtime as lr
import superlocalmemory.retrieval.sufficiency as sufficiency_mod
from superlocalmemory.cli import laya_offer
from superlocalmemory.core.config import RetrievalConfig, SLMConfig
from superlocalmemory.storage.models import Mode


@pytest.fixture(autouse=True)
def _world(monkeypatch, tmp_path):
    monkeypatch.delenv("SLM_SKIP_LAYA", raising=False)
    monkeypatch.delenv("SLM_INSTALL_LAYA", raising=False)
    monkeypatch.setattr(lr, "runtime_dir", lambda *a, **k: tmp_path / "laya")
    monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
    monkeypatch.setattr(lr, "detect", lambda *a, **k: lr.LayaRuntimeStatus(
        state=lr.STATE_NOT_INSTALLED))
    calls: dict[str, list] = {"install": [], "popen": []}
    monkeypatch.setattr(lr, "install", lambda **k: calls["install"].append(1)
                        or lr.LayaRuntimeStatus(state=lr.STATE_READY, python="/p"))

    class _Popen:
        def __init__(self, cmd, **kw):
            calls["popen"].append((cmd, kw))

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    return calls


def _offer(retrieval=None, *, interactive=False, answer="y"):
    out = io.StringIO()
    result = laya_offer.offer_after_upgrade(
        retrieval or RetrievalConfig(), interactive=interactive,
        prompt=lambda _q: answer, out=out)
    return result, out.getvalue()


def test_unattended_upgrade_starts_the_install_in_its_own_process(_world, capsys) -> None:
    result, said = _offer()
    assert result == laya_offer.STARTED and _world["install"] == []
    (cmd, kw), = _world["popen"]
    assert cmd == [sys.executable, "-m", "superlocalmemory.cli.laya_offer"]
    assert kw["start_new_session"] is True and kw["stdin"] is subprocess.DEVNULL
    assert "SLM_SKIP_LAYA=1" in said
    assert capsys.readouterr().out == ""  # stdout can carry hook JSON: never touched


def test_interactive_upgrade_asks_installs_on_yes_and_remembers_a_no(_world) -> None:
    assert _offer(interactive=True)[0] == laya_offer.INSTALLED
    assert _world["install"] == [1] and _world["popen"] == []
    assert _offer(interactive=True, answer="n")[0] == laya_offer.DECLINED
    assert laya_offer.declined()
    assert _offer()[0] == laya_offer.NOT_OFFERED  # a no is not overridden unattended


@pytest.mark.parametrize("why", ["skip_env", "legacy_no", "unsupported", "off", "jev",
                                 "ready", "installed_elsewhere", "declined"])
def test_never_offered_when(why, monkeypatch, _world) -> None:
    retrieval = RetrievalConfig()
    if why == "skip_env":
        monkeypatch.setenv("SLM_SKIP_LAYA", "yes")
    elif why == "legacy_no":
        monkeypatch.setenv("SLM_INSTALL_LAYA", "0")
    elif why == "unsupported":
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: False)
    elif why in ("off", "jev"):
        retrieval = RetrievalConfig(sufficiency_judge=why)
    elif why == "ready":
        monkeypatch.setattr(lr, "detect", lambda *a, **k: lr.LayaRuntimeStatus(
            state=lr.STATE_READY))
    elif why == "installed_elsewhere":
        monkeypatch.setattr(lr, "detect", lambda *a, **k: lr.LayaRuntimeStatus(
            state=lr.STATE_FAILED, action=lr.ACTION_CHECK))
    else:
        laya_offer.remember_decline()
    assert _offer(retrieval, interactive=True)[0] == laya_offer.NOT_OFFERED
    assert _world["install"] == [] and _world["popen"] == []


def test_an_unfinished_install_of_our_own_is_resumed(monkeypatch, _world) -> None:
    monkeypatch.setattr(lr, "detect", lambda *a, **k: lr.LayaRuntimeStatus(
        state=lr.STATE_FAILED, action=lr.ACTION_SETUP))
    assert _offer()[0] == laya_offer.STARTED


def test_after_upgrade_never_raises(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(laya_offer, "offer_after_upgrade", boom)
    assert laya_offer.after_upgrade() == laya_offer.NOT_OFFERED


@pytest.mark.parametrize("mode", [Mode.A, Mode.B])
def test_auto_keeps_resolving_to_laya_when_it_is_ready(mode, monkeypatch) -> None:
    from superlocalmemory.core import judge_selection as js

    retrieval = SLMConfig.for_mode(mode).retrieval
    assert retrieval.sufficiency_judge == "auto"
    monkeypatch.setattr(js, "_checked_install_plan", lambda *a, **k: object())
    assert js.resolve_judge_mode(retrieval) == "laya"
    monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: False)
    assert js.resolve_judge_mode(retrieval) == "off"


def test_the_cli_offers_after_the_upgrade_banner_only() -> None:
    import inspect

    from superlocalmemory.cli import main

    # The steps before the command runs live in _prepare, which main calls
    # before it builds the parser.
    text = inspect.getsource(main._prepare)
    banner = text.index("if check_and_emit_upgrade_banner(_ver):")
    assert banner < text.index("laya_offer.after_upgrade(")
    body = inspect.getsource(main.main)
    assert body.index("_prepare(") < body.index("build_parser(_ver)")
