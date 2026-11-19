# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A-17 (Fable audit): a Codex user with the plugin AND the installed host hooks gets one recall.

Both registrations run per-prompt recall: the plugin's ``hooks.json`` runs ``slm hook auto_recall``
and the installed ``codex-prompt`` runs it as one of its children. Two recalls, two ``need_turns``
rows with independent exploration draws, and the same memories injected twice. One turn is now
owned by whichever ``slm hook auto_recall`` reaches it first: a per-turn marker keyed by the
session and the prompt, created atomically; the second answers ``{}``.
"""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path

import pytest

from superlocalmemory.hooks import auto_recall_hook as hook

PROMPT = "what did we decide about the release date?"


@pytest.fixture
def ran(monkeypatch):
    """Replace the recall paths with a counter; every call that gets through is one recall."""
    calls: list[str] = []
    lock = threading.Lock()

    def socket_first(prompt, session_id, budget=None):
        with lock:
            calls.append(session_id)
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                       "additionalContext": "memories"}}

    monkeypatch.setattr(hook, "_try_socket_first", socket_first)
    monkeypatch.setattr(hook, "_switched_off", lambda: False)
    monkeypatch.setattr(hook, "_per_prompt_off", lambda: False)
    return calls


def _hook(monkeypatch, prompt=PROMPT, session="conv-1") -> dict:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"prompt": prompt,
                                                              "session_id": session})))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert hook.main() == 0
    return json.loads(out.getvalue() or "{}")


def test_the_second_registration_of_the_same_turn_answers_nothing(monkeypatch, ran) -> None:
    assert "hookSpecificOutput" in _hook(monkeypatch)  # the plugin's hook
    assert _hook(monkeypatch) == {}  # the installed host hook's child, a moment later
    assert ran == ["conv-1"]


def test_another_prompt_or_another_conversation_is_another_turn(monkeypatch, ran) -> None:
    _hook(monkeypatch)
    _hook(monkeypatch, prompt="and what about the staging cluster?")
    _hook(monkeypatch, session="conv-2")
    assert ran == ["conv-1", "conv-1", "conv-2"]


def _age_marker(seconds: float) -> None:
    import os
    import time

    (marker,) = list(Path(hook._turn_folder()).iterdir())
    when = time.time() - seconds
    os.utime(marker, (when, when))


def test_the_same_prompt_long_after_is_a_new_turn(monkeypatch, ran) -> None:
    _hook(monkeypatch)
    _age_marker(hook.TURN_WINDOW_S - 5)
    assert _hook(monkeypatch) == {}  # still the same turn
    _age_marker(hook.TURN_WINDOW_S + 5)
    assert "hookSpecificOutput" in _hook(monkeypatch)  # a person sending it again later
    assert len(ran) == 2


def test_two_registrations_racing_get_exactly_one_recall(monkeypatch, ran) -> None:
    """Codex starts both hooks together: the marker is created atomically, so one wins."""
    results: list[bool] = []

    def attempt():
        results.append(hook._first_for_this_turn("conv-1", PROMPT))

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1 and results.count(False) == 7


def test_a_marker_that_cannot_be_written_never_suppresses_recall(monkeypatch, ran) -> None:
    monkeypatch.setattr("os.open", lambda *a, **k: (_ for _ in ()).throw(PermissionError("no")))
    assert hook._first_for_this_turn("conv-1", PROMPT) is True
    assert hook._first_for_this_turn("conv-1", PROMPT) is True


def test_the_marker_holds_no_prompt_text_and_old_ones_are_removed(monkeypatch, ran, tmp_path) -> None:
    hook._first_for_this_turn("conv-1", PROMPT)
    folder = Path(hook._turn_folder())
    (marker,) = list(folder.iterdir())
    assert PROMPT not in marker.name and PROMPT not in marker.read_text(errors="ignore")
    assert oct(marker.stat().st_mode & 0o777) == "0o600"
    _age_marker(3600)
    hook._first_for_this_turn("conv-1", "another prompt altogether")
    assert marker.exists() is False


def test_the_overlap_is_real_both_registrations_run_the_same_action() -> None:
    """The reason for the marker: the plugin and the installed hook both end in
    ``slm hook auto_recall`` for one Codex prompt event (the plugin's through its launcher, 4.2.30)."""
    from superlocalmemory.hooks import codex_handlers

    plugin = json.loads((Path(__file__).parents[2] / "codex-plugin" / "hooks" / "hooks.json").read_text())
    commands = [h["command"] for g in plugin["hooks"]["UserPromptSubmit"] for h in g["hooks"]]
    assert any("slm-run" in c and "hook auto_recall" in c for c in commands)
    source = Path(codex_handlers.__file__).read_text()
    assert '"auto_recall"' in source and "def hook_codex_prompt" in source


def test_the_end_to_end_one_codex_turn_with_both_registrations_recalls_once(
        monkeypatch, ran) -> None:
    """Run the hook the way each registration does, in two processes' worth of main() calls."""
    plugin_answer = _hook(monkeypatch)
    host_answer = _hook(monkeypatch)
    assert [bool(plugin_answer), bool(host_answer)] == [True, False]
    assert len(ran) == 1
