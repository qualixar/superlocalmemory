# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A verdict is added to the memories, never put in their place.

``session_init`` and the CLI already keep the memories and add one line when
the answer check says none of them answers. The other auto-injection surfaces
(AutoRecall — which serves ``slm://context`` and ``slm session-context
--full`` — and the per-prompt hook) must do the same: replacing real context
with "no stored memory answers this" on a false abstention hides exactly what
the user stored.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.core.answer_check_notice import answer_check_line
from superlocalmemory.hooks.auto_recall import AutoRecall

INSUFFICIENT = "none of these memories answers the question"


def _response(contents, **meta):
    results = [
        SimpleNamespace(
            fact=SimpleNamespace(fact_id=f"f{i}", content=c, created_at=""),
            score=0.9,
        )
        for i, c in enumerate(contents)
    ]
    return SimpleNamespace(results=results, **meta)


JUDGED_OUT = dict(calibration_status="measured_small_sample_not_calibrated",
                  answer_confidence=0.04, abstained=True,
                  abstention_reason="judged_insufficient")


def test_auto_recall_keeps_every_memory_and_adds_the_verdict():
    auto = AutoRecall(recall_fn=lambda q, limit=10: _response(
        ["JWT with 1h expiry", "Deploys go out on Fridays"], **JUDGED_OUT))
    out = auto.get_session_context(query="recent decisions")
    assert "JWT with 1h expiry" in out
    assert "Deploys go out on Fridays" in out
    assert INSUFFICIENT in out
    assert out.index(INSUFFICIENT) < out.index("JWT with 1h expiry")


def test_auto_recall_says_nothing_extra_without_a_judge():
    auto = AutoRecall(recall_fn=lambda q, limit=10: _response(["JWT with 1h expiry"]))
    out = auto.get_session_context(query="recent decisions")
    assert "JWT with 1h expiry" in out
    assert "Answer check" not in out


def test_the_verdict_line_is_the_same_on_every_surface():
    """session_init's prefix and AutoRecall's line are one function."""
    from superlocalmemory.mcp.tools_active import _answer_check_prefix

    response = _response(["x"], **JUDGED_OUT)
    assert _answer_check_prefix(response) == answer_check_line(response)
    assert answer_check_line(dict(JUDGED_OUT)) == answer_check_line(response)


@pytest.mark.parametrize(("meta", "expected"), [
    ({}, ""),
    ({"calibration_status": "uncalibrated", "abstained": True,
      "abstention_reason": "judged_insufficient"}, ""),
    (dict(JUDGED_OUT), "Answer check: none of these memories answers the question"),
    ({"calibration_status": "x", "answer_confidence": 0.86, "abstained": False},
     "Answer check: likely answered (confidence 0.86)."),
    ({"calibration_status": "x", "abstained": True, "abstention_reason": "evidence_floor"}, ""),
])
def test_answer_check_line_cases(meta, expected):
    line = answer_check_line(meta)
    assert line.startswith(expected) if expected else line == ""


def test_the_prompt_hook_keeps_memories_and_adds_the_verdict():
    from superlocalmemory.hooks.auto_recall_hook import _format_envelope

    results = [{"content": "JWT with 1h expiry", "score": 0.9, "fact_id": "f0"}]
    envelope = _format_envelope(results, response=dict(JUDGED_OUT, results=results))
    context = envelope["hookSpecificOutput"]["additionalContext"]
    assert "JWT with 1h expiry" in context
    assert INSUFFICIENT in context


def test_the_prompt_hook_is_unchanged_without_a_verdict():
    from superlocalmemory.hooks.auto_recall_hook import _format_envelope

    results = [{"content": "JWT with 1h expiry", "score": 0.9, "fact_id": "f0"}]
    assert _format_envelope(results) == _format_envelope(results, response={"results": results})


def test_the_prompt_hook_entry_point_carries_the_verdict(monkeypatch, capsys):
    import io
    import json

    from superlocalmemory.hooks import auto_recall_hook

    results = [{"content": "JWT with 1h expiry", "score": 0.9, "fact_id": "f0"}]
    monkeypatch.setattr(auto_recall_hook, "_try_socket_first", lambda *a, **k: None)
    monkeypatch.setattr(auto_recall_hook, "_per_prompt_off", lambda: False)  # D-47: switched on
    monkeypatch.setattr(auto_recall_hook, "_do_recall_response",
                        lambda *a, **k: dict(JUDGED_OUT, results=results))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"prompt": "how do we sign tokens?", "session_id": "s-1"})))
    assert auto_recall_hook.main() == 0
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "JWT with 1h expiry" in context
    assert INSUFFICIENT in context
