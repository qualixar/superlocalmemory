"""Issue #319: the Stop hook saved its session summary without the session id.

Claude Code sends ``session_id`` in the hook's stdin payload. The Stop hook read
only the ``CLAUDE_SESSION_ID`` environment variable, so the summary was stored
with an empty session id and a session-summary lookup for the live session
found nothing (``no_session``)."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.hooks.hook_handlers import handle_hook


def _stop(monkeypatch, stdin_text, env_id=None):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/proj")
    if env_id is None:
        monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_SESSION_ID", env_id)
    with patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate"), \
         patch("superlocalmemory.hooks.hook_handlers._daemon_post", return_value=True) as post, \
         patch("superlocalmemory.hooks.hook_handlers._captures_session_summary", return_value=True), \
         patch("superlocalmemory.hooks.hook_handlers.subprocess.run",
               return_value=MagicMock(stdout="", returncode=0)), \
         patch("sys.stdin", io.StringIO(stdin_text)):
        with pytest.raises(SystemExit):
            handle_hook("stop")
    remember = [c for c in post.call_args_list if c[0][0] == "/remember"]
    assert remember, "the session summary should be saved"
    return remember[0][0][1]


def test_summary_carries_the_payload_session_id(monkeypatch):
    body = _stop(monkeypatch, json.dumps({"session_id": "c0ffee-1234", "hook_event_name": "Stop"}))
    assert body["session_id"] == "c0ffee-1234"
    assert body["idempotency_key"] == "hook-session-end:c0ffee-1234"


def test_payload_wins_over_the_environment(monkeypatch):
    body = _stop(monkeypatch, json.dumps({"session_id": "from-payload"}), env_id="from-env")
    assert body["session_id"] == "from-payload"


@pytest.mark.parametrize("stdin_text", ["", "not json", json.dumps({"session_id": 7}),
                                        json.dumps({"session_id": "x" * 500})])
def test_no_usable_payload_falls_back_to_the_environment(monkeypatch, stdin_text):
    assert _stop(monkeypatch, stdin_text, env_id="from-env")["session_id"] == "from-env"
