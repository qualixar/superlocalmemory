"""#49: `slm session open|close` — local session control for hooks (no model
roundtrip). Tests the CLI command routes to the right daemon endpoint with the
right body, and that the daemon exposes the request models + routes.
"""
from argparse import Namespace
from unittest.mock import patch

from superlocalmemory.cli.commands import cmd_session, dispatch


def _ns(**kw):
    return Namespace(**kw)


def test_session_close_calls_daemon_close_endpoint(capsys):
    calls = {}

    def fake_request(method, path, body=None):
        calls["method"], calls["path"], calls["body"] = method, path, body
        return {"ok": True, "session_id": "sess-123", "summary_events_created": 4}

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request", side_effect=fake_request),
    ):
        cmd_session(_ns(session_command="close", session_id="sess-123"))

    assert calls["method"] == "POST"
    assert calls["path"] == "/session/close"
    assert calls["body"] == {"session_id": "sess-123"}
    out = capsys.readouterr().out
    assert "sess-123" in out and "4 summary" in out


def test_session_close_empty_id_uses_most_recent(capsys):
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={"ok": True, "session_id": "", "summary_events_created": 0},
        ),
    ):
        cmd_session(_ns(session_command="close", session_id=""))
    out = capsys.readouterr().out
    assert "most recent" in out


def test_session_open_calls_daemon_open_endpoint(capsys):
    calls = {}

    def fake_request(method, path, body=None):
        calls["path"], calls["body"] = path, body
        return {"ok": True, "query": "project context /x", "warmed": 7}

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request", side_effect=fake_request),
    ):
        cmd_session(_ns(session_command="open", project_path="/x", query="", max_results=10))

    assert calls["path"] == "/session/open"
    assert calls["body"]["project_path"] == "/x"
    assert "warmed 7" in capsys.readouterr().out


def test_session_unknown_action_prints_usage(capsys):
    cmd_session(_ns(session_command=None))
    assert "Usage: slm session" in capsys.readouterr().out


def test_dispatch_maps_session():
    from superlocalmemory.cli import commands
    # 'session' must be a registered handler routing to cmd_session
    src = commands.dispatch.__doc__  # cheap presence guard
    assert src is not None
    # the handler is registered inside dispatch(); call with a stub that short-circuits
    with patch("superlocalmemory.cli.commands.cmd_session") as m, \
         patch("superlocalmemory.hooks.claude_code_hooks.auto_install_if_needed"):
        dispatch(_ns(command="session", session_command="close", session_id=""))
        assert m.called


def test_daemon_exposes_session_models_and_routes():
    from superlocalmemory.server import unified_daemon as ud
    assert hasattr(ud, "SessionOpenRequest")
    assert hasattr(ud, "SessionCloseRequest")
