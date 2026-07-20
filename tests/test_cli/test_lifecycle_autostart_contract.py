# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Lifecycle commands must not be pre-empted by global daemon auto-start."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("command", "action", "expected"),
    [
        ("recall", None, True),
        ("serve", "start", False),
        ("serve", "status", False),
        ("serve", "stop", False),
        ("serve", "install", False),
        ("serve", "uninstall", False),
        ("restart", None, False),
    ],
)
def test_command_requires_daemon_is_lifecycle_aware(
    command: str,
    action: str | None,
    expected: bool,
) -> None:
    """A lifecycle command must reach its handler before any daemon mutation."""
    from superlocalmemory.cli.main import _command_requires_daemon

    args = SimpleNamespace(command=command, action=action)
    assert _command_requires_daemon(args) is expected


@pytest.mark.parametrize(
    "argv",
    [
        ["slm", "serve"],
        ["slm", "serve", "start"],
        ["slm", "serve", "status"],
        ["slm", "serve", "stop"],
        ["slm", "restart"],
    ],
)
def test_main_dispatch_does_not_prestart_lifecycle_commands(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    from superlocalmemory.cli import daemon, main as main_module
    from superlocalmemory.cli import commands, setup_wizard, version_banner

    ensure = Mock(return_value=True)
    monkeypatch.setattr(daemon, "ensure_daemon", ensure)
    monkeypatch.setattr(commands, "dispatch", Mock())
    monkeypatch.setattr(setup_wizard, "check_first_use", Mock())
    monkeypatch.setattr(version_banner, "check_and_emit_upgrade_banner", Mock(return_value=False))
    monkeypatch.setattr("sys.argv", argv)

    main_module.main()

    ensure.assert_not_called()


def test_plain_serve_starts_once_inside_handler(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from superlocalmemory.cli import daemon, main as main_module
    from superlocalmemory.cli import setup_wizard, version_banner

    ensure = Mock(return_value=True)
    monkeypatch.setattr(daemon, "ensure_daemon", ensure)
    monkeypatch.setattr(daemon, "is_daemon_running", Mock(return_value=False))
    monkeypatch.setattr(setup_wizard, "check_first_use", Mock())
    monkeypatch.setattr(version_banner, "check_and_emit_upgrade_banner", Mock(return_value=False))
    monkeypatch.setattr("sys.argv", ["slm", "serve"])

    main_module.main()

    ensure.assert_called_once_with()
    assert "Starting SLM daemon" in capsys.readouterr().out
