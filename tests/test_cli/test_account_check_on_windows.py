# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""On a shared Windows computer, another account's daemon is not ours.

Windows has no uid, and the account check answered "this account" for every
process there. So a stale ``daemon.pid`` naming another account's old daemon
was adopted (seen on the Windows CI runner). Windows names the account a
process runs as (``DOMAIN\\user``); that is compared instead. Runs anywhere by
removing ``os.getuid`` the way Windows lacks it.
"""

from __future__ import annotations

import getpass
import os

import pytest

from superlocalmemory.cli import daemon as cli_daemon


class _Process:
    def __init__(self, username) -> None:
        self._username = username

    def username(self):
        if isinstance(self._username, Exception):
            raise self._username
        return self._username


@pytest.fixture()
def windows_account(monkeypatch):
    """No uid, and this process runs as ``OFFICE-PC\\<this user>``."""
    from superlocalmemory.core import platform_utils

    account = f"OFFICE-PC\\{getpass.getuser()}"
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.setattr(platform_utils, "current_account", lambda: account)
    return account


def test_a_process_of_this_account_is_ours(windows_account):
    assert cli_daemon._process_is_this_account(_Process(windows_account)) is True
    # Windows account names are not case-sensitive.
    assert cli_daemon._process_is_this_account(_Process(windows_account.upper())) is True


@pytest.mark.parametrize("other", ["OFFICE-PC\\someone-else", "OTHER-PC\\{me}"])
def test_another_accounts_process_is_not_ours(windows_account, other):
    # The user name comes from the fixture: with os.getuid removed,
    # getpass.getuser() fails where no USER/LOGNAME is set.
    other = other.format(me=windows_account.split("\\", 1)[1])
    assert cli_daemon._process_is_this_account(_Process(other)) is False


def test_a_process_whose_account_cannot_be_read_is_not_ours(windows_account):
    import psutil

    denied = psutil.AccessDenied(pid=4242)
    assert cli_daemon._process_is_this_account(_Process(denied)) is False


def test_the_windows_account_is_read_without_username_in_the_environment(monkeypatch):
    """A daemon started with a stripped environment has no USERNAME, and
    ``getpass.getuser()`` raises there; the account comes from the process
    itself instead (seen on the Windows runner: the daemon could not start)."""
    import psutil

    from superlocalmemory.core import platform_utils
    from superlocalmemory.infra.daemon_identity import owner_id

    for name in ("USERNAME", "USER", "LOGNAME", "LNAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(platform_utils.sys, "platform", "win32")
    monkeypatch.delattr(os, "getuid", raising=False)

    class _Me:
        def __init__(self, _pid) -> None:
            pass

        def username(self):
            return "OFFICE-PC\\alice"

    monkeypatch.setattr(psutil, "Process", _Me)
    assert platform_utils.current_account() == "OFFICE-PC\\alice"
    assert owner_id() == "user:alice"
