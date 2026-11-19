# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-02 (19 Nov 2026): the product never asks the OS keychain when that cannot work without a dialog.

A test daemon built its own environment, ``HOME`` pointed at a folder with no login keychain, and
macOS showed the person at the computer a "keychain not found" dialog. The test builders are now
guarded, and this is the second lock: whatever builds the environment, signing answers
"unavailable" without making an operating-system call when

* ``SLM_TEST_ISOLATION=1``, or
* on macOS the effective HOME has no login keychain (``login.keychain-db`` or ``login.keychain``).
"""

from __future__ import annotations

import sys

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.remote_connections import credentials
from superlocalmemory.trust import signing_key as sk
from tests.helpers.flag_baseline import off_except


@pytest.fixture
def os_is_never_asked(monkeypatch):
    """Any route to the native keyring fails the test."""
    calls: list[str] = []

    def asked():
        calls.append("native keyring opened")
        raise AssertionError("the operating-system keychain was asked")

    monkeypatch.setattr(credentials, "_native_backend", asked)
    import keyring

    monkeypatch.setattr(keyring, "get_keyring", asked)
    monkeypatch.setattr(keyring, "get_password", lambda *a, **k: asked())
    return calls


def _as_a_real_run(monkeypatch, tmp_path, *, platform: str):
    monkeypatch.delenv("SLM_TEST_ISOLATION", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", platform)


def test_under_test_isolation_signing_is_unavailable_and_the_os_is_not_asked(os_is_never_asked):
    # SLM_TEST_ISOLATION=1 is what the pytest plugin sets for every test run.
    result = sk.load(create=True)
    assert result.signer is None and result.status == sk.UNAVAILABLE_TEST_RUN
    assert os_is_never_asked == []


def test_the_daemon_start_under_test_isolation_asks_nothing(tmp_path, os_is_never_asked):
    flags = lambda home: FeatureFlags(  # noqa: E731
        home, environ={"SLM_FEATURE_TRUST_SIGNING": "on"}, file_values=off_except())
    assert sk.start_for_daemon(tmp_path, flags=flags) == sk.UNAVAILABLE_TEST_RUN
    assert sk.current() is None and os_is_never_asked == []


def test_on_a_mac_without_a_login_keychain_in_the_effective_home_it_is_not_asked(
        monkeypatch, tmp_path, os_is_never_asked):
    _as_a_real_run(monkeypatch, tmp_path, platform="darwin")
    (tmp_path / "Library" / "Keychains").mkdir(parents=True)  # the folder alone is not a keychain
    result = sk.load(create=True)
    assert result.signer is None and result.status == sk.UNAVAILABLE_NO_LOGIN_KEYCHAIN
    assert os_is_never_asked == []


@pytest.mark.parametrize("name", ["login.keychain-db", "login.keychain"])
def test_on_a_mac_with_a_login_keychain_the_guard_steps_aside(monkeypatch, tmp_path, name):
    _as_a_real_run(monkeypatch, tmp_path, platform="darwin")
    folder = tmp_path / "Library" / "Keychains"
    folder.mkdir(parents=True)
    (folder / name).write_bytes(b"")
    assert sk.keychain_blocked() is None


def test_off_a_mac_the_home_check_does_not_apply(monkeypatch, tmp_path):
    _as_a_real_run(monkeypatch, tmp_path, platform="linux")
    assert sk.keychain_blocked() is None


def test_an_injected_store_is_never_blocked(tmp_path, keychain):
    # Tests of keychain logic hand in their own fake; only the real, lazily opened one is guarded.
    assert sk.load(sk.KeychainKeyStore(keychain), create=True).ready


def test_both_statuses_say_in_plain_words_that_signing_is_unavailable():
    for status in (sk.UNAVAILABLE_TEST_RUN, sk.UNAVAILABLE_NO_LOGIN_KEYCHAIN):
        text = sk.words(status)
        assert text.startswith("Signing is unavailable") and status not in text
    assert "login keychain" in sk.words(sk.UNAVAILABLE_NO_LOGIN_KEYCHAIN)
