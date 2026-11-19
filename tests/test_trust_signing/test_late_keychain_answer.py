# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-07 and F-11(d): a keychain answer that comes after the 5 s bound is picked up later.

The daemon starts in the background at login and nobody waits for it, so a person may answer the
macOS dialog a minute late. Before 4.2.30's fix that allowed the access and still got no signing
until the next daemon start. Now the signing maintenance step (and only it) collects the late
answer with a growing wait between looks. A hook, a recall and a CLI read command never ask.
"""

from __future__ import annotations

import threading
import time

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.trust import signatures
from superlocalmemory.trust import signing_key as sk
from tests.helpers.flag_baseline import off_except
from tests.test_trust_signing.conftest import SIGNING, FakeKeychain, save


class SlowKeychain(FakeKeychain):
    """Answers only when the gate is opened: the person has not clicked yet."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()

    def get_password(self, service, username):
        self.reads += 1
        self.gate.wait(10)
        return self.entries.get((service, username))


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _on(tmp_path):
    flags = lambda home: FeatureFlags(  # noqa: E731
        home, environ={"SLM_FEATURE_TRUST_SIGNING": "on"}, file_values=off_except())
    return flags


def _start_late(tmp_path, keychain, clock):
    store = sk.KeychainKeyStore(keychain)
    status = sk.start_for_daemon(tmp_path, store=store, timeout=0.15, flags=_on(tmp_path),
                                 clock=clock)
    assert status == sk.UNAVAILABLE_TIMEOUT and sk.current() is None
    return store


def _wait_until_done(keychain, *, creates=True):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not creates or keychain.entries:
            return
        time.sleep(0.02)
    raise AssertionError("the slow read never finished")


def test_a_late_answer_is_picked_up_by_the_retry_without_a_second_keychain_call(tmp_path):
    kc, clock = SlowKeychain(), Clock()
    _start_late(tmp_path, kc, clock)
    kc.gate.set()  # the person clicks Allow, a minute late
    _wait_until_done(kc)
    reads_before = kc.reads
    assert sk.retry_late_key(clock=clock) is None and sk.current() is None  # the wait has not passed
    clock.t += 31
    signer = sk.retry_late_key(clock=clock)
    assert signer is not None and sk.current() is signer and sk.status() == sk.READY
    # the retry took the answer of the read that was already in flight: it asked nobody again
    assert kc.reads == reads_before


def test_the_retry_waits_with_a_growing_gap_and_never_asks_while_the_first_read_hangs(tmp_path):
    kc, clock = SlowKeychain(), Clock()
    _start_late(tmp_path, kc, clock)
    assert sk.retry_late_key(clock=clock) is None  # too soon: nothing happens at all
    gaps = []
    for _ in range(4):
        clock.t += 100_000  # always past the wait
        before = sk._next_try
        assert sk.retry_late_key(clock=clock) is None  # still hung: looked, did not ask again
        gaps.append(sk._next_try - clock.t)
        assert sk._next_try > before
    assert kc.reads == 1  # one dialog only, however long the person takes
    assert gaps == sorted(gaps) and gaps[-1] > gaps[0]
    assert gaps[-1] <= sk.RETRY_MAX_WAIT_S
    kc.gate.set()


def test_a_refusal_that_arrives_late_ends_the_retries(tmp_path):
    class Refuses(SlowKeychain):
        def get_password(self, service, username):
            self.reads += 1
            self.gate.wait(10)
            raise RuntimeError("user denied")

    kc, clock = Refuses(), Clock()
    _start_late(tmp_path, kc, clock)
    kc.gate.set()
    time.sleep(0.3)
    clock.t += 31
    assert sk.retry_late_key(clock=clock) is None
    assert sk.status() == sk.UNAVAILABLE_UNREADABLE
    clock.t += 100_000
    assert sk.retry_late_key(clock=clock) is None and kc.reads == 1  # not asked again


def test_the_retry_does_nothing_when_the_status_is_not_a_timeout(tmp_path):
    clock = Clock()
    for status in (sk.OFF, sk.UNAVAILABLE_STORE, sk.UNAVAILABLE_UNREADABLE, sk.READY):
        sk.install(None, status)
        clock.t += 100_000
        assert sk.retry_late_key(store=_Boom(), clock=clock) is None


class _Boom:
    def read(self):
        raise AssertionError("the keychain must not be asked")

    create = read


def test_the_maintenance_step_signs_after_a_late_key_arrives(db, monkeypatch, tmp_path):
    monkeypatch.setenv(SIGNING, "on")
    kc, clock = SlowKeychain(), Clock()
    _start_late(tmp_path, kc, clock)
    save(db, "Saved while the keychain dialog was still open.")
    monkeypatch.setattr(sk, "_clock", clock)
    assert signatures.maintenance_step(db) == 0  # too soon, no key
    kc.gate.set()
    _wait_until_done(kc)
    clock.t += 31
    assert signatures.maintenance_step(db) == 1  # picked up, and the waiting fact signed
    assert db.execute("SELECT COUNT(*) AS n FROM fact_signatures")[0]["n"] == 1


# -- F-11(d): the maintenance step reads the keychain only for that retry --------------------------

def test_the_maintenance_step_without_a_key_and_without_a_late_read_never_reads_the_keychain(
        db, monkeypatch):
    monkeypatch.setenv(SIGNING, "on")
    save(db, "No key anywhere.")

    def refuse(*_a, **_k):
        raise AssertionError("the keychain was read from the maintenance step")

    monkeypatch.setattr(sk, "load", refuse)
    monkeypatch.setattr(sk.KeychainKeyStore, "read", refuse)
    monkeypatch.setattr(sk.KeychainKeyStore, "create", refuse)
    for status in (sk.OFF, sk.UNAVAILABLE_STORE, sk.UNAVAILABLE_UNREADABLE, sk.NO_KEY):
        sk.install(None, status)
        assert signatures.maintenance_step(db) == 0


def test_the_maintenance_step_does_not_retry_while_the_flag_is_off(db, monkeypatch):
    sk.install(None, sk.UNAVAILABLE_TIMEOUT)
    monkeypatch.setattr(sk, "retry_late_key", lambda **_: (_ for _ in ()).throw(
        AssertionError("retried with the flag off")))
    assert signatures.maintenance_step(db) == 0
