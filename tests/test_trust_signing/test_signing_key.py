# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-16: the key lives in the keychain only; unavailable means unavailable; read once, bounded."""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.trust import signing_key as sk
from tests.test_trust_signing.conftest import FakeKeychain


def test_a_signer_needs_exactly_32_bytes_and_never_shows_its_key() -> None:
    with pytest.raises(ValueError):
        sk.Signer(b"short")
    s = sk.Signer(bytes(range(32)))
    assert "000102" not in repr(s) and "key=" not in repr(s)
    assert len(s.key_id) == 16 and s.key_id == sk.key_id_of(bytes(range(32)))
    assert sk.Signer(bytes(32)).key_id != s.key_id


def test_mac_checks_and_rejects() -> None:
    s = sk.Signer(bytes(range(32)))
    assert s.matches(b"data", s.mac(b"data"))
    assert not s.matches(b"datb", s.mac(b"data"))
    assert not sk.Signer(bytes(32)).matches(b"data", s.mac(b"data"))


def test_create_stores_the_key_only_in_the_keychain_and_reads_it_back(keychain) -> None:
    store = sk.KeychainKeyStore(keychain)
    assert store.read() is None
    key = store.create()
    assert len(key) == 32 and store.read() == key
    ((service, account), text), = keychain.entries.items()
    assert (service, account) == (sk.SERVICE, sk.ACCOUNT) and key.hex() not in text


def test_create_never_overwrites_an_existing_key(keychain) -> None:
    store = sk.KeychainKeyStore(keychain)
    first = store.create()
    with pytest.raises(sk.StoreUnavailable):
        store.create()
    assert store.read() == first


@pytest.mark.parametrize("junk", ["", "!!!", "QUJD"])  # empty, not base64, wrong length
def test_an_entry_that_is_not_a_key_is_unreadable_never_replaced(junk) -> None:
    fake = FakeKeychain({(sk.SERVICE, sk.ACCOUNT): junk})
    result = sk.load(sk.KeychainKeyStore(fake), create=True)
    assert result.signer is None and result.status == sk.UNAVAILABLE_UNREADABLE
    assert fake.entries[(sk.SERVICE, sk.ACCOUNT)] == junk  # a lost key is not "fixed" by overwriting


def test_the_store_itself_refuses_an_entry_of_the_wrong_length(keychain) -> None:
    keychain.entries[(sk.SERVICE, sk.ACCOUNT)] = "QUJD"  # three bytes, valid base64
    with pytest.raises(sk.StoreUnavailable) as exc:
        sk.KeychainKeyStore(keychain).read()
    assert exc.value.status == sk.UNAVAILABLE_UNREADABLE


def test_load_without_create_reports_no_key(keychain) -> None:
    assert sk.load(sk.KeychainKeyStore(keychain)).status == sk.NO_KEY
    assert keychain.entries == {}


def test_load_with_create_makes_one_key_and_then_reuses_it(keychain) -> None:
    store = sk.KeychainKeyStore(keychain)
    first = sk.load(store, create=True)
    second = sk.load(store, create=True)
    assert first.ready and second.ready and first.signer.key_id == second.signer.key_id


def test_no_supported_keychain_means_signing_is_unavailable_and_no_file_is_written(
        monkeypatch, tmp_path) -> None:
    from superlocalmemory.remote_connections import credentials

    def none():
        raise credentials.CredentialError("secure_keyring_unavailable")

    monkeypatch.setattr(credentials, "_native_backend", none)
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    result = sk.load(create=True)
    assert result.signer is None and result.status == sk.UNAVAILABLE_STORE
    assert sorted(p.name for p in tmp_path.rglob("*")) == before  # never a fallback key file
    assert "no key file" in sk.words(sk.UNAVAILABLE_STORE)


def test_a_keychain_that_hangs_is_bounded_by_the_timeout() -> None:
    gate = threading.Event()

    class Hangs:
        def get_password(self, *_):
            gate.wait(10)

        def set_password(self, *_):
            gate.wait(10)

    started = time.monotonic()
    result = sk.load(sk.KeychainKeyStore(Hangs()), create=True, timeout=0.2)
    gate.set()
    assert result.status == sk.UNAVAILABLE_TIMEOUT and result.signer is None
    assert time.monotonic() - started < 3


def test_an_unexpected_error_in_the_store_is_unreadable_not_a_crash() -> None:
    class Boom:
        def read(self):
            raise OSError("secret text that must not be echoed")

    result = sk.load(Boom())
    assert result.status == sk.UNAVAILABLE_UNREADABLE and "secret" not in sk.words(result.status)


# -- the daemon's one read ----------------------------------------------------------------------

class Counting:
    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        raise AssertionError("the keychain must not be asked")


def _flags(**env):
    return lambda home: FeatureFlags(home, environ=env, file_values={})


def test_with_the_flag_off_the_keychain_is_not_asked(tmp_path) -> None:
    store = Counting()
    assert sk.start_for_daemon(tmp_path, store=store, flags=_flags()) == sk.OFF
    assert store.reads == 0 and sk.current() is None and sk.status() == sk.OFF


def test_the_kill_switch_means_the_keychain_is_not_asked(tmp_path) -> None:
    store = Counting()
    from superlocalmemory.core.feature_flags import KILL_ENV

    env = {"SLM_FEATURE_TRUST_SIGNING": "on", KILL_ENV: "on"}
    assert sk.start_for_daemon(tmp_path, store=store, flags=_flags(**env)) == sk.OFF
    assert store.reads == 0 and sk.current() is None


def test_with_the_flag_on_the_daemon_holds_the_key_in_memory(tmp_path, keychain) -> None:
    status = sk.start_for_daemon(tmp_path, store=sk.KeychainKeyStore(keychain),
                                 flags=_flags(SLM_FEATURE_TRUST_SIGNING="on"))
    assert status == sk.READY and sk.current() is not None
    reads = keychain.reads
    assert sk.current() is sk.current() and keychain.reads == reads  # memory only afterwards


def test_shadow_signs_too_because_it_writes_only_its_own_table(tmp_path, keychain) -> None:
    status = sk.start_for_daemon(tmp_path, store=sk.KeychainKeyStore(keychain),
                                 flags=_flags(SLM_FEATURE_TRUST_SIGNING="shadow"))
    assert status == sk.READY


def test_an_unavailable_keychain_leaves_the_daemon_running_unsigned(tmp_path) -> None:
    class Gone:
        def read(self):
            raise sk.StoreUnavailable(sk.UNAVAILABLE_STORE)

    status = sk.start_for_daemon(tmp_path, store=Gone(),
                                 flags=_flags(SLM_FEATURE_TRUST_SIGNING="on"))
    assert status == sk.UNAVAILABLE_STORE and sk.current() is None


def test_a_broken_flag_reader_is_unavailable_not_a_crash(tmp_path) -> None:
    def broken(_home):
        raise RuntimeError("x")

    assert sk.start_for_daemon(tmp_path, flags=broken) == sk.UNAVAILABLE_UNREADABLE
