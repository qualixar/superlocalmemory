"""Grant keys live in the OS secure store only; failures mean no grant."""
from __future__ import annotations

import base64
import json

from superlocalmemory.remote_connections.credentials import SERVICE
from superlocalmemory.remote_connections.grant import GrantKeys
from superlocalmemory.remote_connections.grant_keys import GrantKeyStore

CID = "a" * 32


def key_b64(seed: int) -> str:
    return base64.urlsafe_b64encode(bytes([seed]) * 32).rstrip(b"=").decode()


class Backend:
    def __init__(self):
        self.values = {}

    def get_password(self, service, name):
        return self.values.get((service, name))

    def set_password(self, service, name, value):
        self.values[(service, name)] = value


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def make(backend=None, clock=None):
    backend = backend if backend is not None else Backend()
    clock = clock or Clock()
    return GrantKeyStore(lambda: backend, clock=clock), backend, clock


def test_nothing_stored_means_no_current_key():
    store, backend, _ = make()
    assert store.load(CID) == GrantKeys(None, None)
    assert backend.values == {}


def test_store_and_load_round_trip_uses_the_keyring_entry_shape():
    store, backend, _ = make()
    store.store_new(CID, 1, key_b64(1))
    assert store.load(CID) == GrantKeys((1, bytes([1]) * 32), None)
    raw = backend.values[(SERVICE, f"grant:{CID}")]
    assert json.loads(raw) == {"v": 1, "version": 1, "key": key_b64(1), "previous": None}
    fresh = GrantKeyStore(lambda: backend, clock=Clock())
    assert fresh.load(CID).current == (1, bytes([1]) * 32)


def test_rotation_keeps_the_previous_key_for_two_minutes():
    store, backend, clock = make()
    store.store_new(CID, 1, key_b64(1))
    store.store_new(CID, 2, key_b64(2))
    keys = store.load(CID)
    assert keys.current == (2, bytes([2]) * 32)
    assert keys.previous == (1, bytes([1]) * 32, 1120.0)
    clock.now = 1121.0
    fresh = GrantKeyStore(lambda: backend, clock=clock)
    assert fresh.load(CID).previous is None
    assert fresh.load(CID).current[0] == 2


def test_storing_the_same_version_again_is_a_noop():
    store, backend, _ = make()
    store.store_new(CID, 1, key_b64(1))
    before = dict(backend.values)
    store.store_new(CID, 1, key_b64(1))
    assert backend.values == before


def test_invalid_input_is_refused_and_nothing_is_written():
    store, backend, _ = make()
    for version, key in ((0, key_b64(1)), (True, key_b64(1)), (1, "short"),
                         (1, "!" * 43), (2 ** 53, key_b64(1))):
        try:
            store.store_new(CID, version, key)
        except ValueError:
            continue
        raise AssertionError("accepted invalid grant key")
    try:
        store.store_new("not-a-connection", 1, key_b64(1))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted invalid connection id")
    assert backend.values == {}


def test_backend_failure_means_no_grant_and_never_a_file(tmp_path, monkeypatch):
    class Broken(Backend):
        def get_password(self, service, name):
            raise RuntimeError("locked")

        def set_password(self, service, name, value):
            raise RuntimeError("locked")

    monkeypatch.chdir(tmp_path)
    store, _, _ = make(Broken())
    assert store.load(CID) == GrantKeys(None, None)
    try:
        store.store_new(CID, 1, key_b64(1))
    except ValueError as error:
        assert error.args[0] == "grant_key_store_unavailable"
    else:
        raise AssertionError("a failed keyring write must be reported")
    assert list(tmp_path.iterdir()) == []


def test_factory_that_cannot_open_the_keyring_means_no_grant():
    def broken():
        raise RuntimeError("no keyring")

    store = GrantKeyStore(broken, clock=Clock())
    assert store.load(CID) == GrantKeys(None, None)


def test_corrupt_stored_value_means_no_grant():
    store, backend, _ = make()
    backend.values[(SERVICE, f"grant:{CID}")] = "{not json"
    assert store.load(CID) == GrantKeys(None, None)
    backend.values[(SERVICE, f"grant:{CID}")] = json.dumps(
        {"v": 1, "version": 1, "key": "x", "previous": None})
    assert GrantKeyStore(lambda: backend, clock=Clock()).load(CID).current is None


def test_forget_removes_the_key_with_or_without_delete_support():
    store, backend, _ = make()
    store.store_new(CID, 1, key_b64(1))
    store.forget(CID)
    assert store.load(CID).current is None
    assert GrantKeyStore(lambda: backend, clock=Clock()).load(CID).current is None

    class Deleting(Backend):
        def delete_password(self, service, name):
            self.values.pop((service, name), None)

    store, backend, _ = make(Deleting())
    store.store_new(CID, 1, key_b64(1))
    store.forget(CID)
    assert backend.values == {}


def test_connections_do_not_share_keys():
    store, _, _ = make()
    store.store_new(CID, 1, key_b64(1))
    assert store.load("b" * 32).current is None


def test_secret_never_appears_in_repr_or_error_text():
    store, _, _ = make()
    store.store_new(CID, 1, key_b64(7))
    assert key_b64(7) not in repr(store.load(CID))
