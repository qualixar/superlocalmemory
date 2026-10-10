"""Credential delivery probes use an injected fake; never the live OS keychain."""
from dataclasses import replace
import json
import pytest

try:
    from superlocalmemory.remote_connections.credentials import CredentialVault, ConnectorCredential, CredentialError
except ImportError:
    CredentialVault = ConnectorCredential = None
    CredentialError = RuntimeError

class Backend:
    def __init__(self): self.values = {}
    def get_password(self, service, key): return self.values.get((service, key))
    def set_password(self, service, key, value): self.values[(service, key)] = value

@pytest.fixture
def vault(tmp_path):
    assert CredentialVault is not None
    backend = Backend()
    return CredentialVault(tmp_path / "private", backend=backend, clock=lambda: 1000), backend

def credential(**changes):
    assert ConnectorCredential is not None
    value = ConnectorCredential(installation_id="install-a", owner="owner-a", profile="default",
        connection_id="a" * 32, generation=1, expires_at_ms=2000000,
        device_token="syntheticDevice" * 4, origin_key="slmr_" + "a" * 43)
    return replace(value, **changes)

def test_credential_roundtrip_is_private_and_binding_scoped(vault):
    store, backend = vault
    value = credential(); store.save(value)
    assert store.load("install-a", "owner-a", "default", "a" * 32) == value
    assert store.load("install-a", "other", "default", "a" * 32) is None
    assert store.load("install-a", "owner-a", "other", "a" * 32) is None
    assert value.device_token not in repr(value) and value.origin_key not in repr(value)
    assert not list(store.root.glob("*.json"))
    assert backend.values

def test_missing_credentials_are_not_synthesized(vault):
    store, _ = vault
    assert store.load("install-a", "owner-a", "default", "a" * 32) is None

@pytest.mark.parametrize("changes", [
    {"device_token":"short"}, {"origin_key":"global-api-key"}, {"generation":True},
    {"generation":0}, {"expires_at_ms":True}, {"expires_at_ms":1000000},
    {"connection_id":"../../file"}, {"profile":""},
])
def test_bad_or_expired_credentials_never_enter_storage(vault, changes):
    store, backend = vault
    with pytest.raises(CredentialError): store.save(credential(**changes))
    assert not backend.values

def test_generation_rotation_cannot_replay_or_replace_same_generation(vault):
    store, _ = vault; first = credential(); store.save(first); store.save(first)
    with pytest.raises(CredentialError, match="credential_conflict"):
        store.save(replace(first, device_token="b" * 64))
    newer = replace(first, generation=2, device_token="b" * 64); store.save(newer)
    with pytest.raises(CredentialError, match="credential_stale"): store.save(first)
    assert store.load("install-a","owner-a","default","a"*32) == newer

def test_revocation_is_terminal_without_retaining_secret(vault):
    store, backend = vault; value = credential(); store.save(value)
    store.revoke("install-a","owner-a","default","a"*32)
    assert store.load("install-a","owner-a","default","a"*32) is None
    assert value.device_token not in str(backend.values) and value.origin_key not in str(backend.values)
    with pytest.raises(CredentialError, match="credential_revoked"): store.save(replace(value,generation=2))

def test_storage_failure_never_leaks_secret_or_falls_back_to_plaintext(vault):
    store, backend = vault
    def fail(*args): raise OSError("SECRET backend details")
    backend.set_password = fail
    with pytest.raises(CredentialError, match="credential_store_unavailable") as failure: store.save(credential())
    assert "SECRET" not in str(failure.value) and failure.value.__cause__ is None
    assert not list(store.root.glob("*.json"))

def test_corrupt_or_foreign_keychain_record_is_refused(vault):
    store, backend = vault; store.save(credential())
    key = next(iter(backend.values)); raw = json.loads(backend.values[key]); raw["profile"] = "other"
    backend.values[key] = json.dumps(raw)
    with pytest.raises(CredentialError): store.load("install-a","owner-a","default","a"*32)
    backend.values[key] = "{" 
    with pytest.raises(CredentialError): store.load("install-a","owner-a","default","a"*32)

def test_unavailable_secure_backend_affects_remote_only(tmp_path, monkeypatch):
    assert CredentialVault is not None
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,"keyring",SimpleNamespace(get_keyring=lambda: object()))
    with pytest.raises(CredentialError,match="secure_keyring_unavailable"): CredentialVault(tmp_path/"private")


def test_concurrent_readers_in_one_process_wait_their_turn(tmp_path):
    """The connector and its renewal scheduler read the vault at the same moment
    when a link starts. Each makes its own vault object; both must succeed."""
    from concurrent.futures import ThreadPoolExecutor
    backend = Backend()
    CredentialVault(tmp_path, backend=backend, clock=lambda: 1000).save(credential())
    held = credential()
    identity = (held.installation_id, held.owner, held.profile, held.connection_id)

    def read(_):
        return CredentialVault(tmp_path, backend=backend, clock=lambda: 1000).load(*identity)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(read, range(200)))
    assert all(r == held for r in results)


def test_a_vault_held_by_another_process_is_reported_busy(tmp_path):
    """Another process holding the lock is a temporary condition, reported as such."""
    import multiprocessing

    from superlocalmemory.core.file_lock import exclusive_lock
    from superlocalmemory.remote_connections.credentials import CredentialError
    store = CredentialVault(tmp_path, backend=Backend(), clock=lambda: 1000)
    held = credential()
    # Events from the same start method as the process: Linux defaults to fork
    # before 3.14, and a fork-context lock cannot be handed to a spawned child.
    spawn = multiprocessing.get_context("spawn")
    ready, release = spawn.Event(), spawn.Event()
    holder = spawn.Process(
        target=_hold_lock, args=(str(tmp_path / "credential.lock"), ready, release))
    holder.start()
    try:
        assert ready.wait(10)
        with pytest.raises(CredentialError, match="credential_store_busy"):
            store.load(held.installation_id, held.owner, held.profile, held.connection_id)
    finally:
        release.set()
        holder.join(10)
    assert exclusive_lock  # imported for the child's use


def _hold_lock(path, ready, release):
    from pathlib import Path

    from superlocalmemory.core.file_lock import exclusive_lock

    with exclusive_lock(Path(path)):
        ready.set()
        release.wait(10)
