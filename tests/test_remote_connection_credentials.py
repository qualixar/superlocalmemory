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
