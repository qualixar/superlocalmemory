# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Named remote keys: hashed at rest, revocable, fail closed on a tampered store."""

from __future__ import annotations

import hmac
import json
import os
import stat

import pytest

from superlocalmemory.server import remote_keys
from superlocalmemory.server.remote_keys import RemoteKeyError, RemoteKeyStore


@pytest.fixture()
def store(tmp_path):
    return RemoteKeyStore(tmp_path / "remote_keys.json")


def test_add_verify_revoke_list_round_trip(store) -> None:
    record, secret = store.add("hermes-laptop", "write", profile="default")
    assert secret.startswith("slmr_") and len(secret) == 5 + 43
    assert store.verify(secret) == record
    assert store.verify(secret[:-1] + ("A" if secret[-1] != "A" else "B")) is None
    revoked = store.revoke("hermes-laptop")
    assert revoked.revoked_at is not None
    assert store.verify(secret) is None
    listed = store.list()
    assert [k.name for k in listed] == ["hermes-laptop"] and not listed[0].active


def test_only_a_domain_separated_digest_is_stored(store) -> None:
    import hashlib

    _, secret = store.add("a", "read", profile="default")
    raw = store.path.read_text(encoding="utf-8")
    assert secret not in raw
    assert hashlib.sha256(secret.encode()).hexdigest() not in raw
    assert remote_keys.digest_secret(secret) in raw


def test_revoke_by_key_id(store) -> None:
    record, secret = store.add("a", "read", profile="default")
    store.revoke(record.key_id)
    assert store.verify(secret) is None


def test_duplicate_active_name_refused_but_reusable_after_revoke(store) -> None:
    store.add("a", "read", profile="default")
    with pytest.raises(RemoteKeyError) as err:
        store.add("a", "write", profile="default")
    assert err.value.code == "duplicate_name"
    store.revoke("a")
    store.add("a", "write", profile="default")


@pytest.mark.parametrize("name", ["", "A", "-x", "a b", "a/b", "x" * 49, "név"])
def test_invalid_names_refused(store, name) -> None:
    with pytest.raises(RemoteKeyError):
        store.add(name, "read", profile="default")


def test_invalid_scope_refused(store) -> None:
    with pytest.raises(RemoteKeyError):
        store.add("a", "admin", profile="default")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_atomic_write_leaves_0600(store) -> None:
    store.add("a", "read", profile="default")
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert not [p for p in store.path.parent.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
@pytest.mark.parametrize("mode", [0o620, 0o602, 0o640, 0o604])
def test_group_or_world_accessible_store_fails_closed(store, mode, caplog) -> None:
    _, secret = store.add("a", "write", profile="default")
    os.chmod(store.path, mode)
    assert store.verify(secret) is None
    assert "Remote keys are disabled" in caplog.text
    with pytest.raises(RemoteKeyError) as err:
        store.add("b", "read", profile="default")
    assert err.value.code == "store_untrusted"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_wrong_owner_fails_closed(store, monkeypatch) -> None:
    _, secret = store.add("a", "write", profile="default")
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    assert store.verify(secret) is None


def test_unknown_version_or_corrupt_store_fails_closed(store) -> None:
    _, secret = store.add("a", "write", profile="default")
    data = json.loads(store.path.read_text(encoding="utf-8"))
    data["version"] = 99
    store.path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(store.path, 0o600)
    assert store.verify(secret) is None
    store.path.write_text("{not json", encoding="utf-8")
    assert store.verify(secret) is None


def test_verify_checks_every_record(store, monkeypatch) -> None:
    secrets = [store.add(f"k{i}", "read", profile="default")[1] for i in range(5)]
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(remote_keys.hmac, "compare_digest", spy)
    assert store.verify(secrets[0]) is not None
    assert len(calls) == 5


@pytest.mark.parametrize("presented", ["", "slmr_", "Bearer x", "slmr_" + "A" * 42,
                                       "slmr_" + "A" * 44, "xxxxx" + "A" * 43, None, 7])
def test_malformed_presented_keys_are_refused(store, presented) -> None:
    store.add("a", "write", profile="default")
    assert store.verify(presented) is None


def test_revocation_takes_effect_without_a_new_store_object(store) -> None:
    """Another process (the CLI) revokes; this store object sees it on the next call."""
    _, secret = store.add("a", "write", profile="default")
    assert store.verify(secret) is not None
    RemoteKeyStore(store.path).revoke("a")
    assert store.verify(secret) is None


# -- every key is bound to one profile (audit 4.1.20 L2 F2) ------------------------------


def _write_v1_store(store, secrets_by_name: dict[str, str], revoked: set[str] = frozenset()):
    """A key store exactly as 4.1.19 wrote it: no profile field."""
    keys = [{"key_id": f"rk_{i:08x}", "name": name, "scope": "write",
             "digest": remote_keys.digest_secret(secret), "created_at": "2026-09-01T00:00:00+00:00",
             "revoked_at": "2026-09-02T00:00:00+00:00" if name in revoked else None}
            for i, (name, secret) in enumerate(secrets_by_name.items())]
    store.path.write_text(json.dumps({"version": 1, "keys": keys}), encoding="utf-8")
    os.chmod(store.path, 0o600)


def _secret() -> str:
    import secrets as _s

    return remote_keys.KEY_PREFIX + _s.token_urlsafe(32)


def test_a_key_cannot_be_made_without_a_valid_profile(store) -> None:
    with pytest.raises(TypeError):
        store.add("a", "read")  # type: ignore[call-arg]
    for bad in ("", "a b", "../x", "x" * 65, None):
        with pytest.raises(RemoteKeyError) as err:
            store.add("a", "read", profile=bad)  # type: ignore[arg-type]
        assert err.value.code == "invalid_profile"


def test_the_profile_is_stored_listed_and_returned_by_verify(store) -> None:
    record, secret = store.add("a", "read", profile="clientx")
    assert store.verify(secret).profile == "clientx"
    assert record.public()["profile"] == "clientx"
    assert record.public()["profile_source"] == "chosen"
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert data["version"] == 2 and data["keys"][0]["profile"] == "clientx"


def test_a_pre_4_1_20_store_is_read_and_its_keys_are_unbound(store) -> None:
    s = _secret()
    _write_v1_store(store, {"old": s})
    key = store.verify(s)
    assert key is not None and key.profile is None


def test_bind_unbound_binds_active_old_keys_once_and_says_how(store) -> None:
    old, gone = _secret(), _secret()
    _write_v1_store(store, {"old": old, "gone": gone}, revoked={"gone"})
    bound = store.bind_unbound("work")
    assert [k.name for k in bound] == ["old"]
    key = store.verify(old)
    assert key.profile == "work" and key.profile_source == "bound-on-upgrade"
    assert store.bind_unbound("other") == ()  # once only: a later switch changes nothing
    assert store.verify(old).profile == "work"
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    revoked = next(k for k in data["keys"] if k["name"] == "gone")
    assert revoked["revoked_at"] is not None and revoked.get("profile") is None


def test_binding_never_revives_a_revoked_key(store) -> None:
    old = _secret()
    _write_v1_store(store, {"old": old})
    RemoteKeyStore(store.path).revoke("old")
    store.bind_unbound("work")
    assert store.verify(old) is None


@pytest.mark.parametrize("profile, source", [("../etc", "chosen"), ("work", "made-up"),
                                             (5, "chosen")])
def test_a_tampered_binding_drops_the_key_rather_than_unbinding_it(store, profile,
                                                                  source) -> None:
    _, secret = store.add("a", "write", profile="work")
    data = json.loads(store.path.read_text(encoding="utf-8"))
    data["keys"][0].update(profile=profile, profile_source=source)
    store.path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(store.path, 0o600)
    assert store.verify(secret) is None


def test_the_gate_refuses_an_unbound_key(store) -> None:
    from types import SimpleNamespace

    from superlocalmemory.server.remote_access import gate_remote_mcp

    old = _secret()
    _write_v1_store(store, {"old": old})
    scope = {"scheme": "https", "client": ("192.168.1.9", 1)}
    state = SimpleNamespace(rbac=None)
    decision = gate_remote_mcp(scope, {"authorization": f"Bearer {old}"}, state, store)
    assert decision.status == 403 and decision.body["error"] == "remote_key_unbound"
    store.bind_unbound("work")
    decision = gate_remote_mcp(scope, {"authorization": f"Bearer {old}"}, state, store)
    assert decision.allowed and decision.principal.profile == "work"


# -- opt-in extras (mesh, media) -------------------------------------------------------


def test_a_new_key_has_no_extras_and_none_are_written(store) -> None:
    record, _ = store.add("a", "write", profile="default")
    assert record.extras == frozenset()
    assert "extras" not in json.loads(store.path.read_text(encoding="utf-8"))["keys"][0]
    assert record.public()["extras"] == []


def test_extras_round_trip_and_are_sorted_in_public(store) -> None:
    store.add("a", "write", profile="default")
    changed = store.set_extras("a", {"media", "mesh"})
    assert changed.extras == frozenset({"mesh", "media"})
    reloaded = RemoteKeyStore(store.path).list()[0]
    assert reloaded.extras == frozenset({"mesh", "media"})
    assert reloaded.public()["extras"] == ["media", "mesh"]
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["version"] == 2 and raw["keys"][0]["extras"] == ["media", "mesh"]
    store.set_extras("a", set())
    assert "extras" not in json.loads(store.path.read_text(encoding="utf-8"))["keys"][0]


def test_unknown_extras_are_dropped_on_read(store) -> None:
    store.add("a", "write", profile="default")
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    raw["keys"][0]["extras"] = ["mesh", "root", 5, None]
    store.path.write_text(json.dumps(raw), encoding="utf-8")
    os.chmod(store.path, 0o600)
    assert RemoteKeyStore(store.path).list()[0].extras == frozenset({"mesh"})
    raw["keys"][0]["extras"] = "mesh"
    store.path.write_text(json.dumps(raw), encoding="utf-8")
    assert RemoteKeyStore(store.path).list()[0].extras == frozenset()


def test_a_file_written_before_extras_existed_reads_fine(store) -> None:
    store.add("a", "read", profile="default")
    assert store.list()[0].extras == frozenset()


def test_set_extras_refuses_unknown_values_revoked_and_missing_keys(store) -> None:
    store.add("a", "write", profile="default")
    with pytest.raises(RemoteKeyError) as err:
        store.set_extras("a", {"admin"})
    assert err.value.code == "invalid_extra"
    with pytest.raises(RemoteKeyError) as err:
        store.set_extras("nope", {"mesh"})
    assert err.value.code == "not_found"
    store.revoke("a")
    with pytest.raises(RemoteKeyError):
        store.set_extras("a", {"mesh"})


def test_revoking_keeps_extras_off_a_replacement_key(store) -> None:
    store.add("a", "write", profile="default")
    store.set_extras("a", {"mesh"})
    store.revoke("a")
    record, _ = store.add("a", "write", profile="default")
    assert record.extras == frozenset()


def test_the_extras_cache_is_dropped_when_the_file_is_replaced_in_place(store) -> None:
    kid = store.add("a", "read", profile="default")[0].key_id
    store.set_extras("a", {"mesh"})
    assert store.extras_for(kid) == frozenset({"mesh"})
    assert store.cached_extras(kid) == frozenset({"mesh"})
    before = store.path.stat()
    replacement = store.path.read_text(encoding="utf-8").replace("mesh", "meSh")
    other = store.path.with_name("other.json")
    other.write_text(replacement, encoding="utf-8")
    os.chmod(other, 0o600)
    os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(other, store.path)
    assert os.stat(store.path).st_mtime_ns == before.st_mtime_ns
    assert store.cached_extras(kid) is None


def test_the_file_signature_includes_inode_and_change_time(store) -> None:
    store.add("a", "read", profile="default")
    info = store.path.stat()
    signature = store._file_signature()
    assert info.st_ino in signature and info.st_ctime_ns in signature
