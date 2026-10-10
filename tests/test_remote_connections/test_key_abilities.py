"""The second yes from the dashboard: a connection key's mesh and media opt-ins (package D)."""

from __future__ import annotations

import asyncio

import pytest

from superlocalmemory.remote_connections.journal import JournalConflict
from superlocalmemory.remote_connections.runtime import NativeConnectionRuntime
from superlocalmemory.server.remote_keys import RemoteKeyStore

CID = "c" * 32


class Journal:
    """Owner 'me' holds connection CID on profile 'default'; nobody else holds anything."""

    def get(self, owner, profile, connection_id):
        if (owner, profile, connection_id) != ("me", "default", CID):
            raise JournalConflict("not_found")
        return object()


def _runtime(tmp_path, *, profile="default"):
    rt = object.__new__(NativeConnectionRuntime)
    rt.journal = Journal()
    rt.keys = RemoteKeyStore(tmp_path / "keys.json")
    rt._locks = {}
    rt.ended = []

    async def end(connection_id):
        rt.ended.append(connection_id)

    rt._end_upload_links = end
    rt.keys.add("web-" + CID, "write", profile=profile)
    return rt


def run(coro):
    return asyncio.run(coro)


def test_the_owner_turns_mesh_and_media_on_and_off_for_the_connection_key(tmp_path):
    rt = _runtime(tmp_path)
    assert run(rt.key_abilities("me", "default", CID)) == {"connection_id": CID, "mesh": False, "media": False}
    assert run(rt.set_key_ability("me", "default", CID, "mesh", True))["mesh"] is True
    assert run(rt.set_key_ability("me", "default", CID, "media", True)) == {
        "connection_id": CID, "mesh": True, "media": True}
    key = next(k for k in rt.keys.list() if k.name == "web-" + CID)
    assert set(key.extras) == {"mesh", "media"}          # the same opt-in `slm remote keys allow` sets
    after = run(rt.set_key_ability("me", "default", CID, "media", False))
    assert after["media"] is False and after["mesh"] is True
    assert rt.ended == [CID]                              # stopping pictures ends open upload links


@pytest.mark.parametrize("owner,profile,cid", [("someone", "default", CID), ("me", "work", CID),
                                               ("me", "default", "d" * 32)])
def test_another_owner_profile_or_connection_is_not_found(tmp_path, owner, profile, cid):
    rt = _runtime(tmp_path)
    with pytest.raises(ValueError, match="not_found"):
        run(rt.key_abilities(owner, profile, cid))
    with pytest.raises(ValueError, match="not_found"):
        run(rt.set_key_ability(owner, profile, cid, "mesh", True))


def test_a_key_bound_to_another_profile_is_never_changed(tmp_path):
    rt = _runtime(tmp_path, profile="work")
    with pytest.raises(ValueError, match="not_found"):
        run(rt.set_key_ability("me", "default", CID, "mesh", True))
    assert not next(k for k in rt.keys.list()).extras


def test_only_mesh_and_media_can_be_allowed(tmp_path):
    rt = _runtime(tmp_path)
    with pytest.raises(ValueError):
        run(rt.set_key_ability("me", "default", CID, "admin", True))
