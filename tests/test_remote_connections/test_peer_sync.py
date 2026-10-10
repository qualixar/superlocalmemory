"""Revoking a web app retires its mesh peer: the runtime syncs with the connected-apps list."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.mesh.broker import MeshBroker
from superlocalmemory.remote_connections import peer_names
from superlocalmemory.remote_connections import runtime as runtime_module
from superlocalmemory.remote_connections.grant import peer_ref
from tests.test_mesh.conftest import init_mesh_schema, make_peer, rows
from tests.test_remote_connections.test_grant_key_delivery import (
    FakeProvider,
    app_row,
    completed_runtime,
)


class SyncProvider(FakeProvider):
    """The connected-apps list as the gateway serves it, in both shapes."""

    def __init__(self):
        super().__init__()
        self.v1 = {"apps": []}
        self.v1_calls = 0

    async def list_apps(self, value):
        self.v1_calls += 1
        return self.v1


def listed(*aids: str, version2: bool = True) -> dict:
    extra = {"mesh": True, "media": False} if version2 else {}
    return {"apps": [dict(app_row(**extra), authorization_id=a, name=f"App {a}") for a in aids]}


@pytest.fixture()
def broker(tmp_path, monkeypatch) -> MeshBroker:
    monkeypatch.delenv("SLM_MESH_HOST", raising=False)
    monkeypatch.delenv("SLM_MESH_SHARED_SECRET", raising=False)
    db = str(tmp_path / "mesh.db")
    init_mesh_schema(db)
    return MeshBroker(db)


def make(tmp_path, broker, provider=None):
    provider = provider or SyncProvider()
    runtime, row, _ = completed_runtime(tmp_path, provider)
    runtime._app = SimpleNamespace(state=SimpleNamespace(mesh_broker=broker))
    return runtime, row, provider


def join(broker, cid: str, aid: str) -> str:
    ref = peer_ref(cid, aid)
    assert broker.ensure_web_peer(ref, app="client", display_name="x", connection_id=cid)["ok"]
    return ref


def alive(broker) -> set[str]:
    return {r["peer_id"] for r in rows(broker, "SELECT peer_id FROM mesh_peers")}


@pytest.mark.asyncio
async def test_a_revoked_app_is_retired_and_its_queued_mail_is_dropped(tmp_path, broker):
    runtime, row, provider = make(tmp_path, broker)
    kept, gone = join(broker, row.connection_id, "auth-1"), join(broker, row.connection_id, "auth-2")
    other = join(broker, "f" * 32, "auth-9")
    local = make_peer(broker, "sess-1")
    assert broker.send_message(local, gone, "queued for the revoked app")["ok"]
    provider.apps = listed("auth-1")
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert alive(broker) == {kept, other, local}
    assert rows(broker, "SELECT * FROM mesh_messages") == []
    assert peer_names.display_name(row.connection_id, "auth-1", kept) == "App auth-1"
    peer_names.set_names(row.connection_id, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    {"apps": [{"bogus": 1}]},                 # a row that is not a connected app
    {"apps": "nope"}, {}, {"apps": None},
])
async def test_an_answer_that_cannot_be_trusted_retires_nobody(tmp_path, broker, answer):
    runtime, row, provider = make(tmp_path, broker)
    ref = join(broker, row.connection_id, "auth-2")
    provider.apps = answer
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert ref in alive(broker)


@pytest.mark.asyncio
async def test_an_older_gateway_answering_in_the_first_shape_retires_nobody(tmp_path, broker):
    runtime, row, provider = make(tmp_path, broker)
    ref = join(broker, row.connection_id, "auth-2")
    provider.apps = listed("auth-1", version2=False)
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert ref in alive(broker)


@pytest.mark.asyncio
async def test_an_empty_list_retires_every_app_of_the_connection(tmp_path, broker):
    runtime, row, provider = make(tmp_path, broker)
    ref = join(broker, row.connection_id, "auth-1")
    provider.apps = {"apps": []}
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert ref not in alive(broker)


@pytest.mark.asyncio
async def test_a_peer_registered_while_the_list_was_in_flight_is_not_retired(tmp_path, broker):
    runtime, row, provider = make(tmp_path, broker)
    late = peer_ref(row.connection_id, "auth-new")

    class Racing(SyncProvider):
        async def list_apps_v2(self, value):
            answer = await super().list_apps_v2(value)     # the list is taken ...
            join(broker, row.connection_id, "auth-new")    # ... then a new app is first used
            return answer

    runtime.provider = Racing()
    runtime.provider.apps = {"apps": []}
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert late in alive(broker)


@pytest.mark.asyncio
async def test_viewing_connected_apps_syncs_the_peers_too(tmp_path, broker):
    runtime, row, provider = make(tmp_path, broker)
    kept, gone = join(broker, row.connection_id, "auth-1"), join(broker, row.connection_id, "auth-2")
    provider.v1 = listed("auth-1", version2=False)
    shown = await runtime.list_apps("owner", "profile", row.connection_id)
    assert [a["authorization_id"] for a in shown["apps"]] == ["auth-1"]
    assert alive(broker) == {kept}
    assert gone not in alive(broker)
    peer_names.set_names(row.connection_id, {})


@pytest.mark.asyncio
async def test_viewing_an_unreadable_list_does_not_retire_or_fail(tmp_path, broker):
    runtime, row, provider = make(tmp_path, broker)
    ref = join(broker, row.connection_id, "auth-2")
    provider.v1 = {"apps": [{"bogus": 1}]}
    shown = await runtime.list_apps("owner", "profile", row.connection_id)
    assert shown["apps"] == [] and ref in alive(broker)


@pytest.mark.asyncio
async def test_without_a_mesh_broker_the_names_still_refresh(tmp_path):
    runtime, row, provider = make(tmp_path, None)
    provider.apps = listed("auth-1")
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert peer_names.display_name(row.connection_id, "auth-1", "w_abcdef012345") == "App auth-1"
    peer_names.set_names(row.connection_id, {})


@pytest.mark.asyncio
async def test_a_running_connection_syncs_on_a_timer_and_stops_with_the_runtime(
        tmp_path, broker, monkeypatch):
    runtime, row, provider = make(tmp_path, broker)
    ref = join(broker, row.connection_id, "auth-2")
    provider.apps = listed("auth-1")
    monkeypatch.setattr(runtime_module, "PEER_SYNC_INTERVAL_S", 0.05)

    class Companion:
        _running = True

        def __init__(self, **kw):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(runtime_module, "Companion", Companion)
    await runtime.start(row)
    for _ in range(100):
        if ref not in alive(broker):
            break
        await asyncio.sleep(0.05)
    assert ref not in alive(broker)
    again = join(broker, row.connection_id, "auth-3")
    for _ in range(100):
        if again not in alive(broker):
            break
        await asyncio.sleep(0.05)
    assert again not in alive(broker)                  # the timer keeps running
    await runtime.stop()
    assert not runtime._peer_sync_tasks
    peer_names.set_names(row.connection_id, {})


@pytest.mark.asyncio
async def test_removing_the_connection_retires_all_of_its_apps(tmp_path, broker):
    runtime, row, _ = make(tmp_path, broker)
    mine, other = join(broker, row.connection_id, "auth-1"), join(broker, "f" * 32, "auth-9")
    await runtime._cancel("owner", "profile", row.connection_id)
    assert mine not in alive(broker) and other in alive(broker)


@pytest.mark.asyncio
async def test_a_failed_read_does_not_end_the_timer(tmp_path, broker, monkeypatch):
    runtime, row, provider = make(tmp_path, broker)
    ref = join(broker, row.connection_id, "auth-2")
    monkeypatch.setattr(runtime_module, "PEER_SYNC_INTERVAL_S", 0.05)
    failures = {"left": 2}
    good = listed("auth-1")

    async def flaky(value):
        if failures["left"]:
            failures["left"] -= 1
            raise ValueError("apps_unavailable")
        return good

    provider.list_apps_v2 = flaky
    task = asyncio.create_task(runtime._sync_peers_while_running(row))
    for _ in range(100):
        if ref not in alive(broker):
            break
        await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert ref not in alive(broker)
    peer_names.set_names(row.connection_id, {})


@pytest.mark.asyncio
async def test_removing_a_connection_whose_row_is_gone_still_retires_its_apps(tmp_path, broker):
    runtime, row, _ = make(tmp_path, broker)
    mine = join(broker, row.connection_id, "auth-1")
    runtime.store.by_connection = lambda *a, **k: None
    assert await runtime._cancel("owner", "profile", row.connection_id) is False
    assert mine not in alive(broker)


@pytest.mark.asyncio
async def test_a_gateway_saying_the_connection_is_gone_retires_all_its_apps(
        tmp_path, broker, monkeypatch):
    runtime, row, provider = make(tmp_path, broker)
    mine, other = join(broker, row.connection_id, "auth-1"), join(broker, "f" * 32, "auth-9")

    async def gone(value):
        raise ValueError("connection_unavailable")

    provider.list_apps_v2 = gone
    await asyncio.wait_for(runtime._sync_peers_while_running(row), 5)   # ends by itself
    assert mine not in alive(broker) and other in alive(broker)


def test_apps_list_403_maps_to_connection_unavailable():
    from superlocalmemory.remote_connections.gateway_provider import CloudGatewayProvider
    assert CloudGatewayProvider._status_error("/owner/apps", 403) == "connection_unavailable"
    assert CloudGatewayProvider._status_error("/owner/apps", 500) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ["refresh", "list", "rotate", "revoke"])
async def test_owner_actions_wait_for_the_connection_lock(tmp_path, broker, call):
    runtime, row, provider = make(tmp_path, broker)
    provider.v1 = listed("auth-1", version2=False)
    cid = row.connection_id
    work = {
        "refresh": lambda: runtime.refresh_peer_names("owner", "profile", cid),
        "list": lambda: runtime.list_apps("owner", "profile", cid),
        "rotate": lambda: runtime.rotate_grant_key("owner", "profile", cid),
        "revoke": lambda: runtime.revoke_app("owner", "profile", cid, "auth-1", 1),
    }[call]
    lock = runtime._locks.setdefault(cid, asyncio.Lock())
    async with lock:
        task = asyncio.create_task(work())
        await asyncio.sleep(0.05)
        assert not task.done()
    await asyncio.gather(task, return_exceptions=True)
    peer_names.set_names(cid, {})


# -- an app revoked anywhere loses its open upload links (audit F6) ----------------

@pytest.fixture()
def upload_store(tmp_path, monkeypatch):
    from superlocalmemory.media import upload_links

    store = upload_links.UploadLinks(tmp_path / "upload-data")
    monkeypatch.setattr(upload_links, "default_links", lambda: store)
    return store


@pytest.mark.asyncio
async def test_a_revoked_apps_open_upload_links_end_with_the_next_list(tmp_path, broker, upload_store):
    runtime, row, provider = make(tmp_path, broker)
    cid = row.connection_id
    gone = upload_store.mint(cid, "key1", "personal", "image", "", authorization_id="auth-2")
    kept = upload_store.mint(cid, "key1", "personal", "image", "", authorization_id="auth-1")
    provider.apps = listed("auth-1")
    await runtime.refresh_peer_names("owner", "profile", cid)
    assert upload_store.find(gone.token, cid).state == "failed"
    assert upload_store.find(kept.token, cid).state == "open"


@pytest.mark.asyncio
async def test_an_answer_that_cannot_be_trusted_ends_no_upload_link(tmp_path, broker, upload_store):
    runtime, row, provider = make(tmp_path, broker)
    link = upload_store.mint(row.connection_id, "key1", "personal", "image", "", authorization_id="auth-2")
    provider.apps = {"apps": [{"bogus": 1}]}
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert upload_store.find(link.token, row.connection_id).state == "open"
