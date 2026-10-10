"""Fetching, rotating and refreshing the grant key; the v2 connected-apps list."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import replace

import pytest

from superlocalmemory.remote_connections.gateway_provider import CloudGatewayProvider
from superlocalmemory.remote_connections.grant_keys import GrantKeyStore
from superlocalmemory.remote_connections.native_enrollment import NativeEnrollmentStore
from superlocalmemory.remote_connections import runtime as runtime_module
from tests.test_remote_connection_runtime import enrolled_runtime
from tests.test_remote_native_enrollment_store import Backend, record

CID = "a" * 32


def key_b64(seed=1) -> str:
    return base64.urlsafe_b64encode(bytes([seed]) * 32).rstrip(b"=").decode()


def provider_with(answer, calls):
    async def http(path, **kwargs):
        calls.append((path, kwargs))
        return answer

    return CloudGatewayProvider(None, redirect_uri="http://127.0.0.1:1/cb", http=http)


def row():
    return replace(record(), access_token="tok")


@pytest.mark.asyncio
async def test_grant_key_posts_an_empty_body_with_owner_proof():
    calls = []
    provider = provider_with({"version": 2, "key": key_b64(), "connection_id": CID}, calls)
    got = await provider.grant_key(row())
    assert got == {"version": 2, "key": key_b64()}
    path, kwargs = calls[0]
    assert path == "/owner/grant-key" and kwargs["json"] == {}
    assert kwargs["headers"]["Authorization"] == "Bearer tok" and kwargs["headers"]["DPoP"]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    {"version": 0, "key": key_b64(), "connection_id": CID},
    {"version": True, "key": key_b64(), "connection_id": CID},
    {"version": 2 ** 53, "key": key_b64(), "connection_id": CID},
    {"version": 1, "key": "short", "connection_id": CID},
    {"version": 1, "key": "!" * 43, "connection_id": CID},
    {"version": 1, "key": key_b64(), "connection_id": "b" * 32},
    {"version": 1, "key": key_b64()},
])
async def test_grant_key_answers_are_validated(answer):
    with pytest.raises(ValueError, match="invalid_grant_key"):
        await provider_with(answer, []).grant_key(row())


def test_grant_key_status_codes_map_to_fixed_errors():
    mapper = CloudGatewayProvider._grant_key_error
    assert mapper("/owner/grant-key", 403) == "connection_unavailable"
    assert mapper("/owner/grant-key", 503) == "grant_unavailable"
    assert mapper("/owner/grant-key", 500) is None
    assert mapper("/owner/renew", 503) is None


@pytest.mark.asyncio
async def test_transport_maps_gateway_refusals_and_allows_the_path(monkeypatch):
    import httpx

    class Response:
        def __init__(self, status):
            self.status_code, self.is_success = status, status < 300

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class Client:
        status = 503

        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            Client.url = url
            return Response(Client.status)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    with pytest.raises(ValueError, match="grant_unavailable"):
        await CloudGatewayProvider._request("/owner/grant-key", json={})
    assert Client.url.endswith("/owner/grant-key")
    Client.status = 403
    with pytest.raises(ValueError, match="connection_unavailable"):
        await CloudGatewayProvider._request("/owner/grant-key", json={})


@pytest.mark.asyncio
async def test_list_apps_v2_asks_for_version_two_and_v1_does_not():
    calls = []
    provider = provider_with({"apps": []}, calls)
    await provider.list_apps_v2(row())
    await provider.list_apps(row())
    assert calls[0][0] == calls[1][0] == "/owner/apps"
    assert calls[0][1]["json"] == {"version": 2}
    assert "json" not in calls[1][1]


def app_row(**permissions):
    perms = {"read": True, "save": False, "session": False, **permissions}
    return {"authorization_id": "auth-1", "name": "ChatGPT", "client_host": None,
            "permissions": perms, "version": 1, "connected_at_ms": None,
            "last_used_at_ms": None}


def test_v2_rows_accept_mesh_and_media_and_v1_stays_exact():
    v2 = app_row(mesh=False, media=True)
    assert runtime_module._connected_app_v2(v2)
    assert not runtime_module._connected_app(v2)
    assert not runtime_module._connected_app_v2(app_row())              # v1 shape
    assert not runtime_module._connected_app_v2(app_row(mesh=1, media=True))
    assert runtime_module._connected_app(app_row())


def completed_runtime(tmp_path, provider, now=1000.0):
    runtime, base = enrolled_runtime(tmp_path)
    base = replace(base, completed=True)
    runtime.store.save(base)
    backend = Backend()
    runtime.grant_keys = GrantKeyStore(lambda: backend, clock=lambda: now)
    runtime.provider = provider
    runtime._grant_now = lambda: now
    runtime.journal  # noqa: B018 - journal row exists via enrolled_runtime
    return runtime, base, backend


class FakeProvider:
    def __init__(self):
        self.versions = iter(range(1, 50))
        self.calls = 0
        self.fail = None
        self.apps = {"apps": []}

    async def exchange(self, value, code):
        return replace(value, access_token="synthetic")

    async def grant_key(self, value):
        self.calls += 1
        if self.fail:
            raise ValueError(self.fail)
        version = next(self.versions)
        return {"version": version, "key": key_b64(version)}

    async def list_apps_v2(self, value):
        return self.apps


@pytest.mark.asyncio
async def test_ensure_grant_key_fetches_once_and_stores_it(tmp_path):
    provider = FakeProvider()
    runtime, row, _ = completed_runtime(tmp_path, provider)
    await runtime.ensure_grant_key(row)
    assert runtime.grant_keys.load(row.connection_id).current == (1, bytes([1]) * 32)
    await runtime.ensure_grant_key(row)
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_ensure_failure_is_silent(tmp_path):
    provider = FakeProvider()
    provider.fail = "grant_unavailable"
    runtime, row, _ = completed_runtime(tmp_path, provider)
    await runtime.ensure_grant_key(row)           # must not raise
    assert runtime.grant_keys.load(row.connection_id).current is None


@pytest.mark.asyncio
async def test_rotate_always_mints_a_new_version_and_keeps_the_old_for_a_while(tmp_path):
    provider = FakeProvider()
    runtime, row, _ = completed_runtime(tmp_path, provider)
    await runtime.ensure_grant_key(row)
    await runtime.rotate_grant_key("owner", "profile", row.connection_id)
    keys = runtime.grant_keys.load(row.connection_id)
    assert keys.current[0] == 2 and keys.previous[0] == 1


@pytest.mark.asyncio
async def test_rotate_for_a_connection_the_owner_does_not_hold_is_not_found(tmp_path):
    runtime, row, _ = completed_runtime(tmp_path, FakeProvider())
    with pytest.raises(ValueError, match="not_found"):
        await runtime.rotate_grant_key("someone-else", "profile", row.connection_id)


@pytest.mark.asyncio
async def test_refresh_requests_are_limited_to_one_per_ten_minutes(tmp_path):
    provider = FakeProvider()
    runtime, row, _ = completed_runtime(tmp_path, provider, now=1000.0)
    clock = {"now": 1000.0}
    runtime._grant_now = lambda: clock["now"]
    assert runtime.request_grant_refresh(row.connection_id) is True
    assert runtime.request_grant_refresh(row.connection_id) is False
    await asyncio.sleep(0.05)
    await asyncio.gather(*runtime._grant_tasks.values(), return_exceptions=True)
    assert provider.calls == 1
    clock["now"] += 599
    assert runtime.request_grant_refresh(row.connection_id) is False
    clock["now"] += 2
    assert runtime.request_grant_refresh(row.connection_id) is True
    await asyncio.gather(*runtime._grant_tasks.values(), return_exceptions=True)
    assert provider.calls == 2
    assert runtime.request_grant_refresh("b" * 32) is False  # unknown connection: nothing to do
    assert set(runtime._grant_asked) == {row.connection_id}


@pytest.mark.asyncio
async def test_refresh_fetches_a_key_even_when_one_is_held(tmp_path):
    provider = FakeProvider()
    runtime, row, _ = completed_runtime(tmp_path, provider)
    await runtime.ensure_grant_key(row)
    runtime.request_grant_refresh(row.connection_id)
    await asyncio.gather(*runtime._grant_tasks.values(), return_exceptions=True)
    assert runtime.grant_keys.load(row.connection_id).current[0] == 2


@pytest.mark.asyncio
async def test_start_fetches_a_missing_key_in_the_background(tmp_path):
    provider = FakeProvider()
    runtime, row, _ = completed_runtime(tmp_path, provider)

    class Companion:
        _running = True

        def __init__(self, **kw):
            pass

        async def start(self):
            pass

    runtime_module.Companion = Companion
    try:
        await runtime.start(row)
        await asyncio.gather(*runtime._grant_tasks.values(), return_exceptions=True)
    finally:
        from superlocalmemory.remote_connections.companion import Companion as Real

        runtime_module.Companion = Real
        for task in list(runtime._renewal_tasks.values()):
            task.cancel()
    assert runtime.grant_keys.load(row.connection_id).current is not None


@pytest.mark.asyncio
async def test_cancel_forgets_the_grant_key(tmp_path):
    provider = FakeProvider()
    runtime, row, _ = completed_runtime(tmp_path, provider)
    await runtime.ensure_grant_key(row)
    assert runtime.grant_keys.load(row.connection_id).current is not None
    await runtime._cancel("owner", "profile", row.connection_id)
    assert runtime.grant_keys.load(row.connection_id).current is None


@pytest.mark.asyncio
async def test_peer_names_come_from_the_v2_list_and_unlisted_apps_vanish(tmp_path):
    from superlocalmemory.remote_connections import peer_names

    provider = FakeProvider()
    provider.apps = {"apps": [app_row(mesh=True, media=False), {"bogus": 1}]}
    runtime, row, _ = completed_runtime(tmp_path, provider)
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert peer_names.display_name(row.connection_id, "auth-1", "w_abcdef012345") == "ChatGPT"
    provider.apps = {"apps": []}
    await runtime.refresh_peer_names("owner", "profile", row.connection_id)
    assert peer_names.display_name(row.connection_id, "auth-1", "w_abcdef012345") == "Web app abcdef"


@pytest.mark.asyncio
async def test_origin_is_wired_to_the_grant_keys_and_refresh(tmp_path):
    runtime, row, _ = completed_runtime(tmp_path, FakeProvider())
    assert runtime.origin._grant_keys is not None
    assert runtime.origin._on_unknown_kid == runtime.request_grant_refresh


@pytest.mark.asyncio
async def test_key_fetches_for_one_connection_never_exchange_the_token_twice_at_once(tmp_path):
    running = {"now": 0, "most": 0}

    class Counting(FakeProvider):
        async def exchange(self, value, code):
            running["now"] += 1
            running["most"] = max(running["most"], running["now"])
            await asyncio.sleep(0.02)
            running["now"] -= 1
            return await super().exchange(value, code)

    runtime, row, _ = completed_runtime(tmp_path, Counting())
    await asyncio.gather(*(runtime.ensure_grant_key(row, force=True) for _ in range(4)))
    assert running["most"] == 1


@pytest.mark.asyncio
async def test_a_key_fetch_waits_for_the_connection_lock_other_exchanges_hold(tmp_path):
    runtime, row, _ = completed_runtime(tmp_path, FakeProvider())
    lock = runtime._locks.setdefault(row.connection_id, asyncio.Lock())
    async with lock:
        task = asyncio.create_task(runtime.ensure_grant_key(row))
        await asyncio.sleep(0.05)
        assert not task.done()
    await task
    assert runtime.grant_keys.load(row.connection_id).current is not None


# -- upload links die with the consent, the grant key or the connection --------

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100
NONCE = "n" * 22


@pytest.fixture()
def links(tmp_path, monkeypatch):
    from superlocalmemory.media import upload_links

    store = upload_links.UploadLinks(tmp_path / "data")
    monkeypatch.setattr(upload_links, "default_links", lambda: store)
    return store


def started_link(links, cid):
    link = links.mint(cid, "key1", "personal", "image", "")
    links.accept_chunk(link.token, cid, 0, len(PNG) + 5, PNG, NONCE)
    return link


@pytest.mark.asyncio
async def test_rotating_the_grant_key_ends_every_open_link_of_that_connection(tmp_path, links):
    runtime, row, _ = completed_runtime(tmp_path, FakeProvider())
    mine, other = started_link(links, row.connection_id), links.mint("b" * 32, "key1", "personal", "image", "")
    await runtime.rotate_grant_key("owner", "profile", row.connection_id)
    assert links.find(mine.token, row.connection_id).state == "failed"
    assert links.find(other.token, "b" * 32).state == "open"


@pytest.mark.asyncio
async def test_revoking_an_app_ends_the_connections_open_links(tmp_path, links):
    provider = FakeProvider()

    async def revoke_app(latest, authorization_id, expected_version):
        return {"revoked": True, "version": expected_version + 1}

    provider.revoke_app = revoke_app
    runtime, row, _ = completed_runtime(tmp_path, provider)
    mine = started_link(links, row.connection_id)
    await runtime.revoke_app("owner", "profile", row.connection_id, "app-1", 1)
    assert links.find(mine.token, row.connection_id).state == "failed"


@pytest.mark.asyncio
async def test_a_failed_revoke_leaves_links_alone(tmp_path, links):
    provider = FakeProvider()

    async def revoke_app(latest, authorization_id, expected_version):
        return {"revoked": False}

    provider.revoke_app = revoke_app
    runtime, row, _ = completed_runtime(tmp_path, provider)
    mine = started_link(links, row.connection_id)
    with pytest.raises(ValueError):
        await runtime.revoke_app("owner", "profile", row.connection_id, "app-1", 1)
    assert links.find(mine.token, row.connection_id).state == "receiving"


@pytest.mark.asyncio
async def test_cancelling_the_connection_ends_its_open_links(tmp_path, links):
    class Provider(FakeProvider):
        async def revoke(self, value):
            return {"revoked": True}

        async def cancel_bootstrap(self, value):
            return {"cancelled": True}

    runtime, row, _ = completed_runtime(tmp_path, Provider())
    mine = started_link(links, row.connection_id)
    await runtime.cancel(row.owner, row.profile, row.connection_id)
    assert links.find(mine.token, row.connection_id).state == "failed"


@pytest.mark.asyncio
async def test_ending_links_never_fails_the_revocation(tmp_path, monkeypatch):
    from superlocalmemory.media import upload_links

    def broken():
        raise RuntimeError("disk full")

    monkeypatch.setattr(upload_links, "default_links", broken)
    runtime, row, _ = completed_runtime(tmp_path, FakeProvider())
    assert await runtime.rotate_grant_key("owner", "profile", row.connection_id) == 1
