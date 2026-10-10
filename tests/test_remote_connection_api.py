"""Actual HTTP route checks; cloud provider is a synthetic fixture."""
import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

try:
    from superlocalmemory.remote_connections.service import RemoteConnectionService, GatewayReceipt
    from superlocalmemory.server.routes.connections import router
except ImportError:
    RemoteConnectionService = None
    GatewayReceipt = None
    router = None

from superlocalmemory.remote_connections.journal import EnrollmentJournal


class Provider:
    calls = 0
    async def enroll(self, *, installation_id, connection_id, owner, profile, intent):
        self.calls += 1
        return GatewayReceipt(connection_id)


def payload():
    return {"host": "muse", "profile_id": "default", "remote_opt_in": True,
            "permissions": {"read": True, "write": False, "correction": False, "session": False}}


@pytest.fixture
def configured(tmp_path, monkeypatch):
    assert router is not None and RemoteConnectionService is not None
    from superlocalmemory.core import security_primitives
    monkeypatch.setattr(security_primitives, "verify_install_token", lambda token: token == "synthetic-local-token")
    app = FastAPI(); app.include_router(router)
    app.state.profile_runtime = SimpleNamespace(snapshot=SimpleNamespace(profile_id="default"))
    app.state.daemon_descriptor = SimpleNamespace(port=9999)
    provider = Provider()
    app.state.remote_connections = RemoteConnectionService(EnrollmentJournal(tmp_path / "remote"), provider, hosts=("muse",))
    with TestClient(app, base_url="http://127.0.0.1:9999", client=("127.0.0.1", 5000)) as client:
        yield client, provider, app


def headers(**extra):
    return {"X-Install-Token": "synthetic-local-token", "Idempotency-Key": "a" * 32, **extra}


def test_disabled_endpoint_does_not_create_remote_state(tmp_path):
    assert router is not None
    app = FastAPI(); app.include_router(router)
    app.state.profile_runtime = SimpleNamespace(snapshot=SimpleNamespace(profile_id="default"))
    with TestClient(app, base_url="http://127.0.0.1:9999", client=("127.0.0.1", 5000)) as client:
        response = client.get("/api/v3/connections/status")
        assert response.status_code == 200
        assert response.json()["available"] is False
        assert response.json()["connections"] == []
    assert not list(tmp_path.iterdir())


def test_real_route_acks_pending_and_retry_does_not_repeat_provider(configured):
    client, provider, app = configured
    first = client.post("/api/v3/connections/initiate", headers=headers(), json=payload())
    assert first.status_code == 200 and first.json()["state"] == "pending"
    second = client.post("/api/v3/connections/initiate", headers=headers(), json=payload())
    assert second.json()["connection_id"] == first.json()["connection_id"]
    assert provider.calls == 1
    status = client.get("/api/v3/connections/status").json()
    assert status["hosts"] == ["muse"] and status["installation_id"]
    assert status["connections"][0]["verified"] is False
    assert status["connections"][0]["intent_key"] == "a" * 32


def test_mutation_requires_explicit_install_credential(configured):
    client, provider, _ = configured
    assert client.post("/api/v3/connections/initiate", headers={"Idempotency-Key": "a" * 32}, json=payload()).status_code == 403
    assert provider.calls == 0


@pytest.mark.parametrize("origin", ["https://evil.example", "http://127.0.0.1:4444", "null"])
def test_cross_origin_denied_even_with_install_token(configured, origin):
    client, provider, _ = configured
    assert client.post("/api/v3/connections/initiate", headers=headers(Origin=origin), json=payload()).status_code == 403
    assert provider.calls == 0


def test_same_origin_allowed(configured):
    client, _, _ = configured
    assert client.post("/api/v3/connections/initiate", headers=headers(Origin="http://127.0.0.1:9999"), json=payload()).status_code == 200


def test_network_peer_and_dns_rebinding_host_denied(configured):
    _, _, app = configured
    with TestClient(app, base_url="http://127.0.0.1:9999", client=("192.0.2.20", 5000)) as client:
        assert client.get("/api/v3/connections/status").status_code == 403
    with TestClient(app, base_url="http://evil.example:9999", client=("127.0.0.1", 5000)) as client:
        assert client.get("/api/v3/connections/status").status_code == 403


def test_selected_profile_and_host_cannot_be_forged(configured):
    client, provider, _ = configured
    data = payload(); data["profile_id"] = "other"
    assert client.post("/api/v3/connections/initiate", headers=headers(), json=data).status_code == 409
    data = payload(); data["host"] = "chatgpt"
    assert client.post("/api/v3/connections/initiate", headers=headers(), json=data).status_code == 403
    assert provider.calls == 0


@pytest.mark.parametrize("change", [
    lambda x: x.update(remote_opt_in=False),
    lambda x: x.update(remote_opt_in="true"),
    lambda x: x.update(unknown="data"),
    lambda x: x["permissions"].update(write="true"),
])
def test_strict_consent_schema(configured, change):
    client, provider, _ = configured
    data = payload(); change(data)
    assert client.post("/api/v3/connections/initiate", headers=headers(), json=data).status_code == 422
    assert provider.calls == 0


def test_intent_conflict_and_no_sensitive_metadata(configured):
    client, _, _ = configured
    client.post("/api/v3/connections/initiate", headers=headers(), json=payload())
    changed = payload(); changed["permissions"]["write"] = True
    assert client.post("/api/v3/connections/initiate", headers=headers(), json=changed).status_code == 409
    response = client.get("/api/v3/connections/status")
    for field in ["synthetic-local-token", "lease_token", "remote_reference", "authorization_url", "deviceToken"]:
        assert field not in response.text


def test_provider_errors_are_sanitized_and_leave_pending_intent(configured):
    client, provider, app = configured
    async def fail(**kwargs):
        raise RuntimeError("SECRET provider failure")
    provider.enroll = fail
    result = client.post("/api/v3/connections/initiate", headers=headers(), json=payload())
    assert result.status_code == 503 and "SECRET" not in result.text
    assert client.get("/api/v3/connections/status").json()["connections"][0]["state"] == "pending"


def test_sign_in_receipt_is_bound_and_survives_service_restart(configured):
    client, provider, app = configured
    async def enroll(**args):
        provider.calls += 1
        return GatewayReceipt(args["connection_id"], "https://auth.superlocalmemory.com/owner-login?connection_id=" + args["connection_id"])
    provider.enroll = enroll
    first = client.post("/api/v3/connections/initiate", headers=headers(), json=payload())
    assert first.status_code == 200 and first.json()["authorization_url"]
    service = app.state.remote_connections
    app.state.remote_connections = RemoteConnectionService(EnrollmentJournal(service.journal.path.parent), provider, hosts=("muse",))
    second = client.post("/api/v3/connections/initiate", headers=headers(), json=payload())
    assert second.json() == first.json() and provider.calls == 1


@pytest.mark.parametrize("url", ["https://evil.example/owner-login?connection_id=x", "https://auth.superlocalmemory.com/owner-login?connection_id=foreign", "https://auth.superlocalmemory.com/owner-login?connection_id=x&token=SECRET"])
def test_untrusted_provider_receipt_is_not_exposed(configured, url):
    client, provider, _ = configured
    async def enroll(**args):
        return GatewayReceipt(args["connection_id"], url)
    provider.enroll = enroll
    result = client.post("/api/v3/connections/initiate", headers=headers(), json=payload())
    assert result.status_code >= 400 and "SECRET" not in result.text and url not in result.text


def test_cancelled_intent_and_late_provider_receipt_never_ack_connected(configured):
    client, provider, app = configured
    service = app.state.remote_connections
    async def enroll(**args):
        row = service.journal.get(args["owner"], args["profile"], args["connection_id"])
        service.journal.cancel(args["owner"], args["profile"], row.connection_id, row.version)
        return GatewayReceipt(row.connection_id)
    provider.enroll = enroll
    first = client.post("/api/v3/connections/initiate", headers=headers(), json=payload())
    assert first.status_code == 409
    assert client.post("/api/v3/connections/initiate", headers=headers(), json=payload()).status_code == 409
    status = client.get("/api/v3/connections/status").json()["connections"][0]
    assert status["state"] == "cancelled" and status["verified"] is False and status["cleanup_pending"] is True


def test_sec_fetch_site_and_missing_retry_key_are_denied(configured):
    client, provider, _ = configured
    assert client.post("/api/v3/connections/initiate", headers=headers(**{"Sec-Fetch-Site": "cross-site"}), json=payload()).status_code == 403
    assert client.post("/api/v3/connections/initiate", headers={"X-Install-Token": "synthetic-local-token"}, json=payload()).status_code == 400
    assert provider.calls == 0


def test_no_service_and_broken_service_do_not_enable_remote(configured):
    client, _, app = configured
    app.state.remote_connections = None
    assert client.post("/api/v3/connections/initiate", headers=headers(), json=payload()).status_code == 503
    app.state.remote_connections = object()
    assert client.get("/api/v3/connections/status").status_code == 503


def test_actual_daemon_advertises_dormant_web_connection_feature(caplog, tmp_path, monkeypatch):
    # Its own data folder: the real app must not depend on (or touch) whatever folder an
    # earlier test left the process pointing at. It failed once in the full suite that way.
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
    from superlocalmemory.server.unified_daemon import create_app
    with caplog.at_level("WARNING"):
        app = create_app()
    # If the optional add-on failed to install, say why (the daemon logs it with a traceback).
    assert "remote_connections_router unavailable" not in caplog.text, caplog.text[-3000:]
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 5000))
    # No lifespan start: this tests the real app/middleware registration without
    # launching background providers, mesh or memory engine workers.
    result = client.get("/api/v3/connections/status")
    assert result.status_code == 200 and result.json()["available"] is True
    assert result.json()["connections"] == []
    assert app.state.remote_connection_runtime._companions == {}


def test_optional_router_import_failure_preserves_local_daemon(monkeypatch, caplog):
    import builtins
    from superlocalmemory.server.unified_daemon import create_app
    original_import = builtins.__import__

    def import_without_addon(name, *args, **kwargs):
        if name == "superlocalmemory.server.routes.connections":
            raise ImportError("synthetic missing remote add-on")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_addon)
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/v3/connections/status" not in paths
    assert "/health" in paths
    assert "remote_connections_router unavailable; local services remain enabled" in caplog.text


def test_cancel_route_is_authenticated_versioned_and_idempotent(configured):
    client, provider, _ = configured
    identifier = client.post("/api/v3/connections/initiate", headers=headers(), json=payload()).json()["connection_id"]
    row = client.get("/api/v3/connections/status").json()["connections"][0]
    path = f"/api/v3/connections/{identifier}/cancel"
    body = {"profile_id": "default", "expected_version": row["version"]}
    assert client.post(path, json=body).status_code == 403
    stale = dict(body, expected_version=row["version"] - 1)
    assert client.post(path, headers=headers(), json=stale).status_code == 409
    first = client.post(path, headers=headers(), json=body)
    assert first.status_code == 200
    assert first.json()["state"] == "cancelled"
    assert first.json()["cleanup_pending"] is True
    assert first.json()["verified"] is False
    assert client.post(path, headers=headers(), json=body).json() == first.json()
    assert provider.calls == 1
    assert client.post("/api/v3/connections/initiate", headers=headers(), json=payload()).status_code == 409


@pytest.mark.parametrize("version", [True, "2", -1])
def test_cancel_schema_rejects_coerced_or_negative_version(configured, version):
    client, _, _ = configured
    result = client.post("/api/v3/connections/" + "b" * 32 + "/cancel", headers=headers(),
                         json={"profile_id": "default", "expected_version": version})
    assert result.status_code == 422


def test_cancel_cannot_cross_profile_or_discover_foreign_connections(configured):
    client, _, _ = configured
    path = "/api/v3/connections/" + "b" * 32 + "/cancel"
    assert client.post(path, headers=headers(), json={"profile_id": "other", "expected_version": 0}).status_code == 409
    result = client.post(path, headers=headers(), json={"profile_id": "default", "expected_version": 0})
    assert result.status_code == 404 and result.json()["detail"] == "not_found"


def test_generic_mcp_client_still_requires_allowed_catalog_and_opt_in(configured):
    client, provider, app = configured
    data = payload(); data['host'] = 'other_mcp'
    assert client.post('/api/v3/connections/initiate', headers=headers(), json=data).status_code == 403
    app.state.remote_connections.hosts = ('muse', 'other_mcp')
    data['remote_opt_in'] = False
    assert client.post('/api/v3/connections/initiate', headers=headers(), json=data).status_code == 422
    data['remote_opt_in'] = True
    response = client.post('/api/v3/connections/initiate', headers=headers(), json=data)
    assert response.status_code == 200 and response.json()['state'] == 'pending'
    assert provider.calls == 1


def test_restart_requires_explicit_opt_in_local_token_and_current_profile(configured):
    client, provider, app = configured
    calls = []
    async def restart(owner, profile, identifier, version):
        calls.append((owner, profile, identifier, version))
        return {'state': 'pending', 'connection_id': 'b' * 32}
    app.state.remote_connections.restart = restart
    path = '/api/v3/connections/' + 'a' * 32 + '/restart'
    data = {'profile_id': 'default', 'expected_version': 3, 'remote_opt_in': True}
    assert client.post(path, json=data).status_code == 403
    assert client.post(path, headers=headers(), json={**data, 'remote_opt_in': False}).status_code == 422
    assert client.post(path, headers=headers(), json={**data, 'profile_id': 'other'}).status_code == 409
    assert client.post(path, headers=headers(), json=data).status_code == 200
    assert len(calls) == 1


def test_successful_callback_redirects_to_dashboard_without_replayable_query(configured):
    client, _, app = configured
    async def callback(state, code): return 'a' * 32
    app.state.remote_connection_runtime = SimpleNamespace(callback=callback)
    response = client.get('/api/v3/connections/callback?state=synthetic&code=synthetic', follow_redirects=False)
    assert response.status_code == 303
    assert response.headers['location'] == '/#apps-pane'
    assert 'synthetic' not in response.headers['location']


class AppsRuntime:
    """Synthetic runtime for the Connected apps routes; no cloud calls."""
    def __init__(self, error=None):
        self.error, self.removed = error, []
    async def resume(self, owner, profile):
        return None
    async def list_apps(self, owner, profile, connection_id):
        if self.error:
            raise ValueError(self.error)
        return {"connection_id": connection_id, "apps": [{"authorization_id": "app-1", "name": "Composio", "client_host": "backend.composio.dev", "permissions": {"read": True, "save": True, "session": False}, "version": 1, "connected_at_ms": 1, "last_used_at_ms": None}]}
    async def revoke_app(self, owner, profile, connection_id, authorization_id, expected_version):
        if self.error:
            raise ValueError(self.error)
        self.removed.append((owner, profile, connection_id, authorization_id, expected_version))
        return {"revoked": True}


def test_connected_apps_list_is_served_to_the_local_dashboard(configured):
    client, _, app = configured
    app.state.remote_connection_runtime = AppsRuntime()
    response = client.get("/api/v3/connections/" + "c" * 32 + "/apps")
    assert response.status_code == 200
    assert response.json()["apps"][0]["name"] == "Composio"


def test_removing_an_app_needs_the_install_credential_and_current_profile(configured):
    client, _, app = configured
    runtime = AppsRuntime(); app.state.remote_connection_runtime = runtime
    url = "/api/v3/connections/" + "c" * 32 + "/apps/app-1/revoke"
    assert client.post(url, json={"profile_id": "default", "expected_version": 1}).status_code == 403
    assert client.post(url, headers=headers(), json={"profile_id": "work", "expected_version": 1}).status_code == 409
    assert client.post(url, headers=headers(), json={"profile_id": "default", "expected_version": 1, "x": 1}).status_code == 422
    ok = client.post(url, headers=headers(), json={"profile_id": "default", "expected_version": 1})
    assert ok.status_code == 200 and ok.json() == {"revoked": True}
    assert runtime.removed == [(runtime.removed[0][0], "default", "c" * 32, "app-1", 1)]


@pytest.mark.parametrize("code,status", [("not_found", 404), ("version_conflict", 409), ("SECRET gateway text", 503)])
def test_connected_apps_errors_are_mapped_and_sanitized(configured, code, status):
    client, _, app = configured
    app.state.remote_connection_runtime = AppsRuntime(code)
    listed = client.get("/api/v3/connections/" + "c" * 32 + "/apps")
    removed = client.post("/api/v3/connections/" + "c" * 32 + "/apps/app-1/revoke", headers=headers(), json={"profile_id": "default", "expected_version": 1})
    assert listed.status_code == status and removed.status_code == status
    assert "SECRET" not in listed.text and "SECRET" not in removed.text


def test_connected_apps_without_runtime_is_unavailable(configured):
    client, _, app = configured
    app.state.remote_connection_runtime = None
    assert client.get("/api/v3/connections/" + "c" * 32 + "/apps").status_code == 503


class AbilitiesRuntime(AppsRuntime):
    """Synthetic runtime for the second yes: the connection key's mesh and media opt-ins."""
    def __init__(self, error=None):
        super().__init__(error)
        self.extras: set[str] = set()
        self.changes: list = []
    async def key_abilities(self, owner, profile, connection_id):
        if self.error:
            raise ValueError(self.error)
        return {"connection_id": connection_id, "mesh": "mesh" in self.extras, "media": "media" in self.extras}
    async def set_key_ability(self, owner, profile, connection_id, extra, allow):
        if self.error:
            raise ValueError(self.error)
        (self.extras.add if allow else self.extras.discard)(extra)
        self.changes.append((profile, connection_id, extra, allow))
        return await self.key_abilities(owner, profile, connection_id)


def test_the_dashboard_reads_and_sets_the_second_yes(configured):
    """Package D: the mesh/media opt-in is a dashboard click, not only a terminal command."""
    client, _, app = configured
    runtime = AbilitiesRuntime(); app.state.remote_connection_runtime = runtime
    url = "/api/v3/connections/" + "c" * 32 + "/abilities"
    got = client.get(url)
    assert got.status_code == 200 and got.json() == {"connection_id": "c" * 32, "mesh": False, "media": False}
    body = {"profile_id": "default", "ability": "mesh", "allow": True}
    assert client.post(url, json=body).status_code == 403                        # needs the install credential
    assert client.post(url, headers=headers(), json={**body, "profile_id": "work"}).status_code == 409
    assert client.post(url, headers=headers(), json={**body, "ability": "admin"}).status_code == 422
    assert client.post(url, headers=headers(), json={**body, "allow": "yes"}).status_code == 422
    ok = client.post(url, headers=headers(), json=body)
    assert ok.status_code == 200 and ok.json()["mesh"] is True and ok.json()["media"] is False
    off = client.post(url, headers=headers(), json={**body, "allow": False})
    assert off.json()["mesh"] is False
    assert runtime.changes == [("default", "c" * 32, "mesh", True), ("default", "c" * 32, "mesh", False)]


def test_the_second_yes_is_refused_from_another_site(configured):
    client, _, app = configured
    app.state.remote_connection_runtime = AbilitiesRuntime()
    url = "/api/v3/connections/" + "c" * 32 + "/abilities"
    body = {"profile_id": "default", "ability": "media", "allow": True}
    assert client.post(url, headers=headers(Origin="https://evil.example"), json=body).status_code == 403
    assert client.post(url, headers=headers(**{"Sec-Fetch-Site": "cross-site"}), json=body).status_code == 403


@pytest.mark.parametrize("code,status", [("not_found", 404), ("SECRET gateway text", 503)])
def test_second_yes_errors_are_mapped_and_sanitized(configured, code, status):
    client, _, app = configured
    app.state.remote_connection_runtime = AbilitiesRuntime(code)
    url = "/api/v3/connections/" + "c" * 32 + "/abilities"
    got = client.get(url)
    put = client.post(url, headers=headers(), json={"profile_id": "default", "ability": "mesh", "allow": True})
    assert got.status_code == status and put.status_code == status
    assert "SECRET" not in got.text and "SECRET" not in put.text
