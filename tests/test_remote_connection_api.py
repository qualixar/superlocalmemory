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
