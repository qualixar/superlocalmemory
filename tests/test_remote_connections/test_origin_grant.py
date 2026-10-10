"""The origin verifies the grant, strips it, and exposes only verified claims."""
from __future__ import annotations

import base64
import time

import pytest

from superlocalmemory.mcp.remote_caller import current_remote_grant, remote_grant
from superlocalmemory.remote_connections.credentials import ConnectorCredential
from superlocalmemory.remote_connections.grant import GrantKeys, sign_grant
from superlocalmemory.remote_connections.origin import CanonicalMcpOrigin

CID = "a" * 32
KEY = bytes(range(1, 33))
NOW = 1_700_000_000.0


def credential(cid=CID):
    return ConnectorCredential("install-a", "owner-a", "default", cid, 1,
                               int((NOW + 600) * 1000), "a" * 64, "slmr_" + "b" * 43)


def frame(grant=None, *, fid="wire_1", generation=1, deadline=None, extra_headers=()):
    headers = [["content-type", "application/json"], *map(list, extra_headers)]
    if grant is not None:
        headers.append(["x-slm-grant", grant])
    return {"v": 1, "kind": "request", "id": fid, "generation": generation,
            "deadlineAt": deadline if deadline is not None else int(NOW * 1000) + 20_000,
            "headers": headers,
            "bodyBase64": base64.b64encode(b'{"jsonrpc":"2.0","id":1,"method":"ping"}').decode()}


def signed(**changes):
    args = dict(connection_id=CID, authorization_id="auth-1", authorization_version=2,
                app="client-1", scopes=("slm:read", "slm:mesh"), folders_visible=False,
                frame_id="wire_1", generation=1, deadline_at_ms=int(NOW * 1000) + 20_000,
                key_version=1)
    args.update(changes)
    return sign_grant(KEY, **args)


class Recorder:
    def __init__(self):
        self.seen = []

    async def __call__(self, scope, receive, send):
        await receive()
        self.seen.append((dict(scope["headers"]), current_remote_grant()))
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})


def origin(app, keys=None, refresh=None):
    return CanonicalMcpOrigin(
        app, clock=lambda: NOW,
        grant_keys=(lambda cid: keys) if keys is not None else None,
        on_unknown_kid=refresh)


KEYS = GrantKeys((1, KEY), None)


@pytest.mark.asyncio
async def test_verified_grant_is_visible_inside_the_app_and_header_is_gone():
    app = Recorder()
    await origin(app, KEYS)(frame(signed()), credential())
    headers, grant = app.seen[0]
    assert b"x-slm-grant" not in headers
    assert grant.authorization_id == "auth-1" and grant.app == "client-1"
    assert grant.scopes == {"slm:read", "slm:mesh"} and grant.connection_id == CID
    assert current_remote_grant() is None  # nothing leaks out of the request


@pytest.mark.asyncio
async def test_header_is_stripped_in_any_letter_case_even_without_keys():
    app = Recorder()
    value = signed()
    packet = frame()
    packet["headers"].append(["X-SLM-Grant", value])
    await origin(app)(packet, credential())
    headers, grant = app.seen[0]
    assert not any(name.lower() == b"x-slm-grant" for name in headers)
    assert grant is None


@pytest.mark.asyncio
async def test_no_header_means_no_grant():
    app = Recorder()
    await origin(app, KEYS)(frame(), credential())
    assert app.seen[0][1] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("changes,packet", [
    ({"frame_id": "other"}, {}),                        # signed for another frame
    ({"generation": 2}, {}),                            # another generation
    ({"deadline_at_ms": int(NOW * 1000) + 1}, {}),      # another deadline
    ({"connection_id": "f" * 32}, {}),                  # another connection
    ({"deadline_at_ms": int(NOW * 1000) - 6000},
     {"deadline": int(NOW * 1000) - 6000}),             # expired
])
async def test_forged_or_stale_grant_is_ignored_but_the_call_still_runs(changes, packet):
    app = Recorder()
    response = await origin(app, KEYS)(
        frame(signed(**changes), **packet), credential())
    assert response.status == 200
    assert app.seen[0][1] is None


@pytest.mark.asyncio
async def test_wrong_key_is_ignored():
    app = Recorder()
    await origin(app, GrantKeys((1, bytes(32)), None))(frame(signed()), credential())
    assert app.seen[0][1] is None


@pytest.mark.asyncio
async def test_replayed_frame_id_gets_no_grant_second_time():
    app = Recorder()
    runner = origin(app, KEYS)
    await runner(frame(signed()), credential())
    await runner(frame(signed()), credential())
    assert app.seen[0][1] is not None and app.seen[1][1] is None


@pytest.mark.asyncio
async def test_unknown_kid_asks_for_a_refresh_with_the_connection_id():
    asked = []
    app = Recorder()
    await origin(app, KEYS, asked.append)(frame(signed(key_version=2)), credential())
    assert asked == [CID] and app.seen[0][1] is None


@pytest.mark.asyncio
async def test_other_refusals_do_not_ask_for_a_refresh():
    asked = []
    await origin(Recorder(), KEYS, asked.append)(
        frame(signed(frame_id="other")), credential())
    assert asked == []


@pytest.mark.asyncio
async def test_refusal_is_logged_with_a_code_and_never_the_value(caplog):
    value = signed(frame_id="other")
    with caplog.at_level("INFO"):
        await origin(Recorder(), KEYS)(frame(value), credential())
    text = caplog.text
    assert "remote grant refused: mismatch" in text
    assert value not in text and value.split(".")[1] not in text


@pytest.mark.asyncio
async def test_key_lookup_failure_means_no_grant():
    def boom(cid):
        raise RuntimeError("keyring locked")

    app = Recorder()
    await CanonicalMcpOrigin(app, clock=lambda: NOW, grant_keys=boom)(
        frame(signed()), credential())
    assert app.seen[0][1] is None


@pytest.mark.asyncio
async def test_a_client_cannot_inject_a_grant_through_the_context_var():
    app = Recorder()
    # An enclosing remote_grant() must not leak into a frame with no valid grant.
    await origin(app, KEYS)(frame(), credential())
    assert app.seen[0][1] is None


def test_remote_grant_context_manager_sets_and_restores():
    assert current_remote_grant() is None
    sentinel = object()
    with remote_grant(sentinel):
        assert current_remote_grant() is sentinel
    assert current_remote_grant() is None


@pytest.mark.asyncio
async def test_missing_header_while_a_key_is_held_asks_for_a_refresh():
    asked = []
    app = Recorder()
    await origin(app, KEYS, asked.append)(frame(), credential())
    assert asked == [CID] and app.seen[0][1] is None


@pytest.mark.asyncio
async def test_missing_header_with_no_key_held_does_not_ask():
    asked = []
    await origin(Recorder(), GrantKeys(None, None), asked.append)(frame(), credential())
    assert asked == []
