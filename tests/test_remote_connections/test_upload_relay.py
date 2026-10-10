"""Upload frames: the only extra thing a relayed request can reach, gated by the one-time token."""
from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass

import pytest

from superlocalmemory.media.upload_links import UploadLinks
from superlocalmemory.remote_connections import codec
from superlocalmemory.remote_connections.credentials import ConnectorCredential
from superlocalmemory.remote_connections.origin import CanonicalMcpOrigin
from superlocalmemory.remote_connections.upload_relay import UPLOAD_HEADER, UploadRelay

CID = "a" * 32
SECRET = "slmr_" + "b" * 43
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100
NOW = 1_700_000_000.0


@dataclass
class FakeKey:
    key_id: str = "key1"
    name: str = "web-" + CID
    scope: str = "write"
    profile: str = "personal"
    extras: frozenset = frozenset({"media"})


class FakeKeys:
    def __init__(self, key=None):
        self.key = key or FakeKey()

    def verify(self, presented):
        return self.key if presented == SECRET and self.key is not None else None


class Finisher:
    def __init__(self, reply=None, delay=0.0):
        self.reply = reply if reply is not None else {"status": "stored", "media_id": "m" * 32}
        self.calls = []
        self.delay = delay

    def __call__(self, upload_id):
        import time
        self.calls.append(upload_id)
        time.sleep(self.delay)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def credential():
    return ConnectorCredential("install-a", "owner-a", "personal", CID, 1, int((NOW + 600) * 1000),
                               "a" * 64, SECRET)


def frame(op, token, index=0, total=0, body=b"", fid="wire_1"):
    return {"v": 1, "kind": "request", "id": fid, "generation": 1, "deadlineAt": int(NOW * 1000) + 20_000,
            "headers": [["content-type", "application/octet-stream"],
                        [UPLOAD_HEADER, f"{op} {token} {index} {total}"]],
            "bodyBase64": base64.b64encode(body).decode()}


@pytest.fixture()
def setup(tmp_path):
    links = UploadLinks(tmp_path, clock=lambda: NOW)
    finisher = Finisher()
    relay = UploadRelay(lambda: links, keys=FakeKeys(), finisher=finisher, finish_wait_s=2.0)
    minted = links.mint(CID, "key1", "personal", "image", "a note")
    return relay, links, finisher, minted


async def call(relay, packet):
    response = await relay.handle(packet, credential())
    assert response.status == 200 and dict(response.headers)["content-type"] == "application/json"
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_info_reports_kind_and_limit_without_using_the_link(setup):
    relay, links, _, minted = setup
    out = await call(relay, frame("info", minted.token))
    assert out == {"ok": True, "kind": "image", "max_bytes": 25 * 1024 * 1024, "expires_at": minted.expires_at}
    assert links.find(minted.token, CID).state == "open"


@pytest.mark.asyncio
async def test_a_bad_token_gets_a_plain_refusal(setup):
    relay, _, _, _ = setup
    out = await call(relay, frame("info", "z" * 43))
    assert out["ok"] is False and out["code"] == "invalid_link" and "not valid" in out["message"]


@pytest.mark.asyncio
async def test_chunks_then_finish_saves_through_the_finisher_once(setup):
    relay, links, finisher, minted = setup
    body = PNG + b"12345"
    assert (await call(relay, frame("chunk", minted.token, 0, len(body), body[:60])))["received"] == 60
    assert (await call(relay, frame("chunk", minted.token, 1, len(body), body[60:])))["received"] == len(body)
    first = await call(relay, frame("finish", minted.token, 0, len(body)))
    assert first == {"ok": True, "done": True, "message": "Saved to your memory."}
    again = await call(relay, frame("finish", minted.token, 0, len(body)))
    assert again == first and len(finisher.calls) == 1


@pytest.mark.asyncio
async def test_a_slow_save_answers_working_then_the_result(tmp_path):
    links = UploadLinks(tmp_path, clock=lambda: NOW)
    finisher = Finisher(delay=0.4)
    relay = UploadRelay(lambda: links, keys=FakeKeys(), finisher=finisher, finish_wait_s=0.05)
    minted = links.mint(CID, "key1", "personal", "image", "")
    await call(relay, frame("chunk", minted.token, 0, len(PNG), PNG))
    assert (await call(relay, frame("finish", minted.token, 0, len(PNG)))) == {"ok": True, "done": False}
    assert (await call(relay, frame("finish", minted.token, 0, len(PNG)))) == {"ok": True, "done": False}
    await asyncio.sleep(0.6)
    assert (await call(relay, frame("finish", minted.token, 0, len(PNG))))["done"] is True
    assert len(finisher.calls) == 1


@pytest.mark.asyncio
async def test_outcomes_map_to_plain_messages(tmp_path):
    cases = [
        ({"status": "duplicate", "media_id": "m"}, True, "That picture was already in your memory."),
        ({"status": "refused", "reason": "That file type is not supported (PNG, JPEG, GIF and WEBP only)."},
         False, "That file type is not supported"),
        ({"status": "warming", "reason": "images are starting up; try again in a minute"}, False, "starting up"),
        ({"success": False, "status": "unavailable"}, False, "could not be saved"),
    ]
    for reply, done, text in cases:
        links = UploadLinks(tmp_path / str(len(text)), clock=lambda: NOW)
        relay = UploadRelay(lambda: links, keys=FakeKeys(), finisher=Finisher(reply), finish_wait_s=2.0)
        minted = links.mint(CID, "key1", "personal", "image", "")
        await call(relay, frame("chunk", minted.token, 0, len(PNG), PNG))
        out = await call(relay, frame("finish", minted.token, 0, len(PNG)))
        assert out["ok"] is done and text in out["message"], (reply, out)
        state = links.find(minted.token, CID).state
        assert state == ("done" if done else "receiving" if reply["status"] == "warming" else "failed")


@pytest.mark.asyncio
async def test_a_finisher_that_raises_fails_the_link_without_leaking(tmp_path):
    links = UploadLinks(tmp_path, clock=lambda: NOW)
    relay = UploadRelay(lambda: links, keys=FakeKeys(), finisher=Finisher(RuntimeError("/Users/x/secret.png")),
                        finish_wait_s=2.0)
    minted = links.mint(CID, "key1", "personal", "image", "")
    await call(relay, frame("chunk", minted.token, 0, len(PNG), PNG))
    out = await call(relay, frame("finish", minted.token, 0, len(PNG)))
    assert out["ok"] is False and "secret" not in json.dumps(out) and "/Users" not in json.dumps(out)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"key": None},                                  # key revoked
    {"key": FakeKey(extras=frozenset())},           # media opt-in removed
    {"key": FakeKey(scope="read")},                 # no longer a write key
    {"key": FakeKey(key_id="other")},               # a different key than the one that minted the link
    {"key": FakeKey(profile="work")},               # re-bound to another profile
    {"key": FakeKey(name="web-" + "c" * 32)},       # not this connection's key
])
async def test_access_taken_away_mid_upload_aborts_every_step(setup, change):
    relay, links, finisher, minted = setup
    await call(relay, frame("chunk", minted.token, 0, len(PNG) + 5, PNG))
    relay._keys = FakeKeys(change["key"]) if change["key"] is not None else _NoKey()
    for packet in (frame("info", minted.token), frame("chunk", minted.token, 1, len(PNG) + 5, b"12345"),
                   frame("finish", minted.token, 0, len(PNG) + 5)):
        out = await call(relay, packet)
        assert out["ok"] is False and out["code"] == "not_allowed"
    assert finisher.calls == []


class _NoKey:
    def verify(self, presented):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [
    "", "chunk", "chunk " + "a" * 43, "chunk " + "a" * 43 + " 0", "other " + "a" * 43 + " 0 0",
    "chunk " + "a" * 42 + " 0 0", "chunk " + "a" * 43 + " -1 5", "chunk " + "a" * 43 + " 0 5 extra",
    "chunk " + "a" * 43 + " 99999999999 5", "chunk  " + "a" * 43 + " 0 5", "CHUNK " + "a" * 43 + " 0 5",
])
async def test_malformed_upload_headers_are_refused(setup, value):
    relay, _, _, _ = setup
    packet = frame("info", "a" * 43)
    packet["headers"][1] = [UPLOAD_HEADER, value]
    out = await call(relay, packet)
    assert out["ok"] is False and out["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_info_and_finish_carry_no_body_and_chunks_always_do(setup):
    relay, _, _, minted = setup
    assert (await call(relay, frame("info", minted.token, body=b"x")))["code"] == "invalid_request"
    assert (await call(relay, frame("chunk", minted.token, 0, 5, b"")))["code"] == "empty"


class Recorder:
    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        raise AssertionError("an upload frame must never reach the MCP app")


@pytest.mark.asyncio
async def test_the_origin_diverts_upload_frames_and_never_reaches_the_app(setup):
    relay, _, _, minted = setup
    app = Recorder()
    origin = CanonicalMcpOrigin(app, clock=lambda: NOW)
    origin._uploads = relay
    response = await origin(frame("info", minted.token), credential())
    assert json.loads(response.body)["ok"] is True and not app.called


def test_the_codec_admits_the_header_on_requests_only():
    wire = codec.encode_frame(frame("info", "a" * 43))
    assert codec.decode_frame(wire)["headers"][1][0] == UPLOAD_HEADER
    bad = frame("info", "a" * 43)
    bad["kind"], bad["status"] = "response", 200
    del bad["deadlineAt"]
    with pytest.raises(codec.FrameError):
        codec.encode_frame(bad)
