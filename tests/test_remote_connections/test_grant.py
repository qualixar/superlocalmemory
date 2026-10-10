"""The signed per-connection grant: byte-exact format, every refusal, replay."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading

import pytest

from superlocalmemory.remote_connections.grant import (
    GRANT_HEADER,
    SCOPE_ORDER,
    GrantError,
    GrantKeys,
    RemoteGrant,
    ReplayGuard,
    peer_ref,
    sign_grant,
    verify_grant,
)

KEY = bytes(range(1, 33))
CID = "c0ffee00c0ffee00c0ffee00c0ffee00"
NOW_MS = 1_700_000_000_000

# Produced by the TypeScript signer for the same key and claims.
TS_VECTOR = (
    "v1.eyJ2IjoxLCJraWQiOjEsImNpZCI6ImMwZmZlZTAwYzBmZmVlMDBjMGZmZWUwMGMwZmZlZTAwIiwiYWlkIjoiYXV0aC0xIiwidmVyIjoxLCJhcHAiOiJjbGllbnQtMSIsInNjcCI6WyJzbG06cmVhZCIsInNsbTptZXNoIl0sImZ2IjpmYWxzZSwiZmlkIjoiZnJhbWUtMSIsImdlbiI6MywiZGwiOjE3MDAwMDAwMDAwMDB9."
    "OhBbGApiFGlFV_pL3osXXRfjieznBJJr5_fn_fXkhH0"
)

VECTOR_PAYLOAD = (
    '{"v":1,"kid":1,"cid":"c0ffee00c0ffee00c0ffee00c0ffee00","aid":"auth-1","ver":1,'
    '"app":"client-1","scp":["slm:read","slm:mesh"],"fv":false,"fid":"frame-1","gen":3,'
    '"dl":1700000000000}'
)


def b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def mint(payload: str | dict, key: bytes = KEY) -> str:
    text = payload if isinstance(payload, str) else json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False)
    segment = b64(text.encode())
    mac = hmac.new(key, ("slm-grant-v1." + segment).encode("ascii"), hashlib.sha256).digest()
    return f"v1.{segment}.{b64(mac)}"


def claims(**changes) -> dict:
    base = {"v": 1, "kid": 1, "cid": CID, "aid": "auth-1", "ver": 1, "app": "client-1",
            "scp": ["slm:read", "slm:mesh"], "fv": False, "fid": "frame-1", "gen": 3,
            "dl": NOW_MS}
    base.update(changes)
    return base


def verify(value: str, *, keys=None, now_ms=NOW_MS, seen=None, cid=CID, fid="frame-1",
           gen=3, dl=NOW_MS) -> RemoteGrant:
    return verify_grant(value, keys=keys or GrantKeys((1, KEY), None), connection_id=cid,
                        frame_id=fid, generation=gen, deadline_at_ms=dl, now_ms=now_ms,
                        seen=seen or ReplayGuard())


def code_of(value: str, **kwargs) -> str:
    with pytest.raises(GrantError) as error:
        verify(value, **kwargs)
    return error.value.args[0]


def sign(**changes) -> str:
    args = dict(connection_id=CID, authorization_id="auth-1", authorization_version=1,
                app="client-1", scopes=("slm:read", "slm:mesh"), folders_visible=False,
                frame_id="frame-1", generation=3, deadline_at_ms=NOW_MS, key_version=1)
    args.update(changes)
    return sign_grant(KEY, **args)


def test_header_name_and_scope_order():
    assert GRANT_HEADER == "x-slm-grant"
    assert SCOPE_ORDER == ("slm:read", "slm:write", "slm:session", "slm:mesh", "slm:media")


def test_vector_payload_is_the_exact_compact_bytes():
    header = sign(deadline_at_ms=1_700_000_000_000)
    assert header.split(".")[1] == b64(VECTOR_PAYLOAD.encode())
    assert header == mint(VECTOR_PAYLOAD)


def test_cross_language_vector():
    if TS_VECTOR is None:
        pytest.skip("TypeScript test vector not filled in yet")
    assert sign() == TS_VECTOR
    assert verify(TS_VECTOR).scopes == {"slm:read", "slm:mesh"}


def test_sign_then_verify_round_trips_all_fields():
    grant = verify(sign(scopes=("slm:media", "slm:read", "slm:write")))
    assert grant == RemoteGrant(
        connection_id=CID, authorization_id="auth-1", authorization_version=1,
        app="client-1", scopes=frozenset({"slm:read", "slm:write", "slm:media"}),
        folders_visible=False, key_version=1)


def test_signing_puts_scopes_in_canonical_order():
    payload = json.loads(base64.urlsafe_b64decode(sign(
        scopes=("slm:media", "slm:read", "slm:mesh")).split(".")[1] + "=="))
    assert payload["scp"] == ["slm:read", "slm:mesh", "slm:media"]


def test_unicode_app_name_is_not_escaped():
    text = base64.urlsafe_b64decode(sign(app="café").split(".")[1] + "==").decode()
    assert "café" in text and "\\u" not in text
    assert verify(sign(app="café")).app == "café"


@pytest.mark.parametrize("value", [
    "", "v2.abc." + "A" * 43, "v1.." + "A" * 43, "v1.abc.short", "v1.abc." + "A" * 44,
    "v1.a b." + "A" * 43, "v1.abc", "v1." + "A" * 8001 + "." + "A" * 43,
    "v1.abc." + "A" * 42 + "=", "x" * 20,
])
def test_bad_shape_is_malformed(value):
    assert code_of(value) == "malformed"


@pytest.mark.parametrize("payload", [
    "not json", "[1]", '{"v":1}',
    json.dumps({**claims(), "extra": 1}),
    json.dumps({k: v for k, v in claims().items() if k != "fv"}),
    json.dumps(claims(v=2)),
    json.dumps(claims(v=True)),
    json.dumps(claims(kid=0)), json.dumps(claims(kid=True)), json.dumps(claims(kid="1")),
    json.dumps(claims(kid=2 ** 53)),
    json.dumps(claims(ver=0)), json.dumps(claims(gen=0)), json.dumps(claims(dl=-1)),
    json.dumps(claims(cid="bad id")), json.dumps(claims(aid="")),
    json.dumps(claims(aid="a" * 257)), json.dumps(claims(fid="x/y")),
    json.dumps(claims(app="")), json.dumps(claims(app="a" * 2049)),
    json.dumps(claims(app="bad\u0007name")),
    json.dumps(claims(scp=["slm:mesh"])),
    json.dumps(claims(scp=["slm:read", "slm:read"])),
    json.dumps(claims(scp=["slm:read", "slm:admin"])),
    json.dumps(claims(scp="slm:read")),
    json.dumps(claims(fv="no")), json.dumps(claims(fv=0)),
    '{"v":1,"v":1,"kid":1,"cid":"c","aid":"a","ver":1,"app":"x","scp":["slm:read"],'
    '"fv":false,"fid":"f","gen":1,"dl":0}',
])
def test_bad_payload_is_malformed(payload):
    assert code_of(mint(payload)) == "malformed"


def test_payload_not_base64_is_malformed():
    assert code_of("v1.!!!!.%s" % ("A" * 43)) == "malformed"


def test_unknown_kid_and_no_keys():
    assert code_of(mint(claims(kid=2))) == "unknown_kid"
    assert code_of(sign(), keys=GrantKeys(None, None)) == "unknown_kid"


def test_previous_key_is_accepted_only_inside_its_window():
    old = bytes(range(40, 72))
    value = mint(claims(kid=1), key=old)
    keys = GrantKeys((2, KEY), (1, old, NOW_MS / 1000 + 120))
    assert verify(value, keys=keys).key_version == 1
    assert code_of(value, keys=keys, now_ms=NOW_MS + 121_000, dl=NOW_MS + 121_000) == "unknown_kid"


def test_bad_mac_wrong_key_or_tampered_payload():
    assert code_of(mint(claims(), key=bytes(32))) == "bad_mac"
    good = sign()
    head, payload, mac = good.split(".")
    tampered = b64(base64.urlsafe_b64decode(payload + "==").replace(b"auth-1", b"auth-2"))
    assert code_of(f"{head}.{tampered}.{mac}") == "bad_mac"


def test_mac_is_over_the_received_segment_not_a_reserialisation():
    spaced = json.dumps(claims())  # spaces after separators, same meaning
    assert code_of(mint(spaced, key=KEY).rsplit(".", 1)[0] + "." + sign().rsplit(".", 1)[1]) == "bad_mac"
    assert verify(mint(spaced)).connection_id == CID


@pytest.mark.parametrize("kwargs", [
    {"cid": "f" * 32}, {"fid": "frame-2"}, {"gen": 4}, {"dl": NOW_MS + 1},
])
def test_binding_mismatch(kwargs):
    assert code_of(sign(), **kwargs) == "mismatch"


def test_expiry_allows_five_seconds_of_skew():
    assert verify(sign(), now_ms=NOW_MS + 5000).app == "client-1"
    assert code_of(sign(), now_ms=NOW_MS + 5001) == "expired"


def test_replay_is_refused_and_a_different_frame_is_not():
    seen = ReplayGuard()
    verify(sign(), seen=seen)
    assert code_of(sign(), seen=seen) == "replay"
    verify(sign(frame_id="frame-2"), seen=seen, fid="frame-2")


def test_failed_verification_does_not_burn_the_frame_id():
    seen = ReplayGuard()
    assert code_of(mint(claims(), key=bytes(32)), seen=seen) == "bad_mac"
    verify(sign(), seen=seen)


def test_replay_guard_prunes_expired_entries_and_refuses_when_full():
    guard = ReplayGuard()
    assert guard.check_and_add("a", 1000, 0)
    assert not guard.check_and_add("a", 1000, 0)
    assert guard.check_and_add("a", 5000, 6000)  # old entry pruned
    full = ReplayGuard()
    for index in range(4096):
        assert full.check_and_add(f"f{index}", 10 ** 12, 0)
    assert not full.check_and_add("one-more", 10 ** 12, 0)
    assert full.check_and_add("late", 10 ** 12, 10 ** 12 + 1)  # all expired, room again


def test_replay_guard_is_thread_safe():
    guard = ReplayGuard()
    wins = []

    def attempt():
        wins.append(guard.check_and_add("same", 10 ** 12, 0))

    threads = [threading.Thread(target=attempt) for _ in range(32)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert wins.count(True) == 1


def test_error_text_never_contains_the_header_value():
    value = sign()
    with pytest.raises(GrantError) as error:
        verify(value, cid="f" * 32)
    assert value not in str(error.value) and value.split(".")[2] not in repr(error.value)


def test_peer_ref_is_stable_and_shaped():
    expected = "w_" + hashlib.sha256(f"{CID}:auth-1".encode()).hexdigest()[:24]
    assert peer_ref(CID, "auth-1") == expected
    assert peer_ref(CID, "auth-1") == peer_ref(CID, "auth-1")
    assert peer_ref(CID, "auth-2") != expected
    assert peer_ref("f" * 32, "auth-1") != expected
