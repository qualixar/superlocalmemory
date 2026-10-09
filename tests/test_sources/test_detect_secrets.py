"""Credential shapes only: a key is found, a hash or ordinary prose is not."""

from __future__ import annotations

import hashlib

from superlocalmemory.core.security_primitives import SecretHit, detect_secrets


def test_no_text_no_hits():
    assert detect_secrets("") == []
    assert detect_secrets("plain meeting notes, nothing secret") == []


def test_fake_key_shapes_are_found_with_positions():
    key = "AKIA" + "ABCDEFGHIJKLMNOP"
    text = f"my aws id is {key} ok"
    hits = detect_secrets(text)
    assert hits == [SecretHit("AWS", text.index(key), text.index(key) + len(key))]


def test_each_known_shape():
    samples = {
        "ANTHROPIC": "sk-ant-" + "a1B2c3D4e5F6g7H8i9J0k1",
        "GITHUB": "ghp_" + "a" * 36,
        "SLACK": "xoxb-" + "1234567890-abc",
        "PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----",
        "JWT": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghij",
    }
    for kind, sample in samples.items():
        kinds = {h.kind for h in detect_secrets(f"x {sample} y")}
        assert kind in kinds, kind


def test_sha256_hex_is_not_a_credential():
    digest = hashlib.sha256(b"hello").hexdigest()
    assert detect_secrets(f"commit {digest} and sum {digest}") == []


def test_long_random_looking_words_are_not_credentials():
    assert detect_secrets("x" + "Zq3vN8rT1yUb5Wk9Xc2Lm7Pd4Hs6Jf0G" * 3) == []


def test_hits_are_ordered_and_do_not_overlap():
    text = "ghp_" + "b" * 36 + " then " + "AKIA" + "ABCDEFGHIJKLMNOP"
    hits = detect_secrets(text)
    assert [h.start for h in hits] == sorted(h.start for h in hits)
    assert all(a.end <= b.start for a, b in zip(hits, hits[1:]))
