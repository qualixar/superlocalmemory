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


def test_vendor_token_shapes():
    stripe = "sk_" + "live_" + "a1b2c3d4e5f6" * 2
    npm = "npm_" + "Ab1Cd2Ef3G" * 4
    for kind, token in (("STRIPE", stripe), ("NPM", npm)):
        text = f"token here {token} end"
        hits = detect_secrets(text)
        assert [(h.kind, text[h.start:h.end]) for h in hits] == [(kind, token)]


def test_kebab_case_slugs_are_not_keys():
    text = ("see [[risk-assessment-for-quarterly-review]] and task-management-system-overview, "
            "also disk-usage-report-for-the-whole-cluster")
    assert detect_secrets(text) == []


def test_a_real_sk_key_is_still_found_after_the_anchor():
    for text in ("key sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2", "(sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2)",
                 "OPENAI_API_KEY=sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"):
        assert detect_secrets(text), text


def test_other_aws_key_id_prefixes_are_found():
    for prefix in ("AKIA", "ASIA", "AGPA", "AIDA", "AROA", "ANPA", "ANVA", "AIPA"):
        key = prefix + "IOSFODNN7EXAMPLE"
        assert [h.kind for h in detect_secrets(f"id {key} end")] == ["AWS"], prefix


def test_an_aws_secret_access_key_label_is_found():
    secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    for text in (f"aws_secret_access_key = {secret}", f"AWS_SECRET_ACCESS_KEY={secret}",
                 f"aws_secret_access_key: {secret}"):
        assert [h.kind for h in detect_secrets(text)] == ["AWS_SECRET"], text


def test_a_url_with_a_password_is_found():
    for url in ("postgres://admin:Sup3rS3cret!@db.internal:5432/app", "https://user:hunter2@example.com/x",
                "redis://:p4ss@cache:6379/0"):
        assert [h.kind for h in detect_secrets(f"DATABASE_URL={url}")] == ["URL_PASSWORD"], url


def test_urls_without_a_password_are_not_credentials():
    assert detect_secrets("see https://example.com:8080/path and ssh://git@github.com/org/repo") == []


def test_the_shared_redaction_patterns_keep_their_original_sk_shapes():
    from superlocalmemory.core.security_primitives import _SECRET_PATTERNS, redact_secrets

    sk = [p.pattern for p, kind in _SECRET_PATTERNS if kind in ("OPENAI", "ANTHROPIC")]
    assert sk == [r"sk-ant-[A-Za-z0-9_\-]{20,}", r"sk-[A-Za-z0-9_\-]{20,}"]
    assert "sk-" not in redact_secrets("key sk-" + "a" * 30)


def test_detection_alone_ignores_kebab_case_words_but_still_finds_a_real_key():
    assert detect_secrets("see risk-assessment-for-quarterly-review today") == []
    assert [h.kind for h in detect_secrets("key sk-" + "a" * 30)] == ["OPENAI"]
