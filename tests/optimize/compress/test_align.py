"""Tests for align.py — CacheAligner volatile token detection."""
from __future__ import annotations

from superlocalmemory.optimize.compress.align import (
    CacheAligner,
    AlignResult,
    VolatileFinding,
    _detect,
    _is_uuid,
    _is_iso8601,
    _is_jwt_shape,
    _is_hex_hash,
    _split_tokens,
)


def test_align_clean_prompt_returns_stable() -> None:
    aligner = CacheAligner()
    result = aligner.detect("You are a helpful assistant.")
    assert result.prefix_stable is True
    assert result.stability_score == 1.0
    assert result.findings == []


def test_align_uuid_detected() -> None:
    aligner = CacheAligner()
    result = aligner.detect(
        "Session ID: 550e8400-e29b-41d4-a716-446655440000. Help the user."
    )
    assert any(f.label == "uuid" for f in result.findings)
    assert result.prefix_stable is False


def test_align_iso8601_detected() -> None:
    aligner = CacheAligner()
    result = aligner.detect("Current date: 2026-06-07T14:30:00Z. Answer accordingly.")
    assert any(f.label == "iso8601" for f in result.findings)


def test_align_hex_hash_detected() -> None:
    aligner = CacheAligner()
    sha256 = "a" * 64
    result = aligner.detect(f"Commit: {sha256}. Review it.")
    assert any(f.label == "hex_hash" for f in result.findings)


def test_align_no_mutation() -> None:
    aligner = CacheAligner()
    prompt = "Session: 550e8400-e29b-41d4-a716-446655440000. Help."
    original = prompt
    _ = aligner.detect(prompt)
    assert prompt == original


def test_align_stability_score_range() -> None:
    aligner = CacheAligner()
    for text in ["", "clean", "uid: " + "x" * 36 + " " * 100]:
        result = aligner.detect(text)
        assert 0.0 <= result.stability_score <= 1.0


def test_align_never_raises() -> None:
    aligner = CacheAligner()
    # Force _split_tokens to receive weird input
    for text in ["", "a" * 100000, "normal text here"]:
        result = aligner.detect(text)
        assert isinstance(result, AlignResult)


def test_align_jwt_in_markdown_backtick_detected() -> None:
    """B-14: JWTs in backtick code spans must be detected."""
    aligner = CacheAligner()
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4"
    result = aligner.detect(f"Token: `{jwt}` — use this for auth.")
    assert any(f.label == "jwt" for f in result.findings), (
        "JWT in markdown backtick code span was not detected"
    )


def test_align_findings_capped_at_20() -> None:
    aligner = CacheAligner()
    many_uuids = " ".join([f"550e8400-e29b-41d4-a716-{i:012d}" for i in range(50)])
    result = aligner.detect(many_uuids)
    assert len(result.findings) <= 20
    assert result.prefix_stable is False


# ── Unit tests for individual detectors ───────────────────────────────────

def test_is_uuid_valid() -> None:
    assert _is_uuid("550e8400-e29b-41d4-a716-446655440000") is True


def test_is_uuid_invalid() -> None:
    assert _is_uuid("not-a-uuid") is False
    assert _is_uuid("550e8400-e29b-41d4-a716") is False  # too short


def test_is_iso8601_valid() -> None:
    assert _is_iso8601("2026-06-07T14:30:00") is True
    assert _is_iso8601("2026-06-07T14:30:00Z") is True


def test_is_iso8601_invalid() -> None:
    assert _is_iso8601("hello") is False
    assert _is_iso8601("2026") is False


def test_is_hex_hash_valid() -> None:
    assert _is_hex_hash("a" * 32) is True
    assert _is_hex_hash("b" * 40) is True
    assert _is_hex_hash("c" * 64) is True


def test_is_hex_hash_invalid() -> None:
    assert _is_hex_hash("g" * 32) is False
    assert _is_hex_hash("a" * 10) is False


def test_split_tokens_strips_backtick() -> None:
    """B-14: backtick must be in strip set."""
    tokens = _split_tokens("`hello` `world`")
    assert "hello" in tokens
    assert "world" in tokens
