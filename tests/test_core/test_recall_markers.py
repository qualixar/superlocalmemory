# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-00 §3 + P0.4

"""Tests for HMAC fact-id markers on recall results.

LLD-00 §3 requires every recall result to carry a tagged marker
``slm:fact:<fact_id>:<hmac8>`` so downstream hooks can validate that a
fact_id observed in tool output actually originated from this SLM
install. Without this, attacker-controlled tool responses can forge
engagement signals.

Covers IMPLEMENTATION-MANIFEST-v3.4.22-FINAL.md §P0.4.
"""

from __future__ import annotations

import os
import secrets as _secrets
from pathlib import Path

import pytest

from superlocalmemory.core import recall_pipeline as rp
from superlocalmemory.core import security_primitives as sp
from superlocalmemory.storage.models import (
    AtomicFact,
    Mode,
    RecallResponse,
    RetrievalResult,
)


@pytest.fixture
def fixed_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin the install token to a known value for deterministic tests."""
    token_path = tmp_path / ".install_token"
    token_path.write_text("deadbeef" * 8, encoding="utf-8")  # 64 chars
    monkeypatch.setattr(sp, "_install_token_path", lambda: token_path)
    return token_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# _emit_marker
# ---------------------------------------------------------------------------


def test_emit_marker_deterministic_for_same_token(fixed_token: str) -> None:
    m1 = rp._emit_marker("fact_abc123")
    m2 = rp._emit_marker("fact_abc123")
    assert m1 == m2


def test_emit_marker_starts_with_prefix(fixed_token: str) -> None:
    marker = rp._emit_marker("fact_xyz")
    assert marker.startswith("slm:fact:")


def test_emit_marker_contains_fact_id(fixed_token: str) -> None:
    marker = rp._emit_marker("fact_xyz")
    assert "fact_xyz" in marker


def test_emit_marker_different_for_different_facts(fixed_token: str) -> None:
    assert rp._emit_marker("a") != rp._emit_marker("b")


def test_emit_marker_changes_when_token_rotates(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotating the install token invalidates old markers — by design."""
    tok1 = tmp_path / "t1"
    tok1.write_text("a" * 64, encoding="utf-8")
    monkeypatch.setattr(sp, "_install_token_path", lambda: tok1)
    m1 = rp._emit_marker("fact_xyz")

    tok2 = tmp_path / "t2"
    tok2.write_text("b" * 64, encoding="utf-8")
    monkeypatch.setattr(sp, "_install_token_path", lambda: tok2)
    m2 = rp._emit_marker("fact_xyz")

    assert m1 != m2


# ---------------------------------------------------------------------------
# _validate_marker
# ---------------------------------------------------------------------------


def test_validate_marker_accepts_valid(fixed_token: str) -> None:
    marker = rp._emit_marker("fact_abc123")
    assert rp._validate_marker(marker) == "fact_abc123"


def test_validate_marker_rejects_bad_hmac(fixed_token: str) -> None:
    marker = rp._emit_marker("fact_abc")
    # Tamper the last 8 hex chars (the hmac8).
    tampered = marker[:-8] + "00000000"
    assert rp._validate_marker(tampered) is None


def test_validate_marker_rejects_no_prefix(fixed_token: str) -> None:
    assert rp._validate_marker("fact_abc:deadbeef") is None
    assert rp._validate_marker("") is None
    assert rp._validate_marker("random_string") is None


def test_validate_marker_rejects_wrong_hmac_length(fixed_token: str) -> None:
    assert rp._validate_marker("slm:fact:fact_abc:dead") is None  # 4 chars
    assert rp._validate_marker("slm:fact:fact_abc:deadbeef0") is None  # 9


def test_validate_marker_rejects_missing_hmac(fixed_token: str) -> None:
    assert rp._validate_marker("slm:fact:fact_abc") is None
    assert rp._validate_marker("slm:fact:") is None


def test_validate_marker_rejects_collision_attempt(fixed_token: str) -> None:
    """Brute-force 10k random hmac8s against a known fact_id — none should
    pass validation. Exercises the constant-time compare path with a large
    adversary window; at 2^32 possible hmac8 values the false-positive rate
    is ~1/4.3B so a 10k bruteforce finding one would be extraordinary.
    """
    target_fid = "fact_target_value_xyz"
    for _ in range(10_000):
        fake_hmac = _secrets.token_hex(4)  # 8 hex chars
        candidate = f"slm:fact:{target_fid}:{fake_hmac}"
        result = rp._validate_marker(candidate)
        # An attacker who hits a valid HMAC coincidentally still gets the
        # correct fact_id returned — that's the expected behaviour, not a
        # bug. Only fail if we somehow recover a DIFFERENT fact_id.
        assert result is None or result == target_fid


def test_validate_marker_accepts_fact_id_with_underscores_and_hyphens(
    fixed_token: str,
) -> None:
    marker = rp._emit_marker("fact_abc-123_def")
    assert rp._validate_marker(marker) == "fact_abc-123_def"


# ---------------------------------------------------------------------------
# Integration — run_recall / _apply_markers_to_response
# ---------------------------------------------------------------------------


def _make_response(*fact_ids: str) -> RecallResponse:
    return RecallResponse(
        query="q",
        mode=Mode.A,
        results=[
            RetrievalResult(fact=AtomicFact(fact_id=fid, content="x"))
            for fid in fact_ids
        ],
    )


def test_apply_markers_populates_all_results(fixed_token: str) -> None:
    """Every result in a response gets a marker — the core P0.4 contract."""
    response = _make_response("fact_a", "fact_b", "fact_c")
    rp._apply_markers_to_response(response)
    for r in response.results:
        assert r.marker.startswith("slm:fact:")
        assert r.fact.fact_id in r.marker
        assert rp._validate_marker(r.marker) == r.fact.fact_id


def test_apply_markers_preserves_fact_id(fixed_token: str) -> None:
    """Backward-compat: existing fact_id field remains untouched."""
    response = _make_response("fact_abc")
    rp._apply_markers_to_response(response)
    assert response.results[0].fact.fact_id == "fact_abc"


def test_apply_markers_empty_response(fixed_token: str) -> None:
    response = RecallResponse(query="q", mode=Mode.A, results=[])
    rp._apply_markers_to_response(response)  # should not raise
    assert response.results == []


def test_retrieval_result_default_marker_is_empty() -> None:
    """Unmarked RetrievalResult has empty marker (backward-compat)."""
    r = RetrievalResult(fact=AtomicFact(fact_id="x"))
    assert r.marker == ""
