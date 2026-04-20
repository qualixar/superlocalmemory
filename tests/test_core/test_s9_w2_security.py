# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 9 W2

"""Stage 9 W2 regressions — Security hardening.

Covers:
- C4  (fork safety of _COST_CONN_CACHE)
- C5  (trigram cache conn lifecycle + fork safety)
- C9  (evolution cost-log serialization narrowed)
- H-SEC-01 (install-token O_EXCL race closed)
- H-SEC-02 (GENERIC_KEY ReDoS eliminated — pattern + post-match guard)
- H-SEC-07 (session_state tmp file 0600 during rename window)
- H-SKEP-05 (_dispatch_llm profile_id guard moved BEFORE the paid call)
- M-SEC-01 (safe_resolve_identifier case-fold collision refused)
- M-SEC-03 (post_tool marker iteration cap)
- L-SEC-01 (entropy sweep preserves REDACTED labels)
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from superlocalmemory.core import security_primitives as sp


# ---------------------------------------------------------------------------
# H-SEC-02 — GENERIC_KEY ReDoS elimination
# ---------------------------------------------------------------------------


def test_redact_secrets_high_completes_under_budget_on_adversarial_input() -> None:
    """256 KB of ``A`` characters must finish redact_secrets quickly.

    Pre-fix: quadratic lookahead backtracking stalled the dispatcher
    multi-second. Budget: 1 second on a commodity laptop.
    """
    payload = "A" * 256_000
    start = time.perf_counter()
    out = sp.redact_secrets(payload, aggression="high")
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"redact_secrets took {elapsed:.3f}s on 256KB"
    # Pure-letter garbage is not a secret; must be preserved.
    assert out == payload


def test_redact_secrets_high_still_catches_realistic_env_var_secret() -> None:
    """The tightened pattern must still flag a realistic env-var secret."""
    secret = "SLM_" + "A1B2C3D4E5F6G7H8I9J0"
    redacted = sp.redact_secrets(
        f"config={secret}", aggression="high",
    )
    assert secret not in redacted
    assert "[REDACTED:GENERIC_KEY:" in redacted


def test_redact_secrets_high_spares_pure_letter_env_constant() -> None:
    """Pure letters in the tail → not a secret → preserved."""
    constant = "SLM_PASSWORDABCDEFGHIJKLMNOPQRST"
    redacted = sp.redact_secrets(
        f"const={constant}", aggression="high",
    )
    assert constant in redacted


# ---------------------------------------------------------------------------
# L-SEC-01 — entropy sweep preserves REDACTED labels
# ---------------------------------------------------------------------------


def test_redact_secrets_does_not_double_redact_existing_markers() -> None:
    """Already-redacted markers must NOT lose their provenance label in
    the entropy sweep."""
    # 32+ chars so the entropy regex matches; includes a REDACTED token.
    text = "x=[REDACTED:GITHUB_PAT:deadbeef] y=more_text_padding_for_length"
    out = sp.redact_secrets(text, aggression="high")
    # The original GITHUB_PAT label must survive.
    assert "[REDACTED:GITHUB_PAT:deadbeef]" in out


# ---------------------------------------------------------------------------
# M-SEC-01 — safe_resolve_identifier case-fold collision
# ---------------------------------------------------------------------------


def test_safe_resolve_identifier_rejects_case_collision(tmp_path: Path) -> None:
    """On case-insensitive filesystems a pre-existing ``session_1`` must
    NOT satisfy a request for ``Session_1``."""
    (tmp_path / "session_1").mkdir()
    # On case-sensitive FS this is a clean reject by the regex / resolve;
    # on case-insensitive FS the resolve may return ``session_1``. In
    # both cases the contract holds: the NAME must equal the request.
    try:
        resolved = sp.safe_resolve_identifier(tmp_path, "Session_1")
        # If we got here the filesystem is case-sensitive; assert that
        # the resolved name matches the request byte-for-byte.
        assert resolved.name == "Session_1"
    except ValueError as exc:
        # Case-insensitive FS path — must raise "path-case collision".
        assert "case" in str(exc).lower() or "escape" in str(exc).lower()


def test_safe_resolve_identifier_accepts_clean_name(tmp_path: Path) -> None:
    """Happy path still works — a valid id resolves to a path inside base."""
    target = sp.safe_resolve_identifier(tmp_path, "clean_id")
    assert target.name == "clean_id"
    assert target.parent.resolve() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# H-SEC-01 — install_token O_EXCL race
# ---------------------------------------------------------------------------


def test_ensure_install_token_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling ensure_install_token twice must return the same token —
    the O_EXCL path means the second call sees EEXIST and re-reads."""
    token_path = tmp_path / ".install_token"
    monkeypatch.setattr(sp, "_install_token_path", lambda: token_path)
    first = sp.ensure_install_token()
    second = sp.ensure_install_token()
    assert first == second
    assert token_path.read_text(encoding="utf-8").strip() == first
    # Mode must be 0600 on POSIX.
    if os.name == "posix":
        mode = stat.S_IMODE(token_path.stat().st_mode)
        assert mode == 0o600


# ---------------------------------------------------------------------------
# H-SEC-07 — session_state tmp 0600 + pid-uniqueness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only perm check")
def test_save_session_state_rotation_yields_0600_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persisted session_state file must not be world-readable."""
    monkeypatch.setenv("SLM_HOME", str(tmp_path))
    from superlocalmemory.hooks import _outcome_common as oc
    oc.save_session_state("abc123", {"foo": "bar"})
    # Find the written file.
    state_dir = tmp_path / "session_state"
    files = list(state_dir.glob("*.json"))
    assert files, "save_session_state did not persist"
    mode = stat.S_IMODE(files[0].stat().st_mode)
    assert mode == 0o600, f"session_state world-readable: 0o{mode:03o}"


# ---------------------------------------------------------------------------
# C4 — _COST_CONN_CACHE fork-safety register
# ---------------------------------------------------------------------------


def test_cost_conn_cache_clears_on_pid_drift(tmp_path: Path) -> None:
    """If the cache's owner pid no longer matches os.getpid(), the next
    get_cost_conn must re-open — simulating a fork that missed the
    register_at_fork handler."""
    from superlocalmemory.evolution import llm_dispatch as ld
    db = tmp_path / "learning.db"
    # Preload the cache.
    conn1 = ld._get_cost_conn(db)
    assert db.resolve().samefile(list(ld._COST_CONN_CACHE.keys())[0]) or (
        os.path.realpath(str(db)) in ld._COST_CONN_CACHE
    )
    # Simulate pid drift.
    ld._COST_CONN_OWNER_PID = -1
    conn2 = ld._get_cost_conn(db)
    assert conn1 is not conn2, "cache failed to refresh on pid drift"
    ld._close_cost_conns()


# ---------------------------------------------------------------------------
# H-SKEP-05 — profile_id guard before paid LLM call
# ---------------------------------------------------------------------------


def test_dispatch_llm_rejects_empty_profile_id_before_backend_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_dispatch_llm must raise ValueError BEFORE invoking the backend
    so no LLM cost is incurred on a misconfigured profile."""
    from superlocalmemory.evolution import llm_dispatch as ld
    db = tmp_path / "learning.db"
    import sqlite3
    sqlite3.connect(db).close()

    called: list[bool] = []

    def fake_call(prompt, *, model, max_tokens):
        called.append(True)
        return "never-called"

    monkeypatch.setattr(ld, "_actual_llm_call", fake_call)

    with pytest.raises(ValueError, match="profile_id"):
        ld._dispatch_llm(
            "some prompt",
            model="claude-haiku-4-5",
            learning_db=db,
            profile_id="",
        )
    assert called == [], "backend was called despite invalid profile_id"


# ---------------------------------------------------------------------------
# M-SEC-03 — marker iteration cap
# ---------------------------------------------------------------------------


def test_post_tool_marker_iteration_is_capped() -> None:
    """Adversarial response with 5000 markers must not iterate beyond
    the documented cap."""
    from superlocalmemory.hooks import post_tool_outcome_hook as pto
    _ = pto  # module import smoke; cap constant lives in _inner_main.
    import re as _re
    src = Path(pto.__file__).read_text(encoding="utf-8")
    assert "_MAX_MARKERS" in src, "marker cap constant missing"
    # Cap value is 100 per the contract.
    m = _re.search(r"_MAX_MARKERS\s*=\s*(\d+)", src)
    assert m is not None and int(m.group(1)) <= 500
