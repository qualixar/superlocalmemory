# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — LLD-07

"""Tests for superlocalmemory.core.security_primitives.

Covers LLD-07 §8.3 — safe_resolve, verify_sha256, redact_secrets,
ensure_install_token, verify_install_token, run_subprocess_safe.

RED→GREEN→REFACTOR. These were written before the implementation.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from superlocalmemory.core import security_primitives as sp


# ---------------------------------------------------------------------------
# safe_resolve
# ---------------------------------------------------------------------------


def test_safe_resolve_rejects_dot_dot(tmp_path: Path) -> None:
    base = tmp_path / "project"
    base.mkdir()
    with pytest.raises(sp.PathTraversalError):
        sp.safe_resolve(base, "../escape.txt")


def test_safe_resolve_rejects_escape_via_absolute(tmp_path: Path) -> None:
    base = tmp_path / "project"
    base.mkdir()
    other = tmp_path / "other" / "file.txt"
    other.parent.mkdir()
    other.write_text("x")
    with pytest.raises(sp.PathTraversalError):
        sp.safe_resolve(base, str(other))


def test_safe_resolve_rejects_deny_prefix_posix(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX-only deny prefixes")
    # Use /usr — stable across macOS and Linux.
    with pytest.raises(sp.PathTraversalError):
        sp.safe_resolve(Path("/"), "usr/local/file")


def test_safe_resolve_allows_legit_relative(tmp_path: Path) -> None:
    base = tmp_path / "project"
    nested = base / ".cursor" / "rules"
    nested.mkdir(parents=True)
    resolved = sp.safe_resolve(base, ".cursor/rules/file.mdc")
    assert str(resolved).startswith(str(base.resolve()))
    assert resolved.name == "file.mdc"


def test_safe_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("symlink support inconsistent on Windows CI")
    base = tmp_path / "project"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = base / "escape"
    link.symlink_to(outside)
    with pytest.raises(sp.PathTraversalError):
        sp.safe_resolve(base, "escape/x.txt")


def test_safe_resolve_rejects_non_string_rel(tmp_path: Path) -> None:
    base = tmp_path / "project"
    base.mkdir()
    with pytest.raises(TypeError):
        sp.safe_resolve(base, 123)  # type: ignore[arg-type]


def test_safe_resolve_accepts_path_rel(tmp_path: Path) -> None:
    base = tmp_path / "project"
    (base / "a").mkdir(parents=True)
    resolved = sp.safe_resolve(base, Path("a") / "b.txt")
    assert resolved.name == "b.txt"


# ---------------------------------------------------------------------------
# verify_sha256
# ---------------------------------------------------------------------------


def test_verify_sha256_accepts_correct() -> None:
    data = b"hello world"
    digest = hashlib.sha256(data).hexdigest()
    sp.verify_sha256(data, digest)  # no exception


def test_verify_sha256_rejects_tamper() -> None:
    data = b"hello world"
    bad = "0" * 64
    with pytest.raises(sp.IntegrityError):
        sp.verify_sha256(data, bad)


def test_verify_sha256_rejects_wrong_length() -> None:
    with pytest.raises(sp.IntegrityError):
        sp.verify_sha256(b"x", "abc")


def test_verify_sha256_case_insensitive() -> None:
    data = b"abc"
    digest = hashlib.sha256(data).hexdigest().upper()
    sp.verify_sha256(data, digest)  # no exception


def test_verify_sha256_rejects_non_string() -> None:
    with pytest.raises(sp.IntegrityError):
        sp.verify_sha256(b"x", 12345)  # type: ignore[arg-type]


def test_safe_resolve_allows_internal_symlink(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("symlink support inconsistent on Windows CI")
    base = tmp_path / "base"
    (base / "real").mkdir(parents=True)
    link = base / "alias"
    link.symlink_to(base / "real")
    # A symlink that stays within base should be allowed — exercises the
    # symlink-walk success branch.
    resolved = sp.safe_resolve(base, "alias/file.txt")
    assert str(resolved).startswith(str(base.resolve()))


def test_safe_resolve_rejects_parent_symlink_escape(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("symlink support inconsistent on Windows CI")
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    # Make a parent directory that IS a symlink escaping base.
    parent_link = base / "evil_parent"
    parent_link.symlink_to(outside)
    # The resolved path would escape base via this symlink parent.
    with pytest.raises(sp.PathTraversalError):
        sp.safe_resolve(base, "evil_parent/child.txt")


def test_redact_preserves_low_entropy_identifier() -> None:
    # 40 char low-entropy string (all same char) — should NOT be redacted.
    text = "var = " + ("a" * 40)
    redacted = sp.redact_secrets(text)
    assert ("a" * 40) in redacted


def test_redact_pem_private_key_header() -> None:
    text = "-----BEGIN RSA PRIVATE KEY-----"
    redacted = sp.redact_secrets(text)
    assert "[REDACTED:PRIVATE_KEY:" in redacted


# ---------------------------------------------------------------------------
# redact_secrets
# ---------------------------------------------------------------------------


def test_redact_openai_key() -> None:
    text = "my key is sk-" + "A" * 40 + " end"
    redacted = sp.redact_secrets(text)
    assert "sk-AAAA" not in redacted
    assert "[REDACTED:OPENAI:" in redacted


def test_redact_anthropic_key() -> None:
    text = "sk-ant-" + "B" * 40
    redacted = sp.redact_secrets(text)
    assert "[REDACTED:ANTHROPIC:" in redacted
    assert "sk-ant-BBBB" not in redacted


def test_redact_github_token() -> None:
    text = "ghp_" + "C" * 36
    redacted = sp.redact_secrets(text)
    assert "[REDACTED:GITHUB:" in redacted


def test_redact_aws_key() -> None:
    text = "AKIA" + "D" * 16
    redacted = sp.redact_secrets(text)
    assert "[REDACTED:AWS:" in redacted


def test_redact_jwt() -> None:
    text = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    redacted = sp.redact_secrets(text)
    assert "[REDACTED:JWT:" in redacted


def test_redact_entropy_catches_random_string() -> None:
    # 64-char mixed-alphabet random-looking token — entropy > 4.5
    random_blob = "aB3xQ9zP2mN7vK8jL5rT4wY6hG1fD0sC-bX_vZ9qM8nK7jL6hF5dG4sA3pO2iU1"
    text = f"token={random_blob} end"
    redacted = sp.redact_secrets(text)
    assert random_blob not in redacted
    assert "[REDACTED:" in redacted


def test_redact_preserves_prose() -> None:
    text = "The quick brown fox jumps over the lazy dog multiple times today."
    redacted = sp.redact_secrets(text)
    assert redacted == text


def test_redact_preserves_short_strings() -> None:
    text = "id=abc123"
    redacted = sp.redact_secrets(text)
    assert redacted == text


def test_redact_non_string_returns_same() -> None:
    # Defensive behavior — non-str inputs should not crash.
    assert sp.redact_secrets("") == ""


# ---------------------------------------------------------------------------
# Install token
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(sp, "_install_token_path", lambda: tmp_path / ".install_token")
    return tmp_path


def test_install_token_created(tmp_home: Path) -> None:
    token = sp.ensure_install_token()
    assert isinstance(token, str)
    assert len(token) >= 32
    token_file = tmp_home / ".install_token"
    assert token_file.exists()


def test_install_token_file_mode_0600(tmp_home: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX-only mode check")
    sp.ensure_install_token()
    token_file = tmp_home / ".install_token"
    mode = stat.S_IMODE(token_file.stat().st_mode)
    assert mode == 0o600


def test_install_token_idempotent(tmp_home: Path) -> None:
    t1 = sp.ensure_install_token()
    t2 = sp.ensure_install_token()
    assert t1 == t2


def test_verify_install_token_correct(tmp_home: Path) -> None:
    token = sp.ensure_install_token()
    assert sp.verify_install_token(token) is True


def test_verify_install_token_wrong(tmp_home: Path) -> None:
    sp.ensure_install_token()
    assert sp.verify_install_token("not the token") is False


def test_verify_install_token_missing_file(tmp_home: Path) -> None:
    # No file yet → must be False, not crash
    assert sp.verify_install_token("anything") is False


def test_verify_install_token_empty(tmp_home: Path) -> None:
    sp.ensure_install_token()
    assert sp.verify_install_token("") is False


def test_verify_install_token_empty_stored_file(tmp_home: Path) -> None:
    # A token file that exists but is empty — verify must return False.
    token_path = tmp_home / ".install_token"
    token_path.write_text("", encoding="utf-8")
    assert sp.verify_install_token("anything") is False


def test_ensure_install_token_regenerates_empty_file(tmp_home: Path) -> None:
    token_path = tmp_home / ".install_token"
    token_path.write_text("", encoding="utf-8")
    token = sp.ensure_install_token()
    assert len(token) >= 32
    assert token_path.read_text(encoding="utf-8").strip() == token


# ---------------------------------------------------------------------------
# run_subprocess_safe
# ---------------------------------------------------------------------------


def test_run_subprocess_refuses_string_argv() -> None:
    with pytest.raises(TypeError):
        sp.run_subprocess_safe("echo hi")  # type: ignore[arg-type]


def test_run_subprocess_runs_list_argv() -> None:
    result = sp.run_subprocess_safe([sys.executable, "-c", "print('ok')"], timeout=5.0)
    assert result.returncode == 0
    assert b"ok" in result.stdout


def test_run_subprocess_enforces_timeout() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        sp.run_subprocess_safe(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.2,
        )


def test_run_subprocess_rejects_empty_argv() -> None:
    with pytest.raises(ValueError):
        sp.run_subprocess_safe([])


def test_run_subprocess_rejects_non_string_entries() -> None:
    with pytest.raises(TypeError):
        sp.run_subprocess_safe([sys.executable, 123])  # type: ignore[list-item]


def test_run_subprocess_uses_restricted_env() -> None:
    # Custom env passed through
    result = sp.run_subprocess_safe(
        [sys.executable, "-c", "import os; print(os.environ.get('SLM_TEST','missing'))"],
        timeout=5.0,
        env={"SLM_TEST": "present"},
    )
    assert b"present" in result.stdout


def test_run_subprocess_default_env_has_minimal_keys() -> None:
    # When env not specified, default should still let the subprocess run.
    result = sp.run_subprocess_safe(
        [sys.executable, "-c", "print('alive')"],
        timeout=5.0,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# safe_resolve_identifier — LLD-00 §4 contract (P0.2)
# ---------------------------------------------------------------------------


def test_safe_resolve_identifier_accepts_valid_id(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    resolved = sp.safe_resolve_identifier(base, "sess_abc-123")
    assert resolved == (base / "sess_abc-123").resolve()


def test_safe_resolve_identifier_rejects_dot_dot(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, "..")


def test_safe_resolve_identifier_rejects_slash(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, "sub/child")


def test_safe_resolve_identifier_rejects_backslash(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, r"sub\child")


def test_safe_resolve_identifier_rejects_null_byte(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, "foo\x00bar")


def test_safe_resolve_identifier_rejects_empty(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, "")


def test_safe_resolve_identifier_rejects_too_long(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, "a" * 129)


def test_safe_resolve_identifier_accepts_max_length(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    long_id = "a" * 128
    resolved = sp.safe_resolve_identifier(base, long_id)
    assert resolved.name == long_id


def test_safe_resolve_identifier_rejects_non_string(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, 123)  # type: ignore[arg-type]


def test_safe_resolve_identifier_rejects_unicode_non_ascii(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, "café")


def test_safe_resolve_identifier_rejects_spaces(tmp_path: Path) -> None:
    base = tmp_path / "sessions"
    base.mkdir()
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, "a b")


def test_safe_resolve_identifier_rejects_path_escape_via_symlink(
    tmp_path: Path,
) -> None:
    if sys.platform == "win32":
        pytest.skip("symlink support inconsistent on Windows CI")
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    # Replace 'base' with a symlink that escapes — the identifier alone is
    # valid, but resolution must still confirm containment.
    escape_id = "escape_link"
    (base / escape_id).symlink_to(outside)
    with pytest.raises(ValueError):
        sp.safe_resolve_identifier(base, escape_id)


# ---------------------------------------------------------------------------
# redact_secrets aggression='high' — LLD-00 §5 contract (P0.3)
# ---------------------------------------------------------------------------


def test_redact_secrets_high_jwt() -> None:
    # A realistic 3-segment JWT — must redact.
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    redacted = sp.redact_secrets(f"Authorization: {jwt}", aggression="high")
    assert jwt not in redacted
    assert "[REDACTED:JWT:" in redacted


def test_redact_secrets_high_bearer() -> None:
    text = "Authorization: Bearer abcdef0123456789ABCDEF.signed-value=="
    redacted = sp.redact_secrets(text, aggression="high")
    assert "abcdef0123456789ABCDEF.signed-value==" not in redacted
    assert "[REDACTED:BEARER:" in redacted


def test_redact_secrets_high_github_pat_classic() -> None:
    token = "ghp_" + "A" * 36
    redacted = sp.redact_secrets(f"token={token}", aggression="high")
    assert token not in redacted
    assert "[REDACTED:GITHUB_PAT:" in redacted


def test_redact_secrets_high_github_pat_oauth() -> None:
    token = "gho_" + "B" * 36
    redacted = sp.redact_secrets(f"oauth={token}", aggression="high")
    assert token not in redacted
    assert "[REDACTED:GITHUB_PAT:" in redacted


def test_redact_secrets_high_github_pat_server() -> None:
    token = "ghs_" + "C" * 36
    redacted = sp.redact_secrets(f"server={token}", aggression="high")
    assert token not in redacted
    assert "[REDACTED:GITHUB_PAT:" in redacted


def test_redact_secrets_high_anthropic_key() -> None:
    key = "sk-ant-api03-" + "D" * 55
    redacted = sp.redact_secrets(f"ANTHROPIC_API_KEY={key}", aggression="high")
    assert key not in redacted
    assert "[REDACTED:ANTHROPIC_KEY:" in redacted


def test_redact_secrets_high_anthropic_admin_key() -> None:
    key = "sk-ant-admin01-" + "E" * 55
    redacted = sp.redact_secrets(f"KEY={key}", aggression="high")
    assert key not in redacted
    assert "[REDACTED:ANTHROPIC_KEY:" in redacted


def test_redact_secrets_high_openai_key() -> None:
    key = "sk-" + "F" * 20 + "T3BlbkFJ" + "G" * 20
    redacted = sp.redact_secrets(f"OPENAI_API_KEY={key}", aggression="high")
    assert key not in redacted
    assert "[REDACTED:OPENAI_KEY:" in redacted


def test_redact_secrets_high_generic_uppercase_key() -> None:
    # [A-Z]{2,5}_[A-Z0-9]{20,} pattern — typical env-var secret shape.
    key = "SLM_" + "A1B2C3D4E5F6G7H8I9J0"
    redacted = sp.redact_secrets(f"config={key}", aggression="high")
    assert key not in redacted
    assert "[REDACTED:GENERIC_KEY:" in redacted


def test_redact_secrets_high_preserves_non_secret_text() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    assert sp.redact_secrets(text, aggression="high") == text


def test_redact_secrets_high_spares_pure_letter_constant() -> None:
    """H-08: long UPPER_SNAKE_CASE constants without digits must NOT trigger
    GENERIC_KEY redaction — they are overwhelmingly variable/field names, not
    real secrets. Real high-entropy keys almost always carry digits."""
    # No digits in the 20+ char tail → must be preserved.
    constant = "SLM_PASSWORDABCDEFGHIJKLMNOPQRST"
    redacted = sp.redact_secrets(f"const={constant}", aggression="high")
    assert constant in redacted
    assert "[REDACTED:GENERIC_KEY:" not in redacted


def test_redact_secrets_high_catches_digit_bearing_generic_key() -> None:
    """H-08: the tightened pattern still catches realistic env-var secrets
    whose 20+ char tail contains at least one digit."""
    key = "SLM_" + "ABCDEFGHIJKLMNOPQRS1"  # 20-char tail with trailing digit
    redacted = sp.redact_secrets(f"config={key}", aggression="high")
    assert key not in redacted
    assert "[REDACTED:GENERIC_KEY:" in redacted


def test_redact_secrets_high_spares_enum_style_names() -> None:
    """H-08: enum-style identifiers that happen to be long remain safe."""
    # 25 char letters-only tail — classic enum/field name.
    name = "CFG_READY_FOR_SHIPMENT_ENUM"
    redacted = sp.redact_secrets(f"state={name}", aggression="high")
    assert name in redacted


def test_redact_secrets_normal_unchanged_behavior() -> None:
    # Regression: default behavior still works with NO aggression kwarg.
    text = "my key is sk-" + "A" * 40 + " end"
    default_redacted = sp.redact_secrets(text)
    explicit_normal = sp.redact_secrets(text, aggression="normal")
    assert default_redacted == explicit_normal
    assert "[REDACTED:" in default_redacted


def test_redact_secrets_high_rejects_invalid_aggression() -> None:
    with pytest.raises(ValueError):
        sp.redact_secrets("x", aggression="ultra")  # type: ignore[arg-type]


def test_redact_secrets_high_masks_show_last4() -> None:
    """LLD-00 §5 mandates [REDACTED:TYPE:last4] format."""
    token = "ghp_" + "A" * 32 + "ABCD"
    redacted = sp.redact_secrets(f"t={token}", aggression="high")
    assert "[REDACTED:GITHUB_PAT:ABCD]" in redacted


def test_redact_secrets_high_multiple_secrets_in_text() -> None:
    """Multiple different secrets → all redacted independently."""
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123def456ghi789"
    gh = "ghp_" + "A" * 36
    text = f"JWT: {jwt}\nGH: {gh}"
    redacted = sp.redact_secrets(text, aggression="high")
    assert jwt not in redacted
    assert gh not in redacted
    assert "[REDACTED:JWT:" in redacted
    assert "[REDACTED:GITHUB_PAT:" in redacted
