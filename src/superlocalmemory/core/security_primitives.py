# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §6

"""Shared security primitives for SLM v3.4.22.

LLD reference: `.backup/active-brain/lld/LLD-07-schema-migrations-and-security-primitives.md`
Section: 6.1 through 6.10.

Every file write, subprocess spawn, and secret-bearing string across SLM
daemon, adapters, hooks, and binary installer routes through this module.
Single source of truth — the hard rules in LLD-07 §7 are enforced here.

All functions are defensive: they raise early, log nothing about the secret
content, and use constant-time comparisons where applicable.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import secrets as _secrets
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PathTraversalError(ValueError):
    """Raised by safe_resolve when a path escapes its allowed base."""


class IntegrityError(ValueError):
    """Raised when a SHA-256 integrity check fails."""


# ---------------------------------------------------------------------------
# 6.1 Safe path resolver (SEC-01-05, SEC-05-01, SEC-06-03)
# ---------------------------------------------------------------------------


_DENY_PREFIXES_POSIX: tuple[str, ...] = (
    "/etc",
    "/usr",
    "/var",
    "/sys",
    "/proc",
    "/bin",
    "/sbin",
    "/System",
    "/Library",
)
_DENY_PREFIXES_WINDOWS: tuple[str, ...] = (
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\ProgramData",
)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _hits_deny_prefix(resolved: Path) -> bool:
    resolved_str = str(resolved)
    if _is_windows():  # pragma: no cover — Windows-only branch
        lower = resolved_str.lower()
        return any(lower.startswith(p.lower()) for p in _DENY_PREFIXES_WINDOWS)
    return any(resolved_str == p or resolved_str.startswith(p + os.sep)
               for p in _DENY_PREFIXES_POSIX)


def safe_resolve(base: Path, rel: str | Path) -> Path:
    """Resolve ``rel`` against ``base`` safely.

    Rules:
      - ``rel`` must be str or Path.
      - ``..`` components are refused outright.
      - Resolved absolute path must be a descendant of ``base.resolve()``.
      - Resolved path must not land in a reserved system prefix.
      - Any symlink in the chain is re-validated: its target must also live
        under ``base``.

    Returns the resolved absolute Path on success; raises PathTraversalError
    otherwise.
    """
    if not isinstance(rel, (str, Path)):
        raise TypeError(f"rel must be str | Path, got {type(rel).__name__}")

    rel_path = Path(rel)
    if ".." in rel_path.parts:
        raise PathTraversalError(f"'..' components are forbidden: {rel!r}")

    if rel_path.is_absolute():
        candidate = rel_path
    else:
        candidate = base / rel_path

    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:  # pragma: no cover — defensive
        raise PathTraversalError(f"cannot resolve {rel!r}: {exc}") from exc

    if _hits_deny_prefix(resolved):
        raise PathTraversalError(f"denied system prefix: {resolved}")

    try:
        base_resolved = base.resolve(strict=False)
    except (OSError, RuntimeError) as exc:  # pragma: no cover — defensive
        raise PathTraversalError(f"cannot resolve base {base!r}: {exc}") from exc

    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise PathTraversalError(
            f"{resolved} escapes base {base_resolved}"
        ) from exc

    # Symlink walk — defense in depth against TOCTOU on a symlink parent.
    # The ``resolved.relative_to(base)`` check above catches the common case;
    # this loop walks the pre-resolution chain so we refuse when any
    # intermediate component is a symlink whose target escapes the base.
    cur = candidate
    while cur != cur.parent:
        if cur.exists() and cur.is_symlink():
            try:
                target = cur.resolve(strict=False)
                target.relative_to(base_resolved)
            except (ValueError, OSError) as exc:  # pragma: no cover — TOCTOU
                raise PathTraversalError(
                    f"symlink {cur} points outside base"
                ) from exc
        cur = cur.parent

    return resolved


# ---------------------------------------------------------------------------
# LLD-00 §4 — safe_resolve_identifier (SEC-C-02 fix)
# ---------------------------------------------------------------------------
#
# The pre-existing ``safe_resolve`` above handles hardcoded relative paths
# (e.g. `.cursor/rules/file.mdc`) against a trusted base. LLD-00 §4 adds a
# stricter contract for *untrusted identifiers* — a ``session_id`` or
# ``profile_id`` that may reach the filesystem via path join. This helper
# enforces the LLD-00 regex AND the base-containment check. Callers in
# LLD-09 (session state files) and LLD-11 (evolution.lock) MUST use this.
#
# Naming deviation from IMPLEMENTATION-MANIFEST P0.2: the manifest reused
# the name ``safe_resolve`` but the existing path-style helper is used in
# 9+ call sites. A separate name avoids breakage. See
# ``.backup/active-brain/MANIFEST-DEVIATION.md`` P0.2 entry.

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def safe_resolve_identifier(base: Path, untrusted: str) -> Path:
    """Return ``base / untrusted`` only if ``untrusted`` is a safe identifier
    AND the resolved path stays within ``base``. Raises ``ValueError`` otherwise.

    Rejects: '..', '/', '\\', null bytes, empty strings, strings longer than
    128 chars, and anything outside ``[a-zA-Z0-9_-]``.

    Used for untrusted filesystem identifiers (``session_id``, ``profile_id``)
    — NOT for hardcoded template paths (use :func:`safe_resolve` for those).
    """
    if not isinstance(untrusted, str):
        raise ValueError(
            f"unsafe identifier: expected str, got {type(untrusted).__name__}"
        )
    if not _SAFE_ID_RE.match(untrusted):
        raise ValueError(f"unsafe identifier: {untrusted!r}")

    base_abs = base.resolve(strict=False)
    target = (base / untrusted).resolve(strict=False)
    # The resolved target must be a direct child of base (or equal to it —
    # defensive, though the regex already forbids the empty case).
    if target != base_abs and base_abs not in target.parents:
        raise ValueError(f"path escape: {untrusted!r}")
    # S9-W2 M-SEC-01: enforce byte-level name equality after resolve.
    # On case-insensitive filesystems (macOS APFS, Windows NTFS) the
    # untrusted id "Session_1" can collide with an existing "session_1"
    # path and ``.resolve()`` returns the on-disk name. Allowing that
    # equivalence would let a second user on the same macOS machine
    # enumerate / overwrite another user's session state by guessing
    # the case-folded identifier.
    if target != base_abs and target.name != untrusted:
        raise ValueError(
            f"path-case collision: resolved {target.name!r} != "
            f"requested {untrusted!r}"
        )
    return target


# ---------------------------------------------------------------------------
# 6.10 SHA-256 integrity verifier (SEC-06-01)
# ---------------------------------------------------------------------------


def verify_sha256(data: bytes, expected_hex: str) -> None:
    """Verify ``hashlib.sha256(data).hexdigest() == expected_hex``.

    Uses ``hmac.compare_digest`` for constant-time comparison.
    Raises IntegrityError on any mismatch.

    Accepts expected_hex in either case (SHA-256 hex is case-insensitive).
    """
    if not isinstance(expected_hex, str):
        raise IntegrityError("expected_hex must be str")
    if len(expected_hex) != 64:
        raise IntegrityError(
            f"expected_hex must be 64 chars, got {len(expected_hex)}"
        )
    actual = hashlib.sha256(data).hexdigest()
    if not hmac.compare_digest(actual.lower(), expected_hex.lower()):
        raise IntegrityError("SHA-256 mismatch")


# ---------------------------------------------------------------------------
# 6.3 Secret redaction (SEC-02-01, SEC-01-03)
# ---------------------------------------------------------------------------


_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "ANTHROPIC"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "OPENAI"),
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "GITHUB"),
    (re.compile(r"AKIA[A-Z0-9]{16}"), "AWS"),
    (re.compile(r"xoxb-[A-Za-z0-9\-]{10,}"), "SLACK"),
    (re.compile(r"ey[A-Za-z0-9_\-]{10,}\.ey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),
     "JWT"),
    (re.compile(r"-----BEGIN [A-Z ]+-----"), "PRIVATE_KEY"),
)

# LLD-00 §5 high-aggression patterns (P0.3). Stricter than the defaults:
# they match concrete, well-known secret shapes only, and are tried FIRST
# so their specific labels win over the broader 'OPENAI'/'ANTHROPIC' fallbacks.
_HIGH_AGGRESSION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Generic JWT — three dot-separated base64url segments starting with "eyJ".
    (re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ), "JWT"),
    # Bearer header — catches `Authorization: Bearer ...` style tokens.
    (re.compile(r"\bBearer\s+[A-Za-z0-9_\-.=]{20,}"), "BEARER"),
    # GitHub PATs: classic + OAuth + server-to-server.
    (re.compile(
        r"\bghp_[A-Za-z0-9]{36}\b|\bgho_[A-Za-z0-9]{36}\b|\bghs_[A-Za-z0-9]{36}\b"
    ), "GITHUB_PAT"),
    # Anthropic API/admin keys — current format as of 2026.
    (re.compile(r"\bsk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_-]{50,}\b"),
     "ANTHROPIC_KEY"),
    # OpenAI modern keys — carry the "T3BlbkFJ" ("OpenAI" base64) sentinel.
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}\b"),
     "OPENAI_KEY"),
    # Generic env-var-style secret (e.g. SLM_API_ABC123...).
    #
    # H-08 (Stage 8): must skip pure-letter UPPER_SNAKE_CASE constants.
    # S9-W2 H-SEC-02: the Stage-8 lookahead ``(?=[A-Z0-9]*\d)`` was
    # linear-in-text but triggered super-linear BACKTRACKING on crafted
    # inputs like ``"A" * 256_000`` because every ``[A-Z]{2,5}`` prefix
    # attempt re-ran the tail lookahead. Under ``redact_secrets`` with
    # an attacker-controlled prompt this stalled the dispatcher for
    # seconds — a reachable DoS. New design: match the broader shape
    # without a lookahead, then check ``"any(ch.isdigit() for ch in m)"``
    # in Python after the match. Python's post-match loop is O(n) with
    # no backtracking regardless of input shape.
    (re.compile(r"\b[A-Z]{2,5}_[A-Z0-9]{20,}\b"), "GENERIC_KEY"),
)

_VALID_AGGRESSION = frozenset({"normal", "high"})


def _shannon_entropy(s: str) -> float:
    if not s:  # pragma: no cover — callers guard
        return 0.0
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    total = len(s)
    entropy = 0.0
    for n in counts.values():
        p = n / total
        entropy -= p * math.log2(p)
    return entropy


def redact_secrets(text: str, *, entropy_threshold: float = 4.5,
                   window: int = 32,
                   aggression: str = "normal") -> str:
    """Replace detected secrets with ``[REDACTED:TYPE:last4]`` markers.

    Three-layer defense:
      1. High-aggression patterns (JWT/Bearer/GitHub PAT/Anthropic/OpenAI/
         GENERIC_KEY) — applied first when ``aggression='high'`` (LLD-00 §5).
         These have concrete, well-known shapes so labels are specific.
      2. Pattern-based fallback (OpenAI/Anthropic/GitHub/AWS/Slack/JWT/PEM).
      3. Entropy-based sweep — any 32+ char contiguous high-entropy run
         of URL-safe characters that survived pattern scan gets redacted as
         ``[REDACTED:ENTROPY:last4]``.

    ``aggression='high'`` is mandatory for every LLM-bound prompt
    (LLD-11 evolution dispatch). Rationale: LLM providers may log or
    retain prompts, so any leaked secret is a breach. LLD-07 §6.3 rule:
    every string entering cache or dashboard goes through this helper.
    """
    if aggression not in _VALID_AGGRESSION:
        raise ValueError(
            f"aggression must be one of {sorted(_VALID_AGGRESSION)}, "
            f"got {aggression!r}"
        )
    if not isinstance(text, str):
        return text  # pragma: no cover — defensive
    if not text:
        return text

    out = text

    if aggression == "high":
        for pat, label in _HIGH_AGGRESSION_PATTERNS:
            def _sub_high(match: re.Match[str], _label: str = label) -> str:
                matched = match.group(0)
                # S9-W2 H-SEC-02: GENERIC_KEY now requires a post-match
                # digit check (replaces the lookahead that caused
                # super-linear backtracking on crafted input). Pure
                # UPPER_SNAKE constants still pass through unredacted.
                if _label == "GENERIC_KEY" and not any(
                    ch.isdigit() for ch in matched
                ):
                    return matched
                last4 = matched[-4:] if len(matched) >= 4 else matched
                return f"[REDACTED:{_label}:{last4}]"
            out = pat.sub(_sub_high, out)

    for pat, label in _SECRET_PATTERNS:
        def _sub(match: re.Match[str], _label: str = label) -> str:
            matched = match.group(0)
            last4 = matched[-4:] if len(matched) >= 4 else matched
            return f"[REDACTED:{_label}:{last4}]"
        out = pat.sub(_sub, out)

    # S9-W2 L-SEC-01: skip strings that already look like a REDACTED
    # marker so the entropy sweep doesn't double-redact and lose the
    # provenance label (e.g. turning ``[REDACTED:GITHUB_PAT:deadbeef]``
    # into ``[REDACTED:ENTROPY:eef]``). Matching is conservative — the
    # regex below is the one ``_emit_marker`` emits.
    _redacted_marker_re = re.compile(r"\[REDACTED:[A-Z_]+:[^\]]+\]")

    # Entropy sweep — scan contiguous URL-safe runs.
    #
    # H-08 (Stage 8): real high-entropy secrets (API keys, hex tokens,
    # base64 blobs) almost always carry at least one digit or lowercase
    # letter. Pure ``UPPER_SNAKE_CASE`` constants clear the 4.5 entropy
    # threshold too — so without this guard the entropy sweep misfires
    # on long variable/field names. We skip tokens whose character set
    # is a subset of ``[A-Z_]``.
    #
    # ``=`` is deliberately excluded from the token class so a
    # ``key=VALUE`` pair splits on the sign; otherwise a short ``key=``
    # prefix bridges into the value and bypasses the pure-upper-snake
    # check. Base64 padding is at most ``==`` — losing those two
    # characters at the tail does not hide a secret.
    token_re = re.compile(r"[A-Za-z0-9_\-./+]{%d,}" % window)
    _pure_upper_snake = re.compile(r"^[A-Z_]+$")

    def _entropy_sub(match: re.Match[str]) -> str:
        token = match.group(0)
        # L-SEC-01: preserve REDACTED markers emitted by earlier passes.
        if _redacted_marker_re.search(token):
            return token
        # S9-SKEP-12: pure UPPER_SNAKE is a legitimate-constant shape
        # ONLY when its entropy is below the secret threshold. A 24-char
        # all-caps mnemonic backup code or a hand-typed token does clear
        # 4.5 bits Shannon entropy and should be redacted — the old
        # unconditional skip let such secrets through. We now require
        # BOTH "looks like a constant" AND "low entropy" before skipping.
        entropy = _shannon_entropy(token)
        if _pure_upper_snake.match(token) and entropy < entropy_threshold:
            return token
        if entropy >= entropy_threshold:
            last4 = token[-4:]
            return f"[REDACTED:ENTROPY:{last4}]"
        return token

    out = token_re.sub(_entropy_sub, out)
    return out


# ---------------------------------------------------------------------------
# 6.6 Install-token generation + verification (SEC-01-02, SEC-06-03)
# ---------------------------------------------------------------------------


def _install_token_path() -> Path:  # pragma: no cover — monkeypatched in tests
    """Default install-token location — override in tests via monkeypatch."""
    return Path.home() / ".superlocalmemory" / ".install_token"


def ensure_install_token() -> str:
    """Create or read the install token at ``~/.superlocalmemory/.install_token``.

    On first call, creates the file with 32 bytes of ``secrets.token_hex``
    and sets mode 0600 on POSIX. On subsequent calls, returns the existing
    token unchanged.

    The token is used as:
      - ``X-SLM-Hook-Token`` header for ``/internal/prewarm`` auth.
      - Cache-install binding via ``slm_meta`` row.
    """
    token_path = _install_token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)

    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
        # Empty file — regenerate.

    # S9-W2 H-SEC-01: close the docstring promise "Open with O_EXCL
    # where possible to prevent races" that the implementation did NOT
    # enforce. O_EXCL | O_CREAT atomically fails if the file exists,
    # which means a second concurrent daemon hitting this path after
    # the first one wrote the token sees EEXIST, re-reads, and returns
    # the token from disk — both daemons converge on the same token.
    # Fallback to the non-EXCL path is preserved for exotic FS that
    # don't support the flag, but the common POSIX case now closes
    # the race.
    token = _secrets.token_hex(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    wrote = False
    try:
        fd = os.open(str(token_path), flags, 0o600)
        try:
            os.write(fd, token.encode("utf-8"))
            wrote = True
        finally:
            os.close(fd)
    except FileExistsError:
        # Someone else won the race. Re-read and return their token.
        try:
            existing = token_path.read_text(encoding="utf-8").strip()
        except OSError:  # pragma: no cover — defensive
            existing = ""
        if existing:
            return existing
        # Empty file left by the racer — overwrite via the non-EXCL
        # path below so we still end up with a valid token.
        try:
            fd = os.open(
                str(token_path),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | (
                    os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
                ),
                0o600,
            )
            try:
                os.write(fd, token.encode("utf-8"))
                wrote = True
            finally:
                os.close(fd)
        except OSError:  # pragma: no cover — fallback for exotic FS
            token_path.write_text(token, encoding="utf-8")
            wrote = True
    except OSError:  # pragma: no cover — fallback for exotic FS
        token_path.write_text(token, encoding="utf-8")
        wrote = True

    if wrote and not _is_windows():
        try:
            os.chmod(token_path, 0o600)
        except OSError:  # pragma: no cover
            pass

    return token


def verify_install_token(presented: str) -> bool:
    """Constant-time compare ``presented`` against the stored install token.

    Returns False (never raises) on missing file, empty input, or mismatch.
    """
    if not isinstance(presented, str) or not presented:
        return False
    token_path = _install_token_path()
    if not token_path.exists():
        return False
    try:
        stored = token_path.read_text(encoding="utf-8").strip()
    except OSError:  # pragma: no cover
        return False
    if not stored:
        return False
    return hmac.compare_digest(stored, presented)


def rotate_install_token() -> tuple[str, str]:
    """S-M07 — atomically rotate the install token.

    Returns ``(old_token, new_token)``. The old token is captured BEFORE
    the rotation so callers that need to invalidate cached HMAC markers
    can detect the change. Atomic file-swap via ``os.replace`` so a
    concurrent ``verify_install_token`` never observes a half-written
    value.

    Callers (e.g. ``slm rotate-token`` CLI) SHOULD restart the daemon
    after a successful rotation: in-memory HMAC marker caches — used by
    ``recall_pipeline._emit_marker`` — retain the old token until the
    next cold start. Without a restart, already-emitted markers fail
    validation on the next ``post_tool_outcome_hook`` call (harmless —
    just a dropped signal, never a security bypass), but new markers
    under the new token mix with old-token markers still in transit.

    Never raises; returns ``("", "")`` on any filesystem error so the
    caller can surface a graceful message to the user.
    """
    token_path = _install_token_path()
    try:
        token_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover — defensive
        return ("", "")

    old = ""
    if token_path.exists():
        try:
            old = token_path.read_text(encoding="utf-8").strip()
        except OSError:  # pragma: no cover
            old = ""

    new_token = _secrets.token_hex(32)
    # Write via tmp + os.replace for atomic swap.
    tmp = token_path.with_suffix(
        token_path.suffix + f".rot.{os.getpid()}.tmp"
    )
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(tmp), flags, 0o600)
        try:
            os.write(fd, new_token.encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(str(tmp), str(token_path))
    except OSError:  # pragma: no cover — fallback
        try:
            token_path.write_text(new_token, encoding="utf-8")
        except OSError:
            return (old, "")
    if not _is_windows():
        try:
            os.chmod(token_path, 0o600)
        except OSError:  # pragma: no cover
            pass
    return (old, new_token)


# ---------------------------------------------------------------------------
# 6.9 Subprocess sanitizer (SEC-05-01)
# ---------------------------------------------------------------------------


_DEFAULT_SAFE_ENV_KEYS: tuple[str, ...] = (
    "PATH", "HOME", "USER", "LANG", "LC_ALL",
    "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE",  # Windows
)


def _default_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _DEFAULT_SAFE_ENV_KEYS if k in os.environ}


def run_subprocess_safe(
    argv: list[str],
    *,
    timeout: float = 5.0,
    env: dict[str, str] | None = None,
    check: bool = False,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Safe wrapper around ``subprocess.run``.

    Rules enforced:
      - ``argv`` must be a list of strings (never a shell string).
      - ``shell=False`` always.
      - ``timeout`` is mandatory.
      - Restricted environment by default — only a minimal set of safe keys.
      - Callers may pass an explicit ``env`` to add specific variables.

    This is the ONE place in the codebase allowed to call ``subprocess.run``.
    Grep guard in CI enforces this (LLD-07 §7 SEC-HR-06).
    """
    if not isinstance(argv, list):
        raise TypeError("argv must be list[str], shell=False only")
    if not argv:
        raise ValueError("argv must be non-empty")
    for i, piece in enumerate(argv):
        if not isinstance(piece, str):
            raise TypeError(f"argv[{i}] must be str, got {type(piece).__name__}")

    effective_env = _default_env()
    if env is not None:
        effective_env.update(env)

    # NOTE: This is the sanctioned subprocess.run call site for SLM.
    return subprocess.run(  # noqa: S603
        argv,
        shell=False,
        timeout=timeout,
        check=check,
        capture_output=capture_output,
        env=effective_env,
    )


__all__ = (
    "PathTraversalError",
    "IntegrityError",
    "safe_resolve",
    "safe_resolve_identifier",
    "verify_sha256",
    "redact_secrets",
    "ensure_install_token",
    "verify_install_token",
    "run_subprocess_safe",
)
