# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Named, scoped, revocable keys for AI tools on other computers.

Each tool that reaches this SuperLocalMemory from another computer gets its own
key (``slm remote keys add <name> [--profile p]``). A key is ``read`` (recall
only) or ``write`` (recall and save), and it is bound to exactly one profile:
the one named with ``--profile``, else the profile that was active when the key
was made. A key never reaches another profile (see
:mod:`server.remote_profile_binding`). Revoking one key cuts off one tool on its
next request, with no restart and without touching any other tool.

Keys made before 4.1.20 had no profile. They are bound, once, to the profile
active when this release first runs (the daemon at start, or any
``slm remote keys`` command), and ``slm remote keys list`` says so. Until then a
key without a profile is refused.

Only a domain-separated SHA-256 digest of each key is stored, in
``<data root>/remote_keys.json`` (mode 0600). The secret is shown once, when it
is created. A key store that another user owns, or that other users can write,
is ignored entirely (every remote key is refused) - a tampered store must never
grant access.

Records are frozen; revoking writes a new record and keeps the old name so a
revoked name is never silently reused.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

KEY_PREFIX = "slmr_"
#: ``secrets.token_urlsafe(32)`` is 43 characters.
_SECRET_BODY_LEN = 43
_KEY_LEN = len(KEY_PREFIX) + _SECRET_BODY_LEN
_DOMAIN = b"superlocalmemory-remote-key-v1\0"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
#: Version 2 added the profile binding. Version 1 stores are still read (their
#: keys have no profile until bound); every write is version 2, which an older
#: release refuses entirely rather than serving without the binding.
_STORE_VERSION = 2
_READABLE_VERSIONS = (1, 2)
STORE_FILE = "remote_keys.json"
#: Profile ids as ``slm profile create`` accepts them.
_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: How a key got its profile: named with --profile, the active profile when the
#: key was made, or the active profile when a pre-4.1.20 key was upgraded.
PROFILE_SOURCES: tuple[str, ...] = ("chosen", "active-at-creation", "bound-on-upgrade")

#: Capabilities a key can be opted in to, beyond recall and save. A web app
#: only gets one when its gateway consent AND this key both allow it.
EXTRAS: tuple[str, ...] = ("mesh", "media")

Scope = Literal["read", "write"]
SCOPES: tuple[str, ...] = ("read", "write")


class RemoteKeyError(ValueError):
    """A key operation was refused. ``code`` is stable for scripts and tests."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RemoteKey:
    key_id: str
    name: str
    scope: Scope
    digest: str
    created_at: str
    revoked_at: str | None = None
    #: The one profile this key reaches. ``None`` only for a key made before
    #: 4.1.20 that has not been bound yet; such a key is refused.
    profile: str | None = None
    profile_source: str | None = None
    #: Opt-ins from :data:`EXTRAS`. Optional in the file; 4.1.24 ignores the field.
    extras: frozenset[str] = frozenset()

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def public(self) -> dict[str, object]:
        """Everything except the digest - safe to print."""
        return {"name": self.name, "key_id": self.key_id, "scope": self.scope,
                "profile": self.profile, "profile_source": self.profile_source,
                "created_at": self.created_at, "revoked_at": self.revoked_at,
                "extras": sorted(self.extras)}


def valid_profile_id(profile: object) -> bool:
    return isinstance(profile, str) and bool(_PROFILE_RE.match(profile))


def digest_secret(secret: str) -> str:
    return hashlib.sha256(_DOMAIN + secret.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_path() -> Path:
    from superlocalmemory.infra.data_root import state_path

    return state_path(STORE_FILE)


def store_problem(path: Path) -> str | None:
    """Why this key store must not be trusted, or ``None`` when it is fine."""
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"cannot read {path.name}: {exc}"
    if os.name != "posix":
        return None
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        return f"{path.name} is owned by another user"
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return f"{path.name} can be changed by other users"
    if info.st_mode & (stat.S_IRGRP | stat.S_IROTH):
        return f"{path.name} can be read by other users"
    return None


def _extras_from(raw: object) -> frozenset[str]:
    """Known opt-ins only; anything else (including a malformed field) is dropped."""
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(x for x in raw if isinstance(x, str) and x in EXTRAS)


def _record_dict(record: RemoteKey) -> dict:
    data = asdict(record)
    if record.extras:
        data["extras"] = sorted(record.extras)
    else:
        del data["extras"]
    return data


def _record_from(raw: object) -> RemoteKey | None:
    if not isinstance(raw, dict):
        return None
    try:
        record = RemoteKey(
            key_id=str(raw["key_id"]), name=str(raw["name"]), scope=raw["scope"],
            digest=str(raw["digest"]), created_at=str(raw["created_at"]),
            revoked_at=(None if raw.get("revoked_at") is None else str(raw["revoked_at"])),
            profile=raw.get("profile"), profile_source=raw.get("profile_source"),
            extras=_extras_from(raw.get("extras")),
        )
    except (KeyError, TypeError):
        return None
    if record.scope not in SCOPES or len(record.digest) != 64:
        return None
    if record.profile is not None and (not valid_profile_id(record.profile)
                                       or record.profile_source not in PROFILE_SOURCES):
        # A malformed binding must never widen what the key reaches: drop the
        # record (the key is refused) rather than treat it as unbound.
        return None
    return record


class RemoteKeyStore:
    """The key file. Reloads on change, so revocation needs no restart."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: tuple[tuple[str, int, int, int], tuple[RemoteKey, ...]] | None = None
        self._warned: str | None = None

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else _default_path()

    # -- reading ---------------------------------------------------------------

    def _load(self) -> tuple[RemoteKey, ...]:
        path = self.path
        problem = store_problem(path)
        if problem is not None:
            if self._warned != problem:
                logger.critical("Remote keys are disabled: %s. Run 'slm remote check'.",
                                problem)
                self._warned = problem
            return ()
        try:
            info = path.stat()
        except FileNotFoundError:
            return ()
        except OSError:
            return ()
        signature = (str(path), info.st_mtime_ns, info.st_size, info.st_ino)
        with self._cache_lock:
            if self._cache is not None and self._cache[0] == signature:
                return self._cache[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.critical("Remote keys are disabled: %s is unreadable (%s).", path.name,
                            type(exc).__name__)
            return ()
        if not isinstance(data, dict) or data.get("version") not in _READABLE_VERSIONS:
            logger.critical("Remote keys are disabled: %s has an unknown format.", path.name)
            return ()
        records = tuple(r for r in (_record_from(x) for x in data.get("keys") or ()) if r)
        with self._cache_lock:
            self._cache = (signature, records)
        return records

    def list(self) -> tuple[RemoteKey, ...]:
        return self._load()

    def verify(self, presented: str) -> RemoteKey | None:
        """The active key matching ``presented``, or ``None``.

        Every record is compared (no early exit), each with
        :func:`hmac.compare_digest`.
        """
        if (not isinstance(presented, str) or len(presented) != _KEY_LEN
                or not presented.startswith(KEY_PREFIX)):
            return None
        candidate = digest_secret(presented)
        match: RemoteKey | None = None
        for record in self._load():
            if hmac.compare_digest(candidate, record.digest) and match is None:
                match = record
        if match is None or not match.active:
            return None
        return match

    # -- writing ---------------------------------------------------------------

    @contextlib.contextmanager
    def _exclusive(self):
        """One writer at a time, across processes (the CLI and the daemon).

        Every change is read-modify-write; without this a revoke racing the
        daemon's one-time upgrade could be overwritten and the key revived.
        """
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows: in-process lock only
                yield
                return
            fd = os.open(path.with_name(f".{path.name}.lock"), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _write(self, records: tuple[RemoteKey, ...]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": _STORE_VERSION,
                              "keys": [_record_dict(r) for r in records]}, indent=2) + "\n"
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            # 0600 means nothing on Windows; owner-only there too, before any byte.
            from superlocalmemory.infra.owner_only_acl import restrict_to_owner

            restrict_to_owner(tmp)
        except BaseException:
            os.close(fd)
            tmp.unlink(missing_ok=True)
            raise
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

    def _records_for_write(self) -> tuple[RemoteKey, ...]:
        problem = store_problem(self.path)
        if problem is not None:
            raise RemoteKeyError("store_untrusted",
                                 f"Refusing to change remote keys: {problem}.")
        return self._load()

    def add(self, name: str, scope: Scope, *, profile: str,
            profile_source: str = "chosen") -> tuple[RemoteKey, str]:
        """Create a key bound to ``profile``. Returns the record and the secret
        (shown once). The caller checks that the profile exists."""
        if not valid_profile_id(profile):
            raise RemoteKeyError("invalid_profile",
                                 "A remote key needs the profile it may reach "
                                 "(letters, digits, '_' or '-').")
        if profile_source not in PROFILE_SOURCES[:2]:
            raise RemoteKeyError("invalid_profile", "Unknown profile source.")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise RemoteKeyError(
                "invalid_name",
                "A key name is 1-48 characters: lower-case letters, digits, '.', '_' "
                "or '-', starting with a letter or digit.")
        if scope not in SCOPES:
            raise RemoteKeyError("invalid_scope", "A key scope is 'read' or 'write'.")
        with self._exclusive():
            records = self._records_for_write()
            if any(r.name == name and r.active for r in records):
                raise RemoteKeyError(
                    "duplicate_name",
                    f"An active key named '{name}' already exists. Revoke it first, or "
                    "choose another name.")
            secret = KEY_PREFIX + secrets.token_urlsafe(32)
            known_ids = {r.key_id for r in records}
            key_id = "rk_" + secrets.token_hex(4)
            while key_id in known_ids:
                key_id = "rk_" + secrets.token_hex(4)
            record = RemoteKey(key_id=key_id, name=name, scope=scope,
                               digest=digest_secret(secret), created_at=_now(),
                               profile=profile, profile_source=profile_source)
            self._write(records + (record,))
        return record, secret

    def revoke(self, name_or_id: str) -> RemoteKey:
        with self._exclusive():
            records = self._records_for_write()
            target = next((r for r in records
                           if r.active and name_or_id in (r.name, r.key_id)), None)
            if target is None:
                raise RemoteKeyError("not_found",
                                     f"No active remote key is named '{name_or_id}'.")
            revoked = replace(target, revoked_at=_now())
            self._write(tuple(revoked if r is target else r for r in records))
        return revoked

    def set_extras(self, name_or_id: str, extras: Iterable[str]) -> RemoteKey:
        """Replace the opt-ins of one active key."""
        wanted = frozenset(extras)
        if not wanted <= frozenset(EXTRAS):
            raise RemoteKeyError("invalid_extra",
                                 "A key can be allowed: " + ", ".join(EXTRAS) + ".")
        with self._exclusive():
            records = self._records_for_write()
            target = next((r for r in records
                           if r.active and name_or_id in (r.name, r.key_id)), None)
            if target is None:
                raise RemoteKeyError("not_found",
                                     f"No active remote key is named '{name_or_id}'.")
            changed = replace(target, extras=wanted)
            self._write(tuple(changed if r is target else r for r in records))
        return changed

    def bind_unbound(self, profile: str) -> tuple[RemoteKey, ...]:
        """Bind every active key that has no profile (made before 4.1.20) to
        ``profile`` - the profile active now, which is the one those keys have
        been reaching. Returns the keys it bound; a no-op when there are none.
        """
        if not valid_profile_id(profile):
            raise RemoteKeyError("invalid_profile", f"Not a profile id: {profile!r}.")
        if not any(r.active and r.profile is None for r in self._load()):
            return ()
        with self._exclusive():
            records = self._records_for_write()
            bound = tuple(replace(r, profile=profile, profile_source="bound-on-upgrade")
                          for r in records if r.active and r.profile is None)
            if not bound:
                return ()
            by_id = {r.key_id: r for r in bound}
            self._write(tuple(by_id.get(r.key_id, r) if r.active else r for r in records))
        logger.warning("Bound %d remote key(s) made before 4.1.20 to profile '%s': %s",
                       len(bound), profile, ", ".join(r.name for r in bound))
        return bound


_DEFAULT_STORE: RemoteKeyStore | None = None
_DEFAULT_LOCK = threading.Lock()


def default_store() -> RemoteKeyStore:
    """The store for the active data root (the path is resolved per call)."""
    global _DEFAULT_STORE
    with _DEFAULT_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = RemoteKeyStore()
        return _DEFAULT_STORE


__all__ = [
    "EXTRAS",
    "KEY_PREFIX",
    "PROFILE_SOURCES",
    "RemoteKey",
    "RemoteKeyError",
    "RemoteKeyStore",
    "SCOPES",
    "STORE_FILE",
    "default_store",
    "digest_secret",
    "store_problem",
    "valid_profile_id",
]
