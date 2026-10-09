# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Namespace-aware identity for the local SuperLocalMemory daemon.

PID files and listening ports are liveness hints, not ownership credentials.
This module binds a daemon process to one canonical SLM data root and one
random process instance through a private, atomically written descriptor.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from superlocalmemory.infra.data_root import canonical_data_root
from superlocalmemory.infra.process_identity import process_start_token_for

if TYPE_CHECKING:  # typing only: infra must not depend on the lock module at runtime
    from superlocalmemory.infra.instance_lock import InstanceLock

DAEMON_DESCRIPTOR_SCHEMA = 1
DAEMON_PROTOCOL = 1
DAEMON_SERVICE = "superlocalmemory-daemon"
_NAMESPACE_DOMAIN = b"superlocalmemory-daemon-namespace-v1\0"
_CAPABILITY_DOMAIN = b"superlocalmemory-daemon-capability-v1\0"

logger = logging.getLogger(__name__)


def _canonical_path(value: str | Path) -> Path:
    expanded = Path(value).expanduser().resolve(strict=False)
    return Path(os.path.normcase(str(expanded)))


def namespace_id_for(data_root: str | Path) -> str:
    """Return a stable, non-reversible identifier for a canonical data root."""
    canonical = _canonical_path(data_root)
    normalized = os.path.normcase(str(canonical)).encode("utf-8")
    return hashlib.sha256(_NAMESPACE_DOMAIN + normalized).hexdigest()


def capability_fingerprint(capability: str) -> str:
    """Hash the private local capability before exposing it through health."""
    return hashlib.sha256(_CAPABILITY_DOMAIN + capability.encode("utf-8")).hexdigest()


def owner_id() -> str:
    """Cross-platform local owner identifier used for accidental isolation."""
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        return f"uid:{getuid()}"
    from superlocalmemory.core.platform_utils import current_user_name

    return f"user:{current_user_name()}"


def process_create_time_for(pid: int) -> float:
    """Return OS process creation time, falling back only when unavailable."""
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return time.time()


@dataclass(frozen=True)
class DaemonDescriptor:
    schema: int
    service: str
    daemon_protocol: int
    namespace_id: str
    instance_id: str
    capability: str
    capability_fingerprint: str
    owner_id: str
    data_root: str
    pid: int
    process_create_time: float
    port: int
    state: str
    version: str
    started_at: float
    # Clock-independent process identity. Optional and defaulted so a
    # descriptor written by an older release still parses; platforms without a
    # boot-relative start time (Windows) legitimately store None and fall back
    # to the creation-time comparison. See infra/process_identity.py.
    process_start_token: str | None = None

    def public_health_fields(self) -> dict[str, Any]:
        """Identity fields safe to expose on the loopback health endpoint."""
        return {
            "service": self.service,
            "daemon_protocol": self.daemon_protocol,
            "namespace_id": self.namespace_id,
            "instance_id": self.instance_id,
            "capability_fingerprint": self.capability_fingerprint,
            "owner_id": self.owner_id,
            "pid": self.pid,
            "port": self.port,
            "state": self.state,
            "version": self.version,
        }


def build_descriptor(
    *,
    data_root: str | Path | None = None,
    port: int,
    version: str,
    pid: int | None = None,
    process_create_time: float | None = None,
    process_start_token: str | None = None,
    instance_id: str | None = None,
    capability: str | None = None,
    state: str = "starting",
    started_at: float | None = None,
) -> DaemonDescriptor:
    """Build one daemon-process descriptor for the selected namespace."""
    root = _canonical_path(data_root) if data_root is not None else canonical_data_root()
    actual_pid = int(pid if pid is not None else os.getpid())
    actual_capability = capability or secrets.token_urlsafe(32)
    return DaemonDescriptor(
        schema=DAEMON_DESCRIPTOR_SCHEMA,
        service=DAEMON_SERVICE,
        daemon_protocol=DAEMON_PROTOCOL,
        namespace_id=namespace_id_for(root),
        instance_id=instance_id or str(uuid.uuid4()),
        capability=actual_capability,
        capability_fingerprint=capability_fingerprint(actual_capability),
        owner_id=owner_id(),
        data_root=str(root),
        pid=actual_pid,
        process_create_time=float(
            process_create_time
            if process_create_time is not None
            else process_create_time_for(actual_pid)
        ),
        port=int(port),
        state=state,
        version=version,
        started_at=float(started_at if started_at is not None else time.time()),
        process_start_token=(
            process_start_token
            if process_start_token is not None
            else process_start_token_for(actual_pid)
        ),
    )


def descriptor_path(data_root: str | Path | None = None) -> Path:
    root = _canonical_path(data_root) if data_root is not None else canonical_data_root()
    return root / "daemon.json"


def _restrict_to_owner(path: Path) -> None:
    """Restrict a sensitive file to the current user on every platform.

    ``daemon.json`` holds the capability token that authorizes write requests.
    POSIX gets a 0600 chmod. On Windows chmod is a no-op, so use icacls to strip
    inherited ACEs and grant the current user only — otherwise the token can be
    read by other local accounts (privilege escalation on shared machines).
    Files under %USERPROFILE% usually inherit user-only ACLs already; this is
    defense-in-depth. Fail-soft — a hardening failure warns, never crashes.
    """
    if sys.platform == "win32":
        try:
            from superlocalmemory.core.platform_utils import current_account

            user = current_account()
            subprocess.run(
                ["icacls", str(path), "/inheritance:r"],
                check=False, capture_output=True,
            )
            if user:
                subprocess.run(
                    ["icacls", str(path), "/grant:r", f"{user}:F"],
                    check=False, capture_output=True,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "could not restrict ACL on %s (%s); the capability token may be "
                "readable by other local users", path, exc,
            )
    else:
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            logger.warning("could not chmod %s to 0600: %s", path, exc)


def write_descriptor(
    descriptor: DaemonDescriptor,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Atomically publish a mode-0600 descriptor in its owned namespace."""
    root = _canonical_path(data_root or descriptor.data_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "daemon.json"
    temporary = root / f".daemon.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    payload = json.dumps(asdict(descriptor), sort_keys=True, separators=(",", ":"))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _restrict_to_owner(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_descriptor(
    *, data_root: str | Path | None = None,
) -> DaemonDescriptor | None:
    """Read and validate the local descriptor; malformed state fails closed."""
    path = descriptor_path(data_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        descriptor = DaemonDescriptor(**raw)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if descriptor.schema != DAEMON_DESCRIPTOR_SCHEMA:
        return None
    if descriptor.service != DAEMON_SERVICE:
        return None
    if descriptor.daemon_protocol != DAEMON_PROTOCOL:
        return None
    expected_root = (
        _canonical_path(data_root) if data_root is not None else canonical_data_root()
    )
    if Path(descriptor.data_root).resolve(strict=False) != expected_root:
        return None
    if descriptor.namespace_id != namespace_id_for(expected_root):
        return None
    if descriptor.capability_fingerprint != capability_fingerprint(
        descriptor.capability
    ):
        return None
    if descriptor.owner_id != owner_id():
        return None
    if not (1 <= descriptor.port <= 65535) or descriptor.pid <= 0:
        return None
    if descriptor.process_start_token is not None and not isinstance(
        descriptor.process_start_token, str
    ):
        return None
    return descriptor


def descriptor_matches_health(
    descriptor: DaemonDescriptor,
    health: Mapping[str, Any],
) -> bool:
    """Require exact owned-daemon identity before a client attaches."""
    expected = descriptor.public_health_fields()
    identity_keys = (
        "service",
        "daemon_protocol",
        "namespace_id",
        "instance_id",
        "capability_fingerprint",
        "owner_id",
        "version",
        "pid",
        "port",
    )
    for key in identity_keys:
        left = str(expected[key])
        right = str(health.get(key, ""))
        if not hmac.compare_digest(left, right):
            return False
    return True


def health_is_same_account(
    health: Mapping[str, Any],
    *,
    data_root: str | Path | None = None,
) -> bool:
    """An SLM daemon of this account and this data folder answered.

    Weaker than :func:`descriptor_matches_health` (it does not prove the exact
    process), for callers that must recognise a sibling daemon of their own
    namespace whose instance the descriptor no longer records. A daemon run by
    another account on this computer never passes: its owner differs, and so
    does its data folder.
    """
    root = _canonical_path(data_root) if data_root is not None else canonical_data_root()
    expected = {
        "service": DAEMON_SERVICE,
        "daemon_protocol": str(DAEMON_PROTOCOL),
        "owner_id": owner_id(),
        "namespace_id": namespace_id_for(root),
    }
    return all(hmac.compare_digest(str(value), str(health.get(key, "")))
               for key, value in expected.items())


def health_is_other_account(health: Mapping[str, Any]) -> bool:
    """The answer came from an SLM daemon run by another account on this computer."""
    other = str(health.get("owner_id", "") or "")
    return (str(health.get("service", "")) == DAEMON_SERVICE
            and bool(other) and other != owner_id())


def clear_descriptor(
    instance_id: str,
    *,
    data_root: str | Path | None = None,
) -> bool:
    """Delete the descriptor only when the caller still owns that instance."""
    descriptor = read_descriptor(data_root=data_root)
    if descriptor is None or not hmac.compare_digest(
        descriptor.instance_id, instance_id
    ):
        return False
    try:
        descriptor_path(data_root).unlink()
        return True
    except OSError:
        return False


def write_mirror_atomic(path: Path, text: str) -> None:
    """Write a plain-text mirror (pid or port) atomically with mode 0600."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_if_owner(descriptor: DaemonDescriptor, lock: "InstanceLock") -> bool:
    """Publish the daemon record, but only as the holder of the instance lock.

    The record names the one process that owns the data folder. A process that
    does not hold the lock never writes it, however it came to believe it
    should; that is what keeps a losing starter from overwriting the winner.
    """
    still_owns = getattr(lock, "still_owns_file", None)
    if not lock.held or (still_owns is not None and not still_owns()):
        logger.warning("not the instance-lock owner; record untouched")
        return False
    root = _canonical_path(descriptor.data_root)
    write_descriptor(descriptor, data_root=root)
    base = root / "daemon.json"
    # Mirrors are a courtesy; a reader holding one open (Windows) must not
    # fail a daemon start. The guardian repairs a missing mirror.
    for name, value in (("daemon.pid", descriptor.pid), ("daemon.port", descriptor.port)):
        try:
            write_mirror_atomic(base.with_name(name), str(value))
        except OSError as exc:
            logger.warning("could not write the %s mirror: %s", name, exc)
    return True


def clear_descriptor_if_owner(
    instance_id: str,
    lock: "InstanceLock",
    *,
    data_root: str | Path | None = None,
) -> bool:
    """Remove the record and its mirrors, only as owner of that instance."""
    if not lock.held:
        return False
    descriptor = read_descriptor(data_root=data_root)
    if descriptor is None or not clear_descriptor(instance_id, data_root=data_root):
        return False
    base = descriptor_path(data_root)
    for name, expected in (
        ("daemon.pid", str(descriptor.pid)),
        ("daemon.port", str(descriptor.port)),
    ):
        path = base.with_name(name)
        try:
            if path.read_text(encoding="utf-8").strip() == expected:
                path.unlink()
        except OSError:
            pass
    return True
