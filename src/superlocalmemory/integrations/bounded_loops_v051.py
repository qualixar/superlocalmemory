"""Read-only public-CLI adapter for bounded-loops v0.5.1 graph receipts.

SLM deliberately neither imports bounded-loops nor reimplements its event-log
grammar.  The optional, installed ``bl`` executable is the versioned protocol
port: its public ``graph status`` command reconstructs and validates a graph
receipt before SLM accepts the resulting projection.

v0.5.1's local receipt is explicitly unverified.  Consequently every result
from this adapter is display/observation evidence only; it cannot promote a
memory, alter learning, or assert execution authority.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BoundedLoopsReceiptError(ValueError):
    """Raised when a bounded-loops v0.5.1 receipt cannot be safely observed."""


_VERSION = "0.5.1"
_TIMEOUT_SECONDS = 15
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024
_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})


@dataclass(frozen=True)
class VerifiedBoundedLoopsReceipt:
    """A normalized, non-promotable projection of a local v0.5.1 receipt."""

    organization_id: str
    project_id: str
    run_id: str
    terminal_status: str
    receipt_digest: str
    artifact_digests: tuple[str, ...]
    event_count: int
    demonstration: bool
    trust_level: str = "local_unverified"
    eligible_for_learning: bool = False


def verify_v051_graph_receipt(
    run_dir: str | Path,
    *,
    bl_executable: str | Path | None = None,
) -> VerifiedBoundedLoopsReceipt:
    """Read a v0.5.1 graph run validated by the bounded-loops public CLI.

    ``run_dir`` must be an existing local directory, not a symlink.  The
    executable is discovered from the local environment or supplied as an
    absolute path for an explicitly configured integration.  Arguments are
    always passed as an argv list, never through a shell.
    """
    directory = _safe_run_directory(run_dir)
    executable = _resolve_v051_executable(bl_executable)
    status = _command_json(executable, "graph", "status", "--run", str(directory), "--json")
    return _normalize_projection(status)


def _safe_run_directory(run_dir: str | Path) -> Path:
    supplied = Path(run_dir)
    if not supplied.is_absolute():
        raise BoundedLoopsReceiptError("run directory must be an absolute path")
    try:
        directory = supplied.resolve(strict=True)
    except OSError as exc:
        raise BoundedLoopsReceiptError("run directory is unavailable") from exc
    if supplied.is_symlink() or not directory.is_dir():
        raise BoundedLoopsReceiptError("run directory must be a real directory")
    return directory


def _resolve_v051_executable(configured: str | Path | None) -> str:
    if configured is None:
        discovered = shutil.which("bl")
        if discovered is None:
            raise BoundedLoopsReceiptError("bounded-loops v0.5.1 is not installed")
        executable = Path(discovered)
    else:
        executable = Path(configured)
        if not executable.is_absolute():
            raise BoundedLoopsReceiptError("configured bl executable must be an absolute path")
    try:
        resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise BoundedLoopsReceiptError("configured bl executable is unavailable") from exc
    if not resolved.is_file():
        raise BoundedLoopsReceiptError("configured bl executable is not a file")
    version = _run_command(str(resolved), "--version")
    if version.strip() != f"bl {_VERSION}":
        raise BoundedLoopsReceiptError("bounded-loops executable must be exactly v0.5.1")
    return str(resolved)


def _command_json(executable: str, *arguments: str) -> Any:
    output = _run_command(executable, *arguments)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise BoundedLoopsReceiptError("bounded-loops returned invalid JSON") from exc


def _run_command(executable: str, *arguments: str) -> str:
    try:
        process = subprocess.Popen(
            [executable, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = _read_bounded_output(process)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BoundedLoopsReceiptError("bounded-loops command did not complete") from exc
    if process.returncode != 0:
        raise BoundedLoopsReceiptError("bounded-loops rejected the graph receipt")
    try:
        return stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BoundedLoopsReceiptError("bounded-loops response is not UTF-8") from exc


def _read_bounded_output(process: subprocess.Popen[bytes]) -> bytes:
    """Read both pipes and terminate a command whose output exceeds the cap.

    One reader thread per pipe, as ``subprocess.communicate`` does, because
    Windows can only ``select()`` on sockets: with a selector every command
    failed there (WinError 10038). Returns stdout; stderr is drained and
    counted toward the cap but not kept.
    """
    if process.stdout is None or process.stderr is None:
        raise BoundedLoopsReceiptError("bounded-loops command pipes are unavailable")
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    stdout = bytearray()
    counted = threading.Lock()
    total = [0]
    too_big = threading.Event()

    def pump(stream, keep: bool) -> None:
        while not too_big.is_set():
            chunk = stream.read1(64 * 1024)
            if not chunk:
                return
            with counted:
                total[0] += len(chunk)
                if total[0] > _MAX_OUTPUT_BYTES:
                    too_big.set()
                    return
                if keep:
                    stdout.extend(chunk)

    readers = [
        threading.Thread(target=pump, args=(process.stdout, True),
                         name="bl-stdout", daemon=True),
        threading.Thread(target=pump, args=(process.stderr, False),
                         name="bl-stderr", daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        while any(reader.is_alive() for reader in readers) and not too_big.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, _TIMEOUT_SECONDS)
            too_big.wait(min(remaining, 0.05))
        if too_big.is_set():
            raise BoundedLoopsReceiptError(
                "bounded-loops response exceeds the import size limit"
            )
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for reader in readers:  # the pipes close with the process
            reader.join(timeout=1.0)
    with counted:
        return bytes(stdout)


def _normalize_projection(status: Any) -> VerifiedBoundedLoopsReceipt:
    if not isinstance(status, dict):
        raise BoundedLoopsReceiptError("bounded-loops response has an invalid shape")
    required_strings = (
        "organization_id", "project_id", "run_id", "run_state", "receipt_head_hash",
    )
    if any(not isinstance(status.get(name), str) or not status[name] for name in required_strings):
        raise BoundedLoopsReceiptError("bounded-loops status lacks required receipt fields")
    if status["run_state"] not in _TERMINAL_STATES:
        raise BoundedLoopsReceiptError("bounded-loops receipt is not terminal")
    if status.get("verified") is not False:
        raise BoundedLoopsReceiptError("unexpected bounded-loops receipt verification state")
    sequence = status.get("receipt_sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise BoundedLoopsReceiptError("bounded-loops receipt sequence is invalid")
    if not _is_hash(status["receipt_head_hash"]):
        raise BoundedLoopsReceiptError("bounded-loops receipt digest is invalid")
    artifact_digests = _projection_artifact_digests(status.get("nodes"))
    demonstration = status.get("demonstration")
    if not isinstance(demonstration, bool):
        raise BoundedLoopsReceiptError("bounded-loops demonstration marker is invalid")
    return VerifiedBoundedLoopsReceipt(
        organization_id=status["organization_id"],
        project_id=status["project_id"],
        run_id=status["run_id"],
        terminal_status=status["run_state"],
        receipt_digest=f"sha256:{status['receipt_head_hash']}",
        artifact_digests=artifact_digests,
        event_count=sequence,
        demonstration=demonstration,
    )


def _projection_artifact_digests(nodes: Any) -> tuple[str, ...]:
    """Read only event-log-bound artifacts from the verified arena projection."""
    if not isinstance(nodes, list):
        raise BoundedLoopsReceiptError("bounded-loops status lacks node projections")
    digests: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise BoundedLoopsReceiptError("bounded-loops node projection is invalid")
        raw_digests = node.get("artifact_digests")
        if not isinstance(raw_digests, list) or not all(_is_digest(value) for value in raw_digests):
            raise BoundedLoopsReceiptError("bounded-loops node artifact digests are invalid")
        digests.update(raw_digests)
    return tuple(sorted(digests))


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
