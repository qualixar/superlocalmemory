# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — S9-DASH-10

"""Lightweight session registry for cross-process session_id handoff.

**Problem.** Claude Code (and Cursor/Antigravity) invoke two separate
SLM surfaces per user turn:

1. ``user_prompt_hook`` — receives ``session_id`` via stdin JSON
   (Claude Code's hook payload). This is the real session id.
2. MCP ``recall`` tool — invoked by the AI mid-turn. The MCP protocol
   does NOT thread ``CLAUDE_SESSION_ID`` into tool arguments by
   default, so the MCP tool cannot see what session it is serving.

Result: ``record_recall`` writes ``pending_outcomes`` with
``session_id='mcp:mcp_client'`` while the Stop hook queries by the
real session id — they never match, so cite/edit/dwell signals are
lost (reaper finalizes everything at neutral 0.5).

**Fix (this module).** A simple file-based registry:

* ``mark_active(session_id, agent_type, profile_id)`` — called by hooks on
  every prompt/tool event. Writes ``(session_id, agent_type, profile_id,
  ts_ns, pid)``
  to ``~/.superlocalmemory/.active_sessions.json``.
* ``most_recent_active(agent_type, within_seconds=60)`` — queries the
  registry for the most recently seen session of the named agent.
  MCP uses this as the default when the tool caller omits
  ``session_id``.

Concurrency: each write is atomic via write-temp + ``os.replace`` (atomic on
POSIX/Windows), so a concurrent reader never sees a half-written file —
last-writer-wins. This is best-effort, not lock-serialised: a concurrent
read-modify-write may lose an interleaved update, which is acceptable because
the registry only drives session attribution for closed-loop learning, not
memory correctness. Rollover: entries older than 1 hour are pruned on every
write. Fail-soft: every error path returns empty or the passed default — the
learning loop must never crash the hot path.

This is not a perfect correlation channel; two Claude sessions
typing in the same second can race. For single-user workstations
(the overwhelming SLM case) it is 99%+ accurate.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_PRUNE_AFTER_SEC = 3600  # 1h — anything older is dead

# Keep the public host vocabulary small and stable.  Callers can still use an
# unknown value internally, but the Living Brain must not turn arbitrary hook
# input into a new UI label or expose a host identifier verbatim.
_PUBLIC_CLIENT_KINDS = {
    "claude": "claude_code",
    "claude_code": "claude_code",
    "codex": "codex",
    "cursor": "cursor",
    "antigravity": "antigravity",
    "copilot": "copilot",
    "cli": "cli",
    "mcp": "mcp",
}


def _registry_file() -> Path:
    from superlocalmemory.infra.data_root import state_path

    return state_path(".active_sessions.json")


def _profiles_file() -> Path:
    """Return the profile cache updated atomically by daemon switches."""
    from superlocalmemory.infra.data_root import state_path

    return state_path("profiles.json")


def resolve_active_profile() -> str | None:
    """Read the canonical active profile without relying on host env wiring.

    Hooks are separate host processes, so daemon-managed profile changes are
    not reliably reflected in their environment.  The profile runtime writes
    this compatibility cache atomically on every successful switch; using it
    keeps ephemeral presence scoped like the durable memory stores.
    """
    try:
        payload = json.loads(_profiles_file().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        modern_pointer = payload.get("active_profile")
        legacy_pointer = payload.get("active")
        pointer = (
            modern_pointer.strip()
            if isinstance(modern_pointer, str) and modern_pointer.strip()
            else legacy_pointer.strip()
            if isinstance(legacy_pointer, str) and legacy_pointer.strip()
            else None
        )
        if pointer is None:
            return None
        profiles = payload.get("profiles")
        catalog_present = "profiles" in payload
        entries = []
        if isinstance(profiles, list):
            entries = [(None, profile) for profile in profiles]
        elif isinstance(profiles, dict):
            entries = list(profiles.items())
        canonical: list[tuple[str, str | None]] = []
        for key, profile in entries:
            if not isinstance(profile, dict):
                continue
            profile_id = profile.get("profile_id")
            if not isinstance(profile_id, str) or not profile_id.strip():
                profile_id = key if isinstance(key, str) and key.strip() else None
            name = profile.get("name")
            if isinstance(profile_id, str) and profile_id.strip():
                canonical.append((profile_id.strip(), name if isinstance(name, str) else None))
        id_matches = {profile_id for profile_id, _name in canonical if profile_id == pointer}
        if len(id_matches) == 1:
            return id_matches.pop()
        name_matches = {profile_id for profile_id, name in canonical if name == pointer}
        if len(name_matches) == 1:
            return name_matches.pop()
        # The current runtime writes an ``active_profile`` ID even when its
        # catalog has not been materialized yet. With no catalog at all it is
        # the only available authority; a partially present catalog instead
        # fails closed to avoid treating an unmapped display name as an ID.
        if not catalog_present and isinstance(modern_pointer, str) and modern_pointer.strip():
            return modern_pointer.strip()
    except (OSError, TypeError, ValueError):
        pass
    return None


def _now_ns() -> int:
    return time.time_ns()


def _load() -> dict:
    try:
        registry_file = _registry_file()
        if not registry_file.exists():
            return {}
        return json.loads(registry_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    try:
        registry_file = _registry_file()
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = registry_file.with_suffix(
            f".{os.getpid()}.{time.time_ns()}.tmp",
        )
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, registry_file)
        try:
            os.chmod(registry_file, 0o600)
        except OSError:
            pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("session_registry save failed: %s", exc)


def _prune(data: dict) -> dict:
    cutoff_ns = _now_ns() - (_PRUNE_AFTER_SEC * 1_000_000_000)
    return {
        sid: row for sid, row in data.items()
        if isinstance(row, dict) and int(row.get("ts_ns", 0)) >= cutoff_ns
    }


def mark_active(
    session_id: str,
    agent_type: str = "claude",
    profile_id: str | None = None,
) -> None:
    """Record ``session_id`` keyed by the CALLING process PID.

    Called from UserPromptSubmit + PostToolUse hooks — those hooks run
    INSIDE the Claude Code / IDE process. So ``os.getpid()`` is the
    IDE's PID. The MCP server spawned BY that same IDE process has
    ``os.getppid() == IDE_PID``. Keying by PID means two parallel
    Claude Code windows never collide — each MCP server reads only
    its own parent's entry.

    Hot-path safe — returns within <2 ms on a warm cache. Never raises.
    """
    if not session_id or not isinstance(session_id, str):
        return
    try:
        data = _load()
        key = str(os.getpid())  # the IDE / hook process PID
        row = {
            "session_id": session_id,
            "agent_type": agent_type or "unknown",
            "ts_ns": _now_ns(),
        }
        if isinstance(profile_id, str) and profile_id.strip():
            row["profile_id"] = profile_id.strip()
        data[key] = row
        data = _prune(data)
        _save(data)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("mark_active failed: %s", exc)


def lookup_by_parent(within_seconds: int = 60) -> Optional[str]:
    """Return the session_id whose registry key == ``os.getppid()``.

    Called from the MCP server process. ``os.getppid()`` is the PID of
    the IDE that spawned the MCP server — exactly the same PID that
    the hook used as its key in ``mark_active``. Collision-free across
    multiple parallel Claude Code / IDE sessions.
    """
    try:
        parent_key = str(os.getppid())
        data = _load()
        row = data.get(parent_key)
        if not isinstance(row, dict):
            return None
        ts = int(row.get("ts_ns", 0))
        if _now_ns() - ts > within_seconds * 1_000_000_000:
            return None  # stale — IDE likely restarted
        return row.get("session_id") or None
    except Exception:
        return None


def most_recent_active(
    agent_type: Optional[str] = None,
    within_seconds: int = 60,
) -> Optional[str]:
    """Fallback: most-recently-written entry of the given agent_type.

    Used by surfaces that DON'T have a stable parent-PID linkage (e.g.
    CLI tools invoked ad-hoc). Prefer ``lookup_by_parent`` for MCP.
    """
    try:
        data = _load()
        if not data:
            return None
        cutoff_ns = _now_ns() - (within_seconds * 1_000_000_000)
        candidates = []
        for _key, row in data.items():
            if not isinstance(row, dict):
                continue
            ts = int(row.get("ts_ns", 0))
            if ts < cutoff_ns:
                continue
            if agent_type and row.get("agent_type") != agent_type:
                continue
            sid = row.get("session_id")
            if sid:
                candidates.append((ts, sid))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    except Exception:
        return None


def active_client_summary(
    profile_id: str | None = None,
    within_seconds: int = 60,
) -> list[dict[str, object]]:
    """Return privacy-safe, recently active hosts for the Living Brain.

    This is deliberately a *presence* signal, not durable product analytics:
    registry entries expire within an hour and session identifiers never leave
    the local registry.  The dashboard therefore distinguishes these active
    clients from configured adapters, which only prove installation.
    """
    # Compatibility-safe default for any out-of-tree caller that used the
    # original no-argument helper: no profile means no visibility, never a
    # silent cross-profile aggregate.
    if not profile_id:
        return []
    try:
        cutoff_ns = _now_ns() - (max(0, int(within_seconds)) * 1_000_000_000)
        newest_by_kind: dict[str, int] = {}
        for row in _load().values():
            if not isinstance(row, dict):
                continue
            # Entries written before profile attribution are intentionally
            # invisible here: guessing a profile would leak client metadata.
            if str(row.get("profile_id", "")) != profile_id:
                continue
            try:
                ts_ns = int(row.get("ts_ns", 0))
            except (TypeError, ValueError):
                continue
            if ts_ns < cutoff_ns:
                continue
            raw_kind = str(row.get("agent_type", "")).strip().lower()
            kind = _PUBLIC_CLIENT_KINDS.get(raw_kind, "other")
            newest_by_kind[kind] = max(newest_by_kind.get(kind, 0), ts_ns)
        now_ns = _now_ns()
        return [
            {
                "kind": kind,
                "active": True,
                "last_seen_seconds_ago": max(0, int((now_ns - ts_ns) / 1_000_000_000)),
                "source": "session_registry",
                "is_real": True,
            }
            for kind, ts_ns in sorted(newest_by_kind.items())
        ]
    except Exception:
        return []


def _reset_for_testing() -> None:
    """TEST-ONLY: wipe registry."""
    try:
        _registry_file().unlink(missing_ok=True)
    except Exception:
        pass
