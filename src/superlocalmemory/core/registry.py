# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Agent registry — multi-agent profile isolation.

Tracks which agents are using which profiles and prevents
two agents from writing to the same profile simultaneously
via an advisory lock mechanism. Persisted as JSON for
survival across process restarts.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProfileLockError(Exception):
    """Raised when an agent tries to write-lock an already-locked profile."""


class AgentRegistry:
    """In-memory agent registry with JSON file persistence.

    Each agent registers with an ID and a target profile.
    Advisory write locks prevent two agents from mutating
    the same profile at the same time.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._agents: dict[str, dict[str, Any]] = {}
        self._write_locks: dict[str, str] = {}  # profile_id -> agent_id
        self._path = persist_path
        if self._path and self._path.exists():
            self._load()

    # -- Public API ---------------------------------------------------------

    def register_agent(self, agent_id: str, profile_id: str) -> None:
        """Register an agent for a profile. Acquires advisory write lock."""
        if agent_id in self._agents:
            self.unregister_agent(agent_id)

        holder = self._write_locks.get(profile_id)
        if holder and holder != agent_id:
            raise ProfileLockError(
                f"Profile '{profile_id}' is locked by agent '{holder}'."
            )

        self._agents[agent_id] = {
            "profile_id": profile_id,
            "registered_at": time.time(),
        }
        self._write_locks[profile_id] = agent_id
        self._save()
        logger.info("Agent '%s' registered for profile '%s'", agent_id, profile_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent and release its write lock."""
        entry = self._agents.pop(agent_id, None)
        if entry:
            profile_id = entry["profile_id"]
            if self._write_locks.get(profile_id) == agent_id:
                del self._write_locks[profile_id]
            self._save()
            logger.info("Agent '%s' unregistered", agent_id)

    def get_agent_profile(self, agent_id: str) -> str:
        """Return the profile ID for a registered agent.

        Raises KeyError if the agent is not registered.
        """
        entry = self._agents.get(agent_id)
        if entry is None:
            raise KeyError(f"Agent '{agent_id}' is not registered.")
        return entry["profile_id"]

    def list_agents(self) -> list[dict[str, Any]]:
        """Return a snapshot of all registered agents."""
        return [
            {
                "agent_id": aid,
                "profile_id": info["profile_id"],
                "registered_at": info["registered_at"],
            }
            for aid, info in self._agents.items()
        ]

    # -- Persistence --------------------------------------------------------

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"agents": self._agents, "write_locks": self._write_locks}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._agents = data.get("agents", {})
            self._write_locks = data.get("write_locks", {})
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt registry file at %s — starting fresh.", self._path)
            self._agents = {}
            self._write_locks = {}
