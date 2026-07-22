# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Trust Gate (Pre-Operation Checks).

Enforces minimum trust thresholds before allowing write/delete operations.
Read operations always pass but are logged for audit purposes.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.trust.scorer import TrustScorer

logger = logging.getLogger(__name__)


def _env_threshold(name: str, default: float) -> float:
    """Read a [0,1] threshold from the environment, falling back on parse/range error."""
    try:
        value = float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if 0.0 <= value <= 1.0 else default


# M-02/M-03 (3.7.9): thresholds are operator-tunable so multi-agent deployments
# can tighten the write gate and opt into a read gate. Defaults are unchanged —
# the audit's "raise write_threshold to 0.7" would block every new agent (a new
# agent scores 0.5, Beta(1,1)); loopback callers bypass the gate entirely, so
# this only affects remote credentialed agents. Read gate is OFF by default.
_DEFAULT_WRITE_THRESHOLD = _env_threshold("SLM_TRUST_WRITE_THRESHOLD", 0.3)
_DEFAULT_DELETE_THRESHOLD = _env_threshold("SLM_TRUST_DELETE_THRESHOLD", 0.5)
_DEFAULT_READ_THRESHOLD = _env_threshold("SLM_TRUST_READ_THRESHOLD", 0.1)
_READ_GATE_ENABLED = os.environ.get("SLM_TRUST_READ_GATE") == "1"


class TrustError(PermissionError):
    """Raised when an agent fails a trust check.

    Attributes:
        agent_id: The agent that failed the check.
        trust_score: The agent's current trust score.
        threshold: The minimum required trust score.
        operation: The operation that was attempted.
    """

    def __init__(
        self,
        agent_id: str,
        trust_score: float,
        threshold: float,
        operation: str,
    ) -> None:
        self.agent_id = agent_id
        self.trust_score = trust_score
        self.threshold = threshold
        self.operation = operation
        super().__init__(
            f"Agent '{agent_id}' trust {trust_score:.3f} below "
            f"{operation} threshold {threshold:.3f}"
        )


class TrustGate:
    """Pre-operation trust checks.

    Operations are gated by minimum trust thresholds:
    - write: agent must have trust >= write_threshold (default 0.3)
    - delete: agent must have trust >= delete_threshold (default 0.5)
    - read: always passes (logged for audit trail)

    Raises TrustError if the agent's trust is too low.
    """

    def __init__(
        self,
        scorer: TrustScorer,
        write_threshold: float = _DEFAULT_WRITE_THRESHOLD,
        delete_threshold: float = _DEFAULT_DELETE_THRESHOLD,
        read_threshold: float = _DEFAULT_READ_THRESHOLD,
        read_gate_enabled: bool = _READ_GATE_ENABLED,
    ) -> None:
        if write_threshold < 0 or write_threshold > 1:
            raise ValueError("write_threshold must be in [0, 1]")
        if delete_threshold < 0 or delete_threshold > 1:
            raise ValueError("delete_threshold must be in [0, 1]")
        if read_threshold < 0 or read_threshold > 1:
            raise ValueError("read_threshold must be in [0, 1]")

        self._scorer = scorer
        self._write_threshold = write_threshold
        self._delete_threshold = delete_threshold
        self._read_threshold = read_threshold
        self._read_gate_enabled = read_gate_enabled

    @property
    def write_threshold(self) -> float:
        return self._write_threshold

    @property
    def delete_threshold(self) -> float:
        return self._delete_threshold

    def check_write(self, agent_id: str, profile_id: str) -> None:
        """Check if agent is trusted enough to write.

        Raises:
            TrustError: If agent trust is below write_threshold.
        """
        score = self._scorer.get_agent_trust(agent_id, profile_id)
        logger.debug(
            "trust gate write: agent=%s trust=%.3f threshold=%.3f",
            agent_id, score, self._write_threshold,
        )
        if score < self._write_threshold:
            raise TrustError(
                agent_id, score, self._write_threshold, "write"
            )

    def check_delete(self, agent_id: str, profile_id: str) -> None:
        """Check if agent is trusted enough to delete.

        Delete requires higher trust than write because it is destructive.

        Raises:
            TrustError: If agent trust is below delete_threshold.
        """
        score = self._scorer.get_agent_trust(agent_id, profile_id)
        logger.debug(
            "trust gate delete: agent=%s trust=%.3f threshold=%.3f",
            agent_id, score, self._delete_threshold,
        )
        if score < self._delete_threshold:
            raise TrustError(
                agent_id, score, self._delete_threshold, "delete"
            )

    @property
    def read_threshold(self) -> float:
        return self._read_threshold

    @property
    def read_gate_enabled(self) -> bool:
        return self._read_gate_enabled

    def check_read(self, agent_id: str, profile_id: str) -> None:
        """Read check. Passes by default; logged for the audit trail.

        M-03 (3.7.9): reads are unblocked by default because denying read access
        could break agent functionality. Operators who need to stop a
        compromised agent from exfiltrating the store can set
        ``SLM_TRUST_READ_GATE=1`` (optionally with ``SLM_TRUST_READ_THRESHOLD``)
        to enforce a minimum trust for reads too.
        """
        score = self._scorer.get_agent_trust(agent_id, profile_id)
        logger.debug(
            "trust gate read: agent=%s trust=%.3f gate=%s",
            agent_id, score, self._read_gate_enabled,
        )
        if self._read_gate_enabled and score < self._read_threshold:
            raise TrustError(
                agent_id, score, self._read_threshold, "read"
            )
