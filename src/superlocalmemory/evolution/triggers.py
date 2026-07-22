# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Evolution Triggers — detect when skills need evolution.

3-trigger system (adopted from OpenSpace):
  1. Post-session analysis: session ends, scan for skill failures
  2. Skill degradation: behavioral assertion confidence drops
  3. Periodic health check: consolidation cycle scans all skills

All triggers are zero-LLM, zero-embedding. Pure SQLite queries.
They produce EvolutionCandidates — the evolver decides what to do.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from superlocalmemory.evolution.types import (
    EvolutionCandidate,
    EvolutionType,
    TriggerType,
)

logger = logging.getLogger(__name__)

# Thresholds — conservative, matching C1 design doc
DEGRADATION_THRESHOLD = 0.4      # Effective score below this → FIX candidate
RECOVERY_THRESHOLD = 0.6         # Above this → skill recovered, prune from addressed
DERIVED_THRESHOLD = 0.55         # Moderate effectiveness → DERIVED candidate
MIN_INVOCATIONS = 5              # Don't trigger on insufficient data
NEGATIVE_SIGNALS_THRESHOLD = 2   # Min negative signals in one session for post-session trigger
MAX_RSS_MB = 1024                # Skip evolution if process exceeds 1GB RSS


def _check_memory_pressure() -> bool:
    """Return True if process RSS exceeds threshold. Skip evolution if so."""
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        if rss_mb > MAX_RSS_MB:
            logger.warning(
                "Memory pressure: %dMB RSS > %dMB limit, skipping evolution",
                int(rss_mb), MAX_RSS_MB,
            )
            return True
    except ImportError:
        pass
    return False


class PostSessionTrigger:
    """Trigger 1: Analyze a completed session for skill failures.

    Scans tool_events for the given session_id. If any Skill call
    had 2+ negative signals (retries, errors), creates a FIX candidate.

    Zero-LLM. Runs in ~10ms on indexed table.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)

    def scan(self, session_id: str, profile_id: str = "default") -> list[EvolutionCandidate]:
        if _check_memory_pressure():
            return []

        candidates = []
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row

        try:
            # Get all Skill events for this session
            rows = conn.execute(
                "SELECT id, tool_name, input_summary, output_summary, created_at "
                "FROM tool_events "
                "WHERE session_id = ? AND profile_id = ? AND tool_name = 'Skill' "
                "ORDER BY id ASC",
                (session_id, profile_id),
            ).fetchall()

            if not rows:
                return []

            # Parse skill names and count per-skill signals
            skill_signals: dict[str, dict] = {}
            for row in rows:
                d = dict(row)
                skill_name = self._extract_skill_name(d)
                if not skill_name:
                    continue

                if skill_name not in skill_signals:
                    skill_signals[skill_name] = {
                        "positive": 0, "negative": 0, "event_ids": [],
                    }
                skill_signals[skill_name]["event_ids"].append(d["id"])

            # Check execution traces for each skill
            for skill_name, signals in skill_signals.items():
                for event_id in signals["event_ids"]:
                    trace = conn.execute(
                        "SELECT tool_name, output_summary FROM tool_events "
                        "WHERE profile_id = ? AND id > ? ORDER BY id ASC LIMIT 5",
                        (profile_id, event_id),
                    ).fetchall()

                    has_error = any(
                        "error" in (dict(t).get("output_summary", "") or "").lower()
                        for t in trace[:3]
                        if dict(t)["tool_name"] == "Bash"
                    )
                    has_retry = any(
                        dict(t)["tool_name"] == "Skill"
                        for t in trace[:3]
                    )

                    if has_error or has_retry:
                        signals["negative"] += 1
                    else:
                        signals["positive"] += 1

                total_invocations = len(signals["event_ids"])
                if signals["negative"] >= NEGATIVE_SIGNALS_THRESHOLD and total_invocations >= MIN_INVOCATIONS:
                    evidence = (
                        f"{signals['negative']} negative signals in session {session_id}",
                        f"{signals['positive']} positive signals",
                    )
                    candidates.append(EvolutionCandidate(
                        skill_name=skill_name,
                        evolution_type=EvolutionType.FIX,
                        trigger=TriggerType.POST_SESSION,
                        evidence=evidence,
                        effective_score=signals["positive"] / max(1, signals["positive"] + signals["negative"]),
                        invocation_count=len(signals["event_ids"]),
                        session_id=session_id,
                    ))

        except Exception as exc:
            logger.debug("Post-session trigger scan failed: %s", exc)
        finally:
            conn.close()

        return candidates

    def _extract_skill_name(self, event: dict) -> str:
        for field_key in ("input_summary", "output_summary"):
            raw = event.get(field_key, "") or ""
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
                name = parsed.get("skill") or parsed.get("commandName", "")
                if name:
                    return name
            except (json.JSONDecodeError, TypeError):
                # M-JSONPARSE: Fallback for non-JSON plain text
                m = re.search(r"skill[:\s]+['\"]?(\S+)", raw, re.IGNORECASE)
                if m:
                    return m.group(1).strip("'\"")
        return ""


class DegradationTrigger:
    """Trigger 2: Detect skills with declining performance assertions.

    Scans behavioral_assertions for skill_performance category.
    If a skill's confidence dropped below threshold → FIX candidate.
    If moderate with specific failure pattern → DERIVED candidate.

    Zero-LLM. Runs in ~5ms.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)

    def scan(self, profile_id: str = "default") -> list[EvolutionCandidate]:
        if _check_memory_pressure():
            return []

        candidates = []
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(
                "SELECT trigger_condition, action, confidence, evidence_count "
                "FROM behavioral_assertions "
                "WHERE profile_id = ? AND category = 'skill_performance' "
                "AND confidence > 0",
                (profile_id,),
            ).fetchall()

            for row in rows:
                d = dict(row)
                # H-TRIGGERPARSE: Extract skill name robustly.
                # Primary: regex on trigger_condition. Fallback: old replace approach.
                skill_name = ""
                tc = d["trigger_condition"]
                m = re.search(r"skill\s+(\S+)", tc)
                if m:
                    skill_name = m.group(1)
                else:
                    skill_name = tc.replace("when considering skill ", "")
                confidence = d["confidence"]
                evidence = d["evidence_count"]

                if evidence < MIN_INVOCATIONS:
                    continue

                # Parse effective score from action text
                effective_score = self._parse_effective_score(d["action"])

                if effective_score < DEGRADATION_THRESHOLD:
                    candidates.append(EvolutionCandidate(
                        skill_name=skill_name,
                        evolution_type=EvolutionType.FIX,
                        trigger=TriggerType.DEGRADATION,
                        evidence=(d["action"],),
                        effective_score=effective_score,
                        invocation_count=evidence,
                    ))
                elif effective_score < DERIVED_THRESHOLD:
                    candidates.append(EvolutionCandidate(
                        skill_name=skill_name,
                        evolution_type=EvolutionType.DERIVED,
                        trigger=TriggerType.DEGRADATION,
                        evidence=(d["action"],),
                        effective_score=effective_score,
                        invocation_count=evidence,
                    ))

        except Exception as exc:
            logger.debug("Degradation trigger scan failed: %s", exc)
        finally:
            conn.close()

        return candidates

    def get_active_degraded(self, profile_id: str = "default") -> set[str]:
        """Return names of currently degraded skills (for anti-loop pruning)."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        degraded = set()
        try:
            rows = conn.execute(
                "SELECT trigger_condition, action FROM behavioral_assertions "
                "WHERE profile_id = ? AND category = 'skill_performance'",
                (profile_id,),
            ).fetchall()
            for row in rows:
                d = dict(row)
                score = self._parse_effective_score(d["action"])
                if score < RECOVERY_THRESHOLD:
                    tc = d["trigger_condition"]
                    m = re.search(r"skill\s+(\S+)", tc)
                    name = m.group(1) if m else tc.replace("when considering skill ", "")
                    degraded.add(name)
        except Exception:
            pass
        finally:
            conn.close()
        return degraded

    def _parse_effective_score(self, action_text: str) -> float:
        """Extract effective score from assertion action text."""
        import re
        match = re.search(r"effective score:\s*([-\d.]+)%", action_text)
        if match:
            return float(match.group(1)) / 100.0
        return 0.5


class HealthCheckTrigger:
    """Trigger 3: Periodic scan of all skill entities.

    Runs every N consolidation cycles. Checks Entity Explorer for
    skill entities with low performance.

    Cycle count is persisted to the evolution_cycle_state table so it
    survives process restarts (H-CYCLECNT fix).

    Zero-LLM. Runs in ~20ms.
    """

    _STATE_KEY = "health_check_cycle_count"

    def __init__(self, db_path: str | Path, profile_id: str = "default"):
        self._db_path = str(db_path)
        self._profile_id = profile_id or "default"
        self._check_every_n = 3  # Every 3rd consolidation (~18h)

    def _read_cycle_count(self) -> int:
        """Read this profile's persisted cycle count."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        try:
            row = conn.execute(
                "SELECT value FROM evolution_cycle_state "
                "WHERE profile_id = ? AND key = ?",
                (self._profile_id, self._STATE_KEY),
            ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            # Table may not exist yet
            return 0
        finally:
            conn.close()

    def _write_cycle_count(self, count: int) -> None:
        """Persist this profile's cycle count.

        The evolution_cycle_state table (profile_id, key) is created by
        EvolutionStore with the correct composite-PK schema — we do NOT
        CREATE it here (the old single-key CREATE produced a conflicting
        schema and cross-profile collisions).
        """
        conn = sqlite3.connect(self._db_path, timeout=10)
        try:
            # Correct composite-PK schema (matches EvolutionStore). Created only
            # for standalone use (tests); a real deployment already has it.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evolution_cycle_state ("
                "profile_id TEXT NOT NULL DEFAULT 'default', key TEXT NOT NULL, "
                "value INTEGER DEFAULT 0, updated_at TEXT, "
                "PRIMARY KEY (profile_id, key))",
            )
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO evolution_cycle_state (profile_id, key, value, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(profile_id, key) DO UPDATE SET "
                "value=excluded.value, updated_at=excluded.updated_at",
                (self._profile_id, self._STATE_KEY, count, now),
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("Failed to persist cycle count: %s", exc)
        finally:
            conn.close()

    def should_run(self) -> bool:
        cycle_count = self._read_cycle_count() + 1
        self._write_cycle_count(cycle_count)
        return cycle_count % self._check_every_n == 0

    def scan(self, profile_id: str = "default") -> list[EvolutionCandidate]:
        if _check_memory_pressure():
            return []
        if not self.should_run():
            return []

        # Delegate to DegradationTrigger — same logic, different trigger label
        deg = DegradationTrigger(self._db_path)
        deg_candidates = deg.scan(profile_id)

        # Re-label as health_check trigger
        return [
            EvolutionCandidate(
                skill_name=c.skill_name,
                evolution_type=c.evolution_type,
                trigger=TriggerType.HEALTH_CHECK,
                evidence=c.evidence,
                effective_score=c.effective_score,
                invocation_count=c.invocation_count,
            )
            for c in deg_candidates
        ]
