# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

""""Upgrade memory engine": move an existing store onto the managed model, on request.

The work itself is the ordinary background switch (``ReindexRunner.request_switch``):
recall keeps working on the current model, the new vectors are built next to the old
ones, both change in one step, and the previous vectors stay until they are freed, so
one click goes back. This module only defines the single target, says honestly whether
the upgrade can start right now and what it will cost, and picks up a request the
installer recorded. Text shown to people avoids the word "re-index".
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from superlocalmemory.runtimes import media_models

logger = logging.getLogger(__name__)

PROVIDER = "slm-media"
TURN_ON_COMMAND = "slm media enable"
_BYTES_PER_FLOAT = 4
_MB = 1024 ** 2

EXPLAIN = ("Your memories are re-read with the new engine in the background. Recall keeps working "
           "on the current engine until it finishes, nothing is deleted, and you can roll back "
           "to the previous engine afterwards.")

_STATE_REASONS = {
    "installing": "Images and documents are still being set up. Try again when they finish "
                  "(check with: slm media status).",
    "failed": "Images and documents did not finish setting up, so the new engine is not ready. "
              "See: slm doctor",
    "unsupported": "Images and documents can't be set up on this computer yet, so the new engine "
                   "is not available here.",
}
_NOT_SET_UP = f"Images and documents are not set up yet. Run: {TURN_ON_COMMAND}"
_OFF = (f"Turn on images and documents first (about 1.5 GB): {TURN_ON_COMMAND}")
_ALREADY = "Your memories already use the new engine."
_RUNNING = ("Another change to the memory engine is already running. Wait for it to finish, "
            "or stop it with: slm embedder cancel")


def upgrade_target(live: Any) -> Any:
    """The embedding config an upgrade moves to: the managed model, on the live config's other settings."""
    profile = media_models.profile_for(media_models.EG2_REPO)
    dim = profile.dim if profile is not None else 768
    return replace(live, provider=PROVIDER, model_name=media_models.EG2_REPO, dimension=dim,
                   api_endpoint="", api_key="")


def _view(cfg: Any) -> dict[str, Any]:
    return {"provider": cfg.provider, "model": cfg.model_name, "dimension": int(cfg.dimension)}


def _mb(memories: int, dimension: int) -> int:
    return math.ceil(memories * dimension * _BYTES_PER_FLOAT / _MB)


def _minutes(memories: int) -> int:
    if memories <= 0:
        return 0
    return max(1, round(memories * media_models.PROVISIONAL_SECONDS_PER_MEMORY / 60))


def _minutes_label(minutes: int) -> str:
    """The time is an estimate (PROVISIONAL_SECONDS_PER_MEMORY, not measured on this machine); say so."""
    if minutes <= 0:
        return "about a moment"
    return f"about {minutes} minute" + ("" if minutes == 1 else "s") + " (a rough estimate)"


def _reason(already: bool, job_running: bool, media_enabled: bool, env_state: str) -> str:
    if already:
        return _ALREADY
    if job_running:
        return _RUNNING
    if not media_enabled:
        return _OFF
    if env_state == "ready":
        return ""
    return _STATE_REASONS.get(env_state, _NOT_SET_UP)


def plan(live: Any, fact_count: int, env_state: str, *, media_enabled: bool = True,
         job_running: bool = False) -> dict[str, Any]:
    """What an upgrade would do, whether it can start now, and why not when it cannot."""
    target = upgrade_target(live)
    already = live.provider == PROVIDER and live.model_name == target.model_name
    reason = _reason(already, job_running, media_enabled, env_state)
    memories = max(int(fact_count), 0)
    new_mb, kept_mb = _mb(memories, target.dimension), _mb(memories, live.dimension)
    minutes = _minutes(memories)
    return {
        "available": reason == "", "reason": reason, "already": already,
        "needs_media": not already and not job_running and not media_enabled,
        "turn_on_command": TURN_ON_COMMAND if not already and not job_running and not media_enabled else "",
        "from": _view(live), "to": _view(target), "memories": memories,
        "ram_mb": media_models.load_mb_for(target.model_name),
        "disk_new_mb": new_mb, "disk_kept_mb": kept_mb, "disk_mb": new_mb + kept_mb,
        "minutes": minutes, "minutes_label": _minutes_label(minutes),
        "media_enabled": bool(media_enabled), "env_state": env_state,
        "explain": EXPLAIN, "rollback": True, "label": "upgrade",
    }


# -- facts the plan needs ------------------------------------------------------

def media_state(data_root: Any = None, env: Any = None) -> tuple[bool, str]:
    """``(images and documents are on, state of their environment)``; never raises."""
    from superlocalmemory.runtimes import features

    try:
        status = features.media_feature_status(data_root, env=env)
        return bool(status.get("enabled")), str((status.get("env") or {}).get("state", "not_installed"))
    except Exception:  # noqa: BLE001 - an unreadable state is "not set up", never an error page
        logger.warning("could not read the images and documents state", exc_info=True)
        return False, "not_installed"


def count_memories(db_path: Any) -> int:
    from superlocalmemory.storage import embedding_spaces as sp
    from superlocalmemory.storage.embedding_space_swap import count_facts

    conn = sp.connect(Path(db_path))
    try:
        return count_facts(conn)
    finally:
        conn.close()


def job_is_running(db_path: Any) -> bool:
    from superlocalmemory.storage import embedding_spaces as sp
    from superlocalmemory.storage.embedding_reindex_jobs import active_job

    conn = sp.connect(Path(db_path))
    try:
        return active_job(conn) is not None
    finally:
        conn.close()


def current_plan(live: Any, db_path: Any, *, data_root: Any = None, env: Any = None) -> dict[str, Any]:
    """The plan for this store as it is right now."""
    enabled, state = media_state(data_root, env)
    return plan(live, count_memories(db_path), state, media_enabled=enabled,
                job_running=job_is_running(db_path))


# -- a request the installer recorded ------------------------------------------

def apply_on_start(app_state: Any, config: Any, *, data_root: Any = None, env: Any = None) -> str:
    """At daemon start: act once on a recorded upgrade request. Never raises.

    ``queued`` (switch started, request cleared), ``cleared`` (nothing left to do),
    ``waiting`` (try again at the next start; the request stays), ``none`` (no request).
    """
    from superlocalmemory.runtimes import features

    try:
        if not features.engine_upgrade_requested(data_root):
            return "none"
        runner = getattr(app_state, "embedding_reindex", None)
        live = config.embedding
        if live.provider == PROVIDER:
            features.clear_engine_upgrade_request(data_root)
            return "cleared"
        if runner is None:
            return "waiting"
        current = current_plan(live, config.db_path, data_root=data_root, env=env)
        if not current["available"]:
            return "waiting"
        runner.request_switch(upgrade_target(live))
        features.clear_engine_upgrade_request(data_root)
        logger.info("memory engine upgrade queued from the recorded request")
        return "queued"
    except Exception:  # noqa: BLE001 - start-up must never fail on an optional upgrade
        logger.warning("could not act on the recorded memory engine upgrade request", exc_info=True)
        return "waiting"


__all__ = ["EXPLAIN", "PROVIDER", "TURN_ON_COMMAND", "apply_on_start", "count_memories", "current_plan",
           "job_is_running", "media_state", "plan", "upgrade_target"]
