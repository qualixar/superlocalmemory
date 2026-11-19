# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Pictures, documents and both model packs set themselves up on a computer with 16 GB or more.

Varun, 19 Nov 2026: "most of the users are non-technical ... it should be auto for 16 GB+. SLM should
detect the machine and then start downloading it. If it is a system with less than 16 GB, it should be
one-click download only."

The decision is ``media_env.media_policy`` (``auto`` or ``one_click``). This module is the ``auto`` half and
it runs in one place only: the daemon's background loop (``run``, started from the daemon's lifespan). It is
never called from a hook, a recall, a one-off terminal command or a test (``SLM_NO_AUTO_DOWNLOAD=1``, which the
test isolation plugin sets, switches it off).

What one ``tick`` does, in order, and it does only the first thing that applies:

1. Off switch, kill switch, or a small computer: nothing.
2. Images & documents (the managed environment): when no one turned them off, enable them (the existing
   ``features.enable_media``, which starts the install thread with its progress, cancel, disk check and
   canary) and wait for it to finish.
3. Each pack in turn (captions, then documents): when its flag is on (the default) and its model is not on
   this computer, start the existing ``media_packs.enable`` download and wait for it to finish.
   One download at a time, always.

Nothing here undoes a choice. A part someone turned off (``slm media disable``, ``slm media pack disable``, the
dashboard switch, ``SLM_FEATURE_MEDIA_*=off`` or the kill switch) reads as not enabled and is skipped for good;
an install someone cancelled is not retried. A failed download is retried later, with waits that double
(10 min, 20, 40 ... up to 6 h) and gives up for this run after 6 failures; the reason stays readable in plain
words through ``describe``. A new download never starts while the machine is short of memory or the disk
cannot hold it. Memory is given back by the existing machine-pressure release of the workers; this module
loads no model.

The auto-start is not written to ``decision_log`` (D-49): its kinds are decisions about memories (promote,
settle, label, import...) and its channels are the four doors a caller comes through; a background
default is none of those, so a row would have no honest ``kind`` or ``channel``.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The package also exports a function named ``media_env``, so the module is loaded by name.
_menv = importlib.import_module("superlocalmemory.runtimes.media_env")

logger = logging.getLogger(__name__)

#: Seconds after the daemon has started before the first look: start-up settles first (routing-only).
FIRST_DELAY_S = 180.0
#: Seconds between looks while something downloads (routing-only).
POLL_S = 20.0
#: Seconds to wait when something blocks a start (memory pressure, a busy machine) (routing-only).
WAIT_S = 300.0
RETRY_BASE_S = 600.0
RETRY_MAX_S = 6 * 3600.0
MAX_FAILURES = 6
#: The daemon must have been quiet this long before a download starts.
IDLE_S = 30.0
SOURCE = "auto"
PACK_ORDER = ("captions", "documents")

_NOUN = {"captions": "picture description", "documents": "document reading"}


@dataclass(frozen=True)
class Outcome:
    """What one look did. ``wait_s`` is how long to wait before the next look; ``None`` ends the loop."""

    label: str
    wait_s: float | None


@dataclass
class _Retry:
    failures: int = 0
    next_try: float = 0.0
    open_attempt: bool = False
    reason: str = ""


@dataclass
class AutoState:
    """What the loop remembers between looks: failures and when to try again, per part."""

    parts: dict[str, _Retry] = field(default_factory=dict)

    def part(self, key: str) -> _Retry:
        return self.parts.setdefault(key, _Retry())


_STATE = AutoState()
_STATE_LOCK = threading.RLock()  # re-entered: describing a part while a look holds it


def backoff_s(failures: int) -> float:
    """10 minutes after the first failure, doubling, never more than 6 hours."""
    return min(RETRY_MAX_S, RETRY_BASE_S * (2 ** max(0, failures - 1)))


def auto_disabled(env: Mapping[str, str] | None = None) -> bool:
    """``SLM_NO_AUTO_DOWNLOAD=1``: no automatic download in this process (tests, CI, a pinned machine)."""
    return (os.environ if env is None else env).get(_menv.NO_AUTO_DOWNLOAD_ENV) == "1"


def _root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


def _pressed() -> bool:
    try:
        from superlocalmemory.infra import memory_pressure

        return memory_pressure.pressed()
    except Exception:  # noqa: BLE001 - an unreadable gauge never blocks and never crashes the loop
        return False


def _fail(state: AutoState, key: str, now: float, reason: str) -> Outcome:
    """Count one failed attempt and schedule the next one (or give up for this run)."""
    part = state.part(key)
    part.failures += 1
    part.open_attempt = False
    part.reason = reason
    part.next_try = now + backoff_s(part.failures)
    logger.warning("automatic download of %s failed (%d): %s", key, part.failures, reason)
    if part.failures >= MAX_FAILURES:
        return Outcome("gave_up", None)
    return Outcome("retry_later", min(WAIT_S, part.next_try - now))


def _may_start(state: AutoState, key: str, now: float) -> Outcome | None:
    """A blocking outcome when this part must not start now (waiting out a back-off, giving up)."""
    part = state.part(key)
    if part.failures >= MAX_FAILURES:
        return Outcome("gave_up", None)
    if part.failures and now < part.next_try:
        return Outcome("retry_later", min(WAIT_S, part.next_try - now))
    return None


def _media_step(root: Path, env: Any, state: AutoState, now: float,
                enable_media: Callable[..., dict[str, Any]]) -> Outcome | None:
    """Images and documents: ``None`` when the environment is ready, else what this look did."""
    from superlocalmemory.runtimes import features as feat
    from superlocalmemory.runtimes.managed_env import FAILED, INSTALLING, READY, UNSUPPORTED

    if feat.media_opted_out(root):
        return Outcome("turned_off", WAIT_S)
    managed = feat._env(env, root)
    status = managed.status()
    enabled = feat.media_enabled(root)
    part = state.part("media")
    if enabled and status.state == READY:
        part.failures, part.open_attempt = 0, False
        return None
    if status.state == UNSUPPORTED:
        part.reason = status.step
        return Outcome("unsupported", None)
    if enabled and status.state == INSTALLING:
        part.open_attempt = True
        return Outcome("downloading_media", POLL_S)
    if enabled and status.state == FAILED:
        if status.error_kind == "cancelled":
            return Outcome("stopped_by_you", WAIT_S)
        if part.open_attempt:
            return _fail(state, "media", now, status.step or "The set-up did not finish.")
    blocked = _may_start(state, "media", now)
    if blocked is not None:
        return blocked
    if _pressed():
        return Outcome("waiting_memory", WAIT_S)
    if not managed.precheck().get("disk_ok", True):
        return _fail(state, "media", now, "There is not enough free disk space for pictures and documents.")
    result = enable_media(source=SOURCE, env=env, data_root=root)
    part.open_attempt = True
    if result.get("error"):
        return _fail(state, "media", now, str(result["error"]))
    return Outcome("starting_media", POLL_S)


def _pack_step(root: Path, env: Any, state: AutoState, now: float, pack_id: str,
               enable_pack: Callable[..., dict[str, Any]]) -> Outcome | None:
    """One pack: ``None`` when it needs nothing (installed, or turned off), else what this look did."""
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.runtimes import media_packs

    spec = media_packs.get(pack_id)
    if not FeatureFlags(root).enabled(spec.flag_id):
        return None  # turned off by a person or an agent, or the kill switch: never undone here
    if media_packs.installed(spec, root):
        state.part(pack_id).failures = 0
        return None
    part = state.part(pack_id)
    job = media_packs.status(pack_id, root, env=env)
    if job["state"] == "downloading":
        part.open_attempt = True
        return Outcome("downloading_pack", POLL_S)
    if job["state"] == "failed" and part.open_attempt:
        return _fail(state, pack_id, now, job["error"] or "The download did not finish.")
    blocked = _may_start(state, pack_id, now)
    if blocked is not None:
        return blocked
    if _pressed():
        return Outcome("waiting_memory", WAIT_S)
    result = enable_pack(pack_id, source=SOURCE, data_root=root, env=env)
    part.open_attempt = True
    if result.get("refused"):
        return _fail(state, pack_id, now, str(result["refused"]))
    return Outcome("starting_pack", POLL_S)


def tick(data_root: str | Path | None = None, *, ram_bytes: int | None = None, env: Any = None,
         state: AutoState | None = None, now: float | None = None,
         environ: Mapping[str, str] | None = None,
         enable_media: Callable[..., dict[str, Any]] | None = None,
         enable_pack: Callable[..., dict[str, Any]] | None = None) -> Outcome:
    """One look. Does the first thing that applies and returns what it did; never raises."""
    try:
        if auto_disabled(environ):
            return Outcome("disabled", None)
        root = _root(data_root)
        from superlocalmemory.core.feature_flags import FeatureFlags
        from superlocalmemory.runtimes import features as feat
        from superlocalmemory.runtimes import managed_env, media_packs

        if FeatureFlags(root, environ=environ).killed:
            return Outcome("killed", WAIT_S)
        ram = managed_env._ram_bytes() if ram_bytes is None else ram_bytes
        if _menv.media_policy(ram, environ).mode != _menv.AUTO:
            return Outcome("one_click", None)
        state = _STATE if state is None else state
        moment = time.monotonic() if now is None else now
        with _STATE_LOCK:
            step = _media_step(root, env, state, moment, enable_media or feat.enable_media)
            if step is not None:
                return step
            for pack_id in PACK_ORDER:
                step = _pack_step(root, env, state, moment, pack_id, enable_pack or media_packs.enable)
                if step is not None:
                    return step
        return Outcome("done", None)
    except Exception:  # noqa: BLE001 - a background default never takes the daemon down
        logger.warning("automatic set-up of pictures and documents hit an error", exc_info=True)
        return Outcome("error", WAIT_S)


async def run(data_root: str | Path | None = None, *, first_delay_s: float = FIRST_DELAY_S,
              idle: Callable[[], bool] | None = None,
              tick_fn: Callable[..., Outcome] = tick,
              sleep: Callable[[float], Any] = asyncio.sleep) -> Outcome:
    """The daemon's loop: wait for start-up to settle, then look, and look again until nothing is left.

    Returns the last outcome (the loop ends when nothing more can happen in this run). Cancelled with the daemon.
    """
    if auto_disabled():
        return Outcome("disabled", None)
    await sleep(first_delay_s)
    while True:
        if idle is not None and not idle():
            await sleep(POLL_S)
            continue
        outcome = await asyncio.to_thread(tick_fn, data_root)
        logger.info("automatic set-up of pictures and documents: %s", outcome.label)
        if outcome.wait_s is None:
            return outcome
        await sleep(outcome.wait_s)


# -- plain words for every surface ---------------------------------------------------------------

def _percent(progress: float) -> int:
    return max(0, min(100, int(round(float(progress or 0.0) * 100))))


def last_reason(key: str) -> str:
    """Why the last automatic attempt of one part failed, in plain words ("" when it did not)."""
    with _STATE_LOCK:
        part = _STATE.parts.get(key)
        return part.reason if part and part.failures else ""


def _pending_words(mode: str, what: str, size_words: str = "") -> str:
    if mode == _menv.AUTO:
        return f"Waiting its turn: the {what} model downloads by itself when your computer is idle."
    size = f" ({size_words})" if size_words else ""
    return f"One click to download the {what} model{size}."


def pack_words(pack: Mapping[str, Any], mode: str) -> str:
    """The one sentence a person reads about a part: ``Downloading the picture description model, 40%``."""
    what = _NOUN.get(str(pack.get("pack")), "optional")
    state = str(pack.get("state"))
    if pack.get("killed"):
        return "Switched off with the kill switch."
    if state == "downloading":
        return f"Downloading the {what} model, {_percent(pack.get('progress', 0.0))}%"
    if state == "failed":
        retry = " It will try again later." if mode == _menv.AUTO else ""
        return f"{pack.get('error') or 'The download did not finish.'}{retry}"
    if pack.get("installed") and pack.get("enabled"):
        return "Ready"
    if not pack.get("enabled"):
        return "Turned off by you"
    reason = last_reason(str(pack.get("pack")))
    if reason and mode == _menv.AUTO:
        return f"{reason} It will try again later."
    return _pending_words(mode, what, str(pack.get("size_words", "")))


def media_words(view: Mapping[str, Any], mode: str, *, opted_out: bool = False) -> str:
    """The same for images and documents (``view``: ``enabled``, ``env_state``, ``progress``)."""
    state = str(view.get("env_state", "not_installed"))
    if opted_out and not view.get("enabled"):
        return "Turned off by you"
    if state == "installing":
        return f"Downloading the picture model, {_percent(view.get('progress', 0.0))}%"
    if state == "ready" and view.get("enabled"):
        return "Ready"
    if state == "unsupported":
        return str(view.get("step") or "Not available on this computer.")
    if state == "failed":
        retry = " It will try again later." if mode == _menv.AUTO else ""
        return f"{view.get('step') or 'The set-up did not finish.'}{retry}"
    return _pending_words(mode, "picture", "")


def describe(data_root: str | Path | None = None, *, env: Any = None,
             ram_bytes: int | None = None) -> dict[str, Any]:
    """Everything a surface needs: the mode, the small-computer warning and one sentence per part. Reads only."""
    from superlocalmemory.runtimes import features as feat
    from superlocalmemory.runtimes import managed_env, media_packs

    root = _root(data_root)
    policy = _menv.media_policy(managed_env._ram_bytes() if ram_bytes is None else ram_bytes)
    managed = feat._env(env, root)
    st = managed.status()
    view = {"enabled": feat.media_enabled(root), "env_state": st.state, "progress": st.progress,
            "step": st.step}
    parts = [{"pack": p["pack"], "state": p["state"], "words": pack_words(p, policy.mode)}
             for p in (media_packs.status(pid, root, env=env) for pid in PACK_ORDER)]
    return {"mode": policy.mode, "warning": policy.warning, "auto_disabled": auto_disabled(),
            "media": {"state": st.state, "words": media_words(view, policy.mode,
                                                              opted_out=feat.media_opted_out(root))},
            "parts": parts}


__all__ = ["AutoState", "FIRST_DELAY_S", "MAX_FAILURES", "Outcome", "PACK_ORDER", "auto_disabled",
           "backoff_s", "describe", "last_reason", "media_words", "pack_words", "run", "tick"]
