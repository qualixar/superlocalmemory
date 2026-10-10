# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The switch for optional features, starting with images and documents.

State lives in its own file, ``<data_root>/features.json``, written only when
someone first turns something on: a person who never does sees no new file.
Reads never create anything.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from superlocalmemory.runtimes.managed_env import ManagedEnv

logger = logging.getLogger(__name__)

FEATURES_FILE = "features.json"
SOURCES = ("cli", "dashboard", "npm", "api")
INSTALL_THREAD_NAME = "media-env-install"

_stop_hook: Callable[[], None] | None = None
_install_thread: threading.Thread | None = None
_lock = threading.Lock()
_cancel = threading.Event()
#: How long turning the feature off waits for a cancelled install to stop.
_STOP_WAIT_S = 15.0


def _root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


def features_path(data_root: str | Path | None = None) -> Path:
    return _root(data_root) / FEATURES_FILE


def _defaults() -> dict[str, Any]:
    off = {"enabled": False, "enabled_at": None, "choice_source": None}
    return {"schema": 1, "media": dict(off), "sources": dict(off)}


def read_features(data_root: str | Path | None = None) -> dict[str, Any]:
    """The saved switches, or the defaults when the file is absent or unreadable."""
    path = features_path(data_root)
    data = _defaults()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for key in ("media", "sources"):
                if isinstance(raw.get(key), dict):
                    data[key].update(raw[key])
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        logger.warning("features file is unreadable; using defaults")
    return data


def media_requested(data_root: str | Path | None = None) -> bool:
    """True when the installer recorded a request that no one has acted on yet."""
    media = read_features(data_root)["media"]
    return bool(media.get("requested")) and not media.get("enabled")


def media_enabled(data_root: str | Path | None = None) -> bool:
    return bool(read_features(data_root)["media"].get("enabled"))


def sources_enabled(data_root: str | Path | None = None) -> bool:
    """Folder sources have their own switch, off until a folder is confirmed."""
    return bool(read_features(data_root)["sources"].get("enabled"))


def _write_features(data_root: str | Path | None, data: dict[str, Any]) -> None:
    path = features_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    if sys.platform != "win32":
        os.chmod(path, 0o600)


def _env(env: ManagedEnv | None, data_root: str | Path | None) -> ManagedEnv:
    if env is not None:
        return env
    from superlocalmemory.runtimes.media_env import media_env

    return media_env(root=_root(data_root) / "runtimes" / "media" if data_root is not None else None)


def media_feature_status(data_root: str | Path | None = None, *, env: ManagedEnv | None = None) -> dict[str, Any]:
    """Read-only summary for the turn-on surfaces and ``slm doctor``; creates nothing."""
    from superlocalmemory.media import media_db_exists

    managed = _env(env, data_root)
    status = managed.status()
    return {"enabled": media_enabled(data_root),
            "env": status.to_dict() if hasattr(status, "to_dict") else dict(vars(status)),
            "precheck": managed.precheck(), "media_db": media_db_exists(_root(data_root))}


def register_media_stop_hook(fn: Callable[[], None] | None) -> None:
    """The media worker registers how to stop itself; called when the feature is turned off."""
    global _stop_hook
    _stop_hook = fn


def _start_install(managed: ManagedEnv) -> None:
    global _install_thread
    with _lock:
        if _install_thread is not None and _install_thread.is_alive():
            return
        if managed.status().state in ("ready", "installing"):
            return
        _cancel.clear()
        _install_thread = threading.Thread(target=lambda: managed.install(cancel=_cancel),
                                           name=INSTALL_THREAD_NAME, daemon=True)
        _install_thread.start()


_NO_EXTENSIONS = ("Images and documents need a Python that can load SQLite extensions, and this one "
                  "can't. Run SLM on a Python built with them (Homebrew, python.org or uv-managed).")


def enable_media(*, source: str, start_install: bool = True, env: ManagedEnv | None = None,
                 data_root: str | Path | None = None) -> dict[str, Any]:
    """Turn images and documents on: save the choice, create media.db, start the install."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    from superlocalmemory.media import MediaVectorsUnavailable, open_media_store

    try:
        data = read_features(data_root)
        data["media"] = {"enabled": True, "enabled_at": datetime.now(timezone.utc).isoformat(),
                         "choice_source": source}
        _write_features(data_root, data)
        store = open_media_store(create=True, data_root=_root(data_root))
        if store is not None:
            store.close()
    except MediaVectorsUnavailable as exc:
        logger.warning("could not turn on images and documents: %s", exc)
        _roll_back_enable(data_root)
        return {**media_feature_status(data_root, env=env), "enabled": False,
                "error": _NO_EXTENSIONS}
    except (OSError, sqlite3.Error, ImportError) as exc:
        logger.warning("could not turn on images and documents: %s", exc)
        _roll_back_enable(data_root)
        return {**media_feature_status(data_root, env=env), "enabled": False,
                "error": "Couldn't save the setting. Check that the data folder is writable."}
    managed = _env(env, data_root)
    if start_install:
        _start_install(managed)
    return media_feature_status(data_root, env=managed)


def _roll_back_enable(data_root: str | Path | None) -> None:
    try:
        data = read_features(data_root)
        data["media"]["enabled"] = False
        _write_features(data_root, data)
    except OSError:
        pass


def _stop_install() -> None:
    """Ask a running install to stop and give it a moment to do so."""
    _cancel.set()
    thread = _install_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(_STOP_WAIT_S)


def disable_media(*, remove_files: bool = False, env: ManagedEnv | None = None,
                  data_root: str | Path | None = None) -> dict[str, Any]:
    """Turn the feature off. Memories and media.db are kept; model files only if asked."""
    data = read_features(data_root)
    data["media"]["enabled"] = False
    data["media"].pop("requested", None)  # turning it off also withdraws an unanswered request
    try:
        _write_features(data_root, data)
    except OSError as exc:
        logger.warning("could not save the switch: %s", exc)
    if _stop_hook is not None:
        try:
            _stop_hook()
        except Exception:  # noqa: BLE001 - turning off must not fail on a worker that is already gone
            logger.exception("media stop hook failed")
    _stop_install()
    managed = _env(env, data_root)
    if remove_files:
        managed.remove(keep_weights=False)
    return media_feature_status(data_root, env=managed)


def enable_sources(*, source: str, data_root: str | Path | None = None) -> bool:
    """Turn folder sources on and create media.db (where the folder tables live).

    Called only when a person confirms a folder; returns False when it could not be saved.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    from superlocalmemory.media import open_media_store

    try:
        data = read_features(data_root)
        data["sources"] = {"enabled": True, "enabled_at": datetime.now(timezone.utc).isoformat(),
                           "choice_source": source}
        _write_features(data_root, data)
        store = open_media_store(create=True, data_root=_root(data_root))
        if store is not None:
            store.close()
    except (OSError, sqlite3.Error, ImportError) as exc:
        logger.warning("could not turn on folder sources: %s", exc)
        return False
    return True


def disable_sources(*, data_root: str | Path | None = None) -> None:
    """Turn folder sources off. Nothing already saved is removed."""
    if not features_path(data_root).exists():
        return
    data = read_features(data_root)
    data["sources"]["enabled"] = False
    try:
        _write_features(data_root, data)
    except OSError as exc:
        logger.warning("could not save the switch: %s", exc)


def apply_requested(*, source: str = "npm", env: ManagedEnv | None = None,
                    data_root: str | Path | None = None) -> dict[str, Any] | None:
    """Act once on a request the installer recorded; the install runs in this process.

    Does nothing (and returns ``None``) when nothing was requested, when the
    feature is already on, or when the file cannot be read. Never raises.
    """
    try:
        if not media_requested(data_root):
            return None
        return enable_media(source=source, env=env, data_root=data_root)
    except Exception:  # noqa: BLE001 - start-up must never fail on an optional feature
        logger.warning("could not act on the saved request for images and documents", exc_info=True)
        return None


# -- has this process picked the feature up? ----------------------------------
_media_loaded = threading.Event()


def mark_media_loaded() -> None:
    """The media components started in this process (called by whoever starts them)."""
    _media_loaded.set()


def _reset_media_loaded() -> None:
    _media_loaded.clear()


def note_started(*, env: ManagedEnv | None = None, data_root: str | Path | None = None) -> None:
    """At daemon start: media on and its environment already ready means the media
    components load from it in this process, so no restart is needed. Never raises."""
    try:
        if media_enabled(data_root) and _env(env, data_root).status().state == "ready":
            mark_media_loaded()
    except Exception:  # noqa: BLE001 - start-up must never fail on an optional feature
        logger.warning("could not check the images and documents environment", exc_info=True)


def restart_required(status: dict[str, Any]) -> bool:
    """On and ready, but this running process has not loaded the media components."""
    return bool(status.get("enabled")) and (status.get("env") or {}).get("state") == "ready" \
        and not _media_loaded.is_set()
