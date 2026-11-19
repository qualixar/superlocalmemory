# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The optional multimodal pack: which model each part needs, and turning a part on or off (WS-8, D-19).

Two parts, each off by default and each with its own explicit download:

* ``captions``: ``florence-community/Florence-2-base`` (MIT) writes a short description of a picture on
  every platform (flag ``media.captions``).
* ``documents``: ``PaddlePaddle/PaddleOCR-VL-1.6`` (Apache-2.0) reads scanned document pages
  (flag ``media.document_pack``).

Both load through the native classes of the ``transformers`` version the media environment already pins
(no new runtime, no ``trust_remote_code``). The repository and revision of each are data below; the sizes
are the true pinned sizes from the Hugging Face API on 18 Nov 2026 (routing only: they are shown to a
person before anything is downloaded).

A model is downloaded only when a person (or an agent acting for one) turns the part on, through the same
managed-environment source the picture model uses. A flag alone never downloads, and nothing here is called
from a hook or a recall. The kill switch turns both parts off at once.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Seconds a pack worker stays loaded after its last request, then its memory goes back (routing-only).
PACK_IDLE_S = 60.0
#: Free disk wanted, as a multiple of the download, while it is being unpacked (routing-only).
DISK_HEADROOM = 2.0
READY_FILE = "ready.json"

KILLED_WORDS = ("The 4.2.30 features are switched off (the kill switch is on), so this cannot be turned on. "
                "Switch them back on first.")
NEEDS_MEDIA_WORDS = "Turn on Images & documents first (slm media enable); this part adds to it."


@dataclass(frozen=True)
class PackSpec:
    pack_id: str
    flag_id: str
    role: str                 # the worker role that loads it
    repo: str
    revision: str             # a commit on the hub, never a branch name
    download_bytes: int       # the true size of the pinned revision, from the hub API
    verb: str                 # the words on the button: "Describe pictures"
    what: str                 # one plain sentence for a person
    native_class: str         # the transformers class the worker loads it with (not trust_remote_code)
    licence: str


CAPTIONS = PackSpec(
    "captions", "media.captions", "caption", "florence-community/Florence-2-base",
    "00921df66db728a9ceb750f5eca43e5c203a2051", 468_569_086, "Describe pictures",
    "Writes a short description of each picture you save, so a picture can be found by what it shows. "
    "The description is marked as written by a model.",
    "Florence2ForConditionalGeneration", "MIT")
DOCUMENTS = PackSpec(
    "documents", "media.document_pack", "document", "PaddlePaddle/PaddleOCR-VL-1.6",
    "132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd", 1_930_469_491, "Read scanned documents better",
    "Reads scanned pages with a larger model. The standard reader stays as the fallback.",
    "PaddleOCRVLForConditionalGeneration", "Apache-2.0")

PACKS: Mapping[str, PackSpec] = MappingProxyType({p.pack_id: p for p in (CAPTIONS, DOCUMENTS)})

#: The third part has no model: it only keeps the page and the box each picture text came from
#: (flag ``media.regions``), so it is a switch with nothing to download.
REGIONS_ID = "regions"
REGIONS_FLAG = "media.regions"
REGIONS_LABEL = "Remember where text was found"
REGIONS_WHAT = ("Keeps the page and place each piece of text came from, so a memory can say \"from page 3, top "
                "left\". Nothing is downloaded.")
PART_NAMES = (*PACKS, REGIONS_ID)


def get(pack_id: str) -> PackSpec:
    try:
        return PACKS[pack_id]
    except KeyError:
        raise ValueError(f"unknown pack {pack_id!r}; choose one of: {', '.join(PACKS)}") from None


def size_words(n_bytes: int) -> str:
    """``0.5 GB`` (decimal, one place, like the hub and the file manager)."""
    return f"{n_bytes / 1e9:.1f} GB"


def download_words(spec: PackSpec) -> str:
    return f"{spec.verb}: a {size_words(spec.download_bytes)} download"


def _root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


def pack_dir(spec: PackSpec, data_root: str | Path | None = None) -> Path:
    return _root(data_root) / "runtimes" / "media" / "packs" / spec.pack_id


def model_dir(spec: PackSpec, data_root: str | Path | None = None) -> Path:
    return pack_dir(spec, data_root) / "model"


def installed(spec: PackSpec, data_root: str | Path | None = None) -> bool:
    """True when this pinned revision was downloaded completely (the marker is written last)."""
    try:
        info = json.loads((pack_dir(spec, data_root) / READY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(info, dict) and info.get("repo") == spec.repo and info.get("revision") == spec.revision
            and model_dir(spec, data_root).is_dir())


def _flags(data_root: str | Path | None):
    from superlocalmemory.core.feature_flags import FeatureFlags

    return FeatureFlags(_root(data_root))


def is_active(pack_id: str, data_root: str | Path | None = None) -> bool:
    """The part is on, the kill switch is off and its model is on this computer. Never raises."""
    try:
        spec = get(pack_id)
        return _flags(data_root).enabled(spec.flag_id) and installed(spec, data_root)
    except Exception:  # noqa: BLE001 - a question about an optional part is "no", never an error
        return False


def flag_enabled(flag_id: str, data_root: str | Path | None = None) -> bool:
    """One of the pack flags is on (and the kill switch is off). Never raises."""
    try:
        return _flags(data_root).enabled(flag_id)
    except Exception:  # noqa: BLE001
        return False


# -- the download ------------------------------------------------------------------

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_CANCELS: dict[str, threading.Event] = {}
_THREADS: dict[str, threading.Thread] = {}


def _job(pack_id: str) -> dict[str, Any]:
    with _LOCK:
        return dict(_JOBS.get(pack_id) or {})


def _set_job(pack_id: str, **fields: Any) -> None:
    with _LOCK:
        _JOBS[pack_id] = {**_JOBS.get(pack_id, {}), **fields}


def _managed(data_root: Path, env: Any):
    if env is not None:
        return env
    from superlocalmemory.runtimes.media_env import media_env

    return media_env(root=data_root / "runtimes" / "media")


def _media_ready(root: Path, env: Any) -> bool:
    try:
        from superlocalmemory.runtimes import features as feat

        return feat.media_enabled(root) and _managed(root, env).status().state == "ready"
    except Exception:  # noqa: BLE001 - an unreadable environment is "not ready"
        logger.debug("media environment state unreadable", exc_info=True)
        return False


def _regions_status(root: Path, env: Any) -> dict[str, Any]:
    flags = _flags(root)
    return {"pack": REGIONS_ID, "label": REGIONS_LABEL, "what": REGIONS_WHAT,
            "state": "on" if flags.enabled(REGIONS_FLAG) else "off", "enabled": flags.enabled(REGIONS_FLAG),
            "installed": True, "killed": flags.killed, "size_bytes": 0, "size_words": "0 GB",
            "download_words": "No download", "licence": "", "model": "", "revision": "", "progress": 1.0,
            "step": "", "error": "", "media_ready": _media_ready(root, env)}


def _with_words(view: dict[str, Any]) -> dict[str, Any]:
    """Add ``mode`` (``auto`` / ``one_click``), the small-computer ``warning`` and one plain sentence,
    ``words``, so every surface (dashboard, ``slm media pack status``, MCP) says the same thing."""
    try:
        from superlocalmemory.runtimes import media_auto
        from superlocalmemory.runtimes.media_env import current_media_policy

        policy = current_media_policy()
        if view.get("pack") == REGIONS_ID:
            words = ("Switched off with the kill switch." if view.get("killed") else (
                "Ready" if view.get("enabled") else "Turned off by you"))
        else:
            words = media_auto.pack_words(view, policy.mode)
        return {**view, "mode": policy.mode, "warning": policy.warning, "words": words}
    except Exception:  # noqa: BLE001 - words are a courtesy; the state itself must still be served
        logger.debug("could not word the pack status", exc_info=True)
        return {**view, "mode": "one_click", "warning": "", "words": ""}


def status(pack_id: str, data_root: str | Path | None = None, *, env: Any = None) -> dict[str, Any]:
    """Where one part stands, in words a person can read. Reads only; starts and creates nothing."""
    return _with_words(_status(pack_id, data_root, env=env))


def _status(pack_id: str, data_root: str | Path | None = None, *, env: Any = None) -> dict[str, Any]:
    if pack_id == REGIONS_ID:
        return _regions_status(_root(data_root), env)
    spec, root = get(pack_id), _root(data_root)
    flags = _flags(root)
    job = _job(pack_id)
    have = installed(spec, root)
    state = "downloading" if job.get("state") == "downloading" else (
        "failed" if job.get("state") == "failed" else ("on" if flags.enabled(spec.flag_id) and have else (
            "ready" if have else "off")))
    media_ready = _media_ready(root, env)
    return {
        "pack": spec.pack_id, "label": spec.verb, "what": spec.what, "state": state,
        "enabled": flags.enabled(spec.flag_id), "installed": have, "killed": flags.killed,
        "size_bytes": spec.download_bytes, "size_words": size_words(spec.download_bytes),
        "download_words": download_words(spec), "licence": spec.licence,
        "model": spec.repo, "revision": spec.revision,
        "progress": float(job.get("progress", 1.0 if have else 0.0)), "step": str(job.get("step", "")),
        "error": str(job.get("error", "")) if state == "failed" else "", "media_ready": media_ready,
    }


def status_all(data_root: str | Path | None = None, *, env: Any = None) -> list[dict[str, Any]]:
    return [status(p, data_root, env=env) for p in PART_NAMES]


def _refusal(spec: PackSpec, root: Path, env: Any) -> str:
    """Why this part cannot be turned on now, or ``""``."""
    if _flags(root).killed:
        return KILLED_WORDS
    from superlocalmemory.runtimes import features as feat

    if not feat.media_enabled(root) or _managed(root, env).status().state != "ready":
        return NEEDS_MEDIA_WORDS
    if not installed(spec, root):
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        if free < spec.download_bytes * DISK_HEADROOM:
            need = size_words(int(spec.download_bytes * DISK_HEADROOM))
            return f"Not enough free disk space: this needs about {need} free and {size_words(free)} is free."
    return ""


def _write_flag(spec: PackSpec, root: Path, on: bool) -> None:
    from superlocalmemory.core.feature_flags import FlagState, write_flag

    root.mkdir(parents=True, exist_ok=True)
    write_flag(root, spec.flag_id, FlagState.ON if on else FlagState.OFF)


def _download(spec: PackSpec, root: Path, env: Any, source: Any, cancel: threading.Event) -> None:
    pid = spec.pack_id
    partial = pack_dir(spec, root) / "model.partial"
    final = model_dir(spec, root)

    def progress(fraction: float, step: str) -> None:
        _set_job(pid, progress=max(0.0, min(float(fraction), 1.0)), step=step)

    try:
        pack_dir(spec, root).mkdir(parents=True, exist_ok=True)
        python = _managed(root, env).python()
        ok, kind, detail = source.fetch(python, partial, progress=progress, cancel=cancel)
    except Exception as exc:  # noqa: BLE001 - the thread must end in a state a person can read
        logger.warning("pack %s download failed (%s)", pid, type(exc).__name__)
        _set_job(pid, state="failed", error="The download did not finish. Check your connection and try again.")
        return
    if not ok:
        logger.warning("pack %s download stopped: %s %s", pid, kind, str(detail)[:200])
        words = {"cancelled": "The download was stopped.", "disk": "The disk is full."}.get(
            kind, "The download did not finish. Check your connection and try again.")
        _set_job(pid, state="failed", error=words)
        return
    try:
        shutil.rmtree(final, ignore_errors=True)
        partial.replace(final)
        (pack_dir(spec, root) / READY_FILE).write_text(json.dumps({
            "repo": spec.repo, "revision": spec.revision, "bytes": spec.download_bytes,
            "installed_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        _write_flag(spec, root, True)
    except OSError as exc:
        logger.warning("pack %s could not be put in place (%s)", pid, type(exc).__name__)
        _set_job(pid, state="failed", error="The files could not be saved. Check the data folder is writable.")
        return
    _set_job(pid, state="done", progress=1.0, step="Ready")


def _switch_regions(on: bool, root: Path, env: Any) -> dict[str, Any]:
    from superlocalmemory.core.feature_flags import FlagState, write_flag

    if on:
        reason = KILLED_WORDS if _flags(root).killed else ("" if _media_ready(root, env) else NEEDS_MEDIA_WORDS)
        if reason:
            return {**_regions_status(root, env), "refused": reason}
    root.mkdir(parents=True, exist_ok=True)
    write_flag(root, REGIONS_FLAG, FlagState.ON if on else FlagState.OFF)
    return _regions_status(root, env)


def enable(pack_id: str, *, source: str = "api", data_root: str | Path | None = None, env: Any = None,
           model_source: Any = None, background: bool = True) -> dict[str, Any]:
    """Turn one part on. Installed already: just switch it on. Otherwise download it (the size was shown
    first by the caller) in the background and switch it on when it is complete.

    Refused, with a plain reason in ``refused``, when the kill switch is on, when images and documents are
    not on, when the computer is too small, or when there is not enough disk space; nothing is written then.
    """
    if pack_id == REGIONS_ID:
        return _with_words(_switch_regions(True, _root(data_root), env))
    spec, root = get(pack_id), _root(data_root)
    reason = _refusal(spec, root, env)
    if reason:
        return {**status(pack_id, root, env=env), "refused": reason}
    if installed(spec, root):
        _write_flag(spec, root, True)
        return status(pack_id, root, env=env)
    with _LOCK:
        running = _THREADS.get(pack_id)
        busy = (running is not None and running.is_alive()) or _JOBS.get(pack_id, {}).get("state") == "downloading"
        if not busy:
            cancel = _CANCELS[pack_id] = threading.Event()
            _JOBS[pack_id] = {"state": "downloading", "progress": 0.0, "step": "Starting the download",
                              "source": source}
    if busy:
        return status(pack_id, root, env=env)  # one download at a time per part
    if model_source is None:
        from superlocalmemory.runtimes.model_sources import HuggingFaceSource

        model_source = HuggingFaceSource(spec.repo, spec.revision, spec.download_bytes)
    thread = threading.Thread(target=_download, args=(spec, root, env, model_source, cancel),
                              name=f"media-pack-{pack_id}", daemon=True)
    with _LOCK:
        _THREADS[pack_id] = thread
    thread.start()
    if not background:
        thread.join()
    return status(pack_id, root, env=env)


def disable(pack_id: str, *, remove_files: bool = False, data_root: str | Path | None = None,
            env: Any = None) -> dict[str, Any]:
    """Turn one part off: the flag, a running download and the loaded model. The files stay unless asked."""
    if pack_id == REGIONS_ID:
        return _with_words(_switch_regions(False, _root(data_root), env))
    spec, root = get(pack_id), _root(data_root)
    try:
        _write_flag(spec, root, False)
    except OSError as exc:
        logger.warning("could not save the switch for %s: %s", pack_id, exc)
    cancel = _CANCELS.get(pack_id)
    if cancel is not None:
        cancel.set()
    thread = _THREADS.get(pack_id)
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(15.0)
    _stop_client(spec, root, env)
    if remove_files:
        shutil.rmtree(pack_dir(spec, root), ignore_errors=True)
    with _LOCK:
        _JOBS.pop(pack_id, None)
    return status(pack_id, root, env=env)


def _stop_client(spec: PackSpec, root: Path, env: Any) -> None:
    from superlocalmemory.runtimes import worker_client

    for client in worker_client.live_pack_clients():
        if client.model_id == spec.repo and str(client.root) == str(_managed(root, env).root):
            try:
                client.stop()
            except Exception:  # noqa: BLE001 - turning off must not fail on a worker that is already gone
                logger.debug("pack worker stop failed", exc_info=True)


def pack_client(pack_id: str, data_root: str | Path | None = None, *, env: Any = None):
    """The worker client for one part, or ``None`` unless the part is on, installed and the media
    environment is ready. Starts nothing: the worker starts on its first request."""
    try:
        spec, root = get(pack_id), _root(data_root)
        if not is_active(pack_id, root):
            return None
        managed = _managed(root, env)
        if managed.status().state != "ready":
            return None
        from superlocalmemory.runtimes.worker_client import shared_pack_client

        return shared_pack_client(managed, spec.repo, spec.revision, spec.role, model_dir(spec, root), PACK_IDLE_S)
    except Exception:  # noqa: BLE001 - an optional part that cannot start is simply absent
        logger.warning("pack %s is not available", pack_id, exc_info=True)
        return None


__all__ = ["CAPTIONS", "DOCUMENTS", "KILLED_WORDS", "NEEDS_MEDIA_WORDS", "PACKS", "PACK_IDLE_S", "PART_NAMES",
           "PackSpec", "REGIONS_ID", "disable", "download_words", "enable", "flag_enabled", "get", "installed",
           "is_active", "model_dir", "pack_client", "pack_dir", "size_words", "status", "status_all"]
