# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon's side of the image-and-text worker (``multimodal_worker.py``).

One worker process, started on first use and stopped when idle or too large;
one request at a time. A recall never starts it: ``embed_query`` answers None
while it is cold and warms it in the background. Nothing here touches the text
embedder, and nothing starts a process while the feature is off.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from superlocalmemory.core import ram_lock
from superlocalmemory.runtimes import media_models
from superlocalmemory.runtimes.features import media_enabled, register_media_stop_hook
from superlocalmemory.runtimes.ports import MediaEmbedderPort

logger = logging.getLogger(__name__)

WORKER_PATH = Path(__file__).resolve().parent / "multimodal_worker.py"
DEFAULT_IDLE_S = 1800.0
DEFAULT_RSS_LIMIT_MB = media_models.DEFAULT_RSS_LIMIT_MB
MAX_TEXTS, MAX_PATHS = 64, 16
_QUIT_WAIT_S = 2.0


class MediaWorkerError(RuntimeError):
    """The worker could not answer. The text is plain language."""


class MediaWorkerWarming(MediaWorkerError):
    """The worker is starting; ask again shortly."""


class _Dead(Exception):
    """The worker process ended (internal; leads to one restart)."""


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _test_mode() -> bool:
    return os.environ.get("SLM_TEST_ISOLATION") == "1"


class MediaWorkerClient(MediaEmbedderPort):
    def __init__(self, env: Any, *, model_id: str, revision: str, idle_s: float | None = None,
                 rss_limit_mb: int | None = None, request_timeout_s: float = 120.0,
                 load_timeout_s: float = 600.0, role: str = "") -> None:
        if model_id.startswith("fake:") and not _test_mode():
            raise ValueError("fake models are for tests only")
        self._env, self.model_id, self.revision = env, model_id, revision
        self.role = role  # "image": a vision-only model that never embeds text
        self.idle_s = float(idle_s if idle_s is not None else _env_number("SLM_MEDIA_WORKER_IDLE_S", DEFAULT_IDLE_S))
        self.rss_limit_mb = int(rss_limit_mb if rss_limit_mb is not None
                                else _env_number("SLM_MEDIA_WORKER_RSS_LIMIT_MB",
                                                         media_models.rss_limit_mb_for(model_id)))
        self.request_timeout_s, self.load_timeout_s = request_timeout_s, load_timeout_s
        self.dim = 0
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self._loaded = False
        self._next_id = 0
        self._timer: threading.Timer | None = None
        self._warming = threading.Event()
        self._halted = False

    # -- state ----------------------------------------------------------------
    @property
    def root(self) -> Path:
        """The managed environment this client's worker runs from."""
        return Path(self._env.root)

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    def is_warm(self) -> bool:
        return self._loaded and self.pid is not None

    # -- process --------------------------------------------------------------
    def _model_arg(self) -> str:
        weights = Path(self._env.weights_dir())
        if not self.model_id.startswith("fake:") and weights.is_dir() and any(weights.iterdir()):
            return str(weights)
        return self.model_id

    def _max_pixels(self) -> int:
        profile = media_models.profile_for(self.model_id)
        return profile.image_max_pixels if profile is not None else 0

    def _worker_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        if not _test_mode():
            env.pop("SLM_MEDIA_WORKER_FAKE", None)
        return env

    def _spawn(self) -> None:
        root = Path(self._env.root)
        proc = subprocess.Popen([str(self._env.python()), "-I", str(WORKER_PATH)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                cwd=str(root) if root.is_dir() else None, env=self._worker_env())
        replies: queue.Queue = queue.Queue()

        def pump() -> None:
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    replies.put(line)
            except (OSError, ValueError):
                pass
            replies.put(None)

        threading.Thread(target=pump, daemon=True, name="media-worker-reader").start()
        self._proc, self._replies, self._loaded = proc, replies, False

    def _kill(self) -> None:
        proc, self._proc, self._loaded = self._proc, None, False
        if proc is None:
            return
        try:
            proc.kill()
        except OSError:
            pass
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - already closed
                pass
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - nothing more can be done
            pass

    def _roundtrip(self, payload: dict, timeout_s: float) -> dict:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise _Dead()
        self._next_id += 1
        want = self._next_id
        try:
            proc.stdin.write(json.dumps({**payload, "id": want}) + "\n")  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]
        except (OSError, ValueError):
            raise _Dead() from None
        while True:
            try:
                line = self._replies.get(timeout=timeout_s)
            except queue.Empty:
                self._kill()
                raise MediaWorkerError("The image model took too long and was stopped (timed out).") from None
            if line is None:
                raise _Dead()
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            if isinstance(reply, dict) and reply.get("id") == want:
                return reply

    def _start_locked(self) -> None:
        """Spawn and load; the RAM reservation covers both so only one heavy start runs at a time."""
        if self.pid is not None and self._loaded:
            return
        self._kill()
        need = 0 if self.model_id.startswith("fake:") else media_models.load_mb_for(self.model_id)
        try:
            with ram_lock.ram_reservation("media-model-load", required_mb=need, timeout_s=self.load_timeout_s):
                self._spawn()
                reply = self._roundtrip({"cmd": "load", "model": self._model_arg(), "revision": self.revision,
                                         "role": self.role, "max_pixels": self._max_pixels(), "hf_home": str(self._env.weights_dir()), "device": "auto"},
                                        self.load_timeout_s)
        except RuntimeError as exc:
            if isinstance(exc, MediaWorkerError):
                raise
            raise MediaWorkerError("Not enough free memory to load the image model right now.") from exc
        except OSError as exc:
            raise MediaWorkerError("The image model could not be started.") from exc
        if not reply.get("ok"):
            self._kill()
            raise MediaWorkerError("The image model could not be loaded.")
        self.dim, self._loaded = int(reply.get("dim", 0)), True

    # -- requests -------------------------------------------------------------
    def _request_locked(self, cmd: str, retry: bool = True, **fields: Any) -> dict:
        self._halted = False
        self._cancel_timer()
        try:
            for attempt in (0, 1):
                try:
                    self._start_locked()
                    reply = self._roundtrip({"cmd": cmd, **fields}, self.request_timeout_s)
                    break
                except _Dead:
                    self._kill()
                    if attempt or not retry or self._halted:
                        raise MediaWorkerError("The image model stopped unexpectedly.") from None
        finally:
            self._after_request()
        if not reply.get("ok"):
            raise MediaWorkerError("The image model could not process that.")
        return reply

    def _request(self, cmd: str, retry: bool = True, **fields: Any) -> dict:
        with self._lock:
            return self._request_locked(cmd, retry, **fields)

    def _after_request(self) -> None:
        pid = self.pid
        if pid is None:
            return
        if self.rss_limit_mb > 0 and self._rss_mb(pid) > self.rss_limit_mb:
            logger.warning("image worker is over its memory limit (%d MB); stopping it", self.rss_limit_mb)
            self._kill()
            return
        self._timer = threading.Timer(self.idle_s, self._idle_expired)
        self._timer.daemon = True
        self._timer.start()

    @staticmethod
    def _rss_mb(pid: int) -> float:
        try:
            import psutil

            return psutil.Process(pid).memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001 - unknown size is treated as fine
            return 0.0

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _idle_expired(self) -> None:
        if not self._lock.acquire(blocking=False):
            return  # busy: the finishing request sets a new timer
        try:
            self._stop_locked(graceful=True)
        finally:
            self._lock.release()

    def _stop_locked(self, *, graceful: bool) -> None:
        self._cancel_timer()
        proc = self._proc
        if graceful and proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
                proc.wait(timeout=_QUIT_WAIT_S)
            except Exception:  # noqa: BLE001 - killed below
                pass
        self._kill()

    def stop(self) -> None:
        """Stop the worker now. Safe to call twice and from any thread."""
        self._halted = True
        if self._lock.acquire(timeout=3.0):
            try:
                self._stop_locked(graceful=False)
            finally:
                self._lock.release()
        else:
            self._kill()  # a request is stuck: end the process so it gives the lock back

    # -- public API -----------------------------------------------------------
    def warm_up(self) -> None:
        """Start and load the worker in the background; returns at once."""
        if self._warming.is_set() or self.is_warm():
            return
        self._warming.set()

        def run() -> None:
            try:
                with self._lock:
                    self._halted = False
                    self._start_locked()
                    self._after_request()
            except Exception as exc:  # noqa: BLE001 - warming is best effort
                logger.info("image worker warm-up failed: %s", exc)
            finally:
                self._warming.clear()

        threading.Thread(target=run, daemon=True, name="media-worker-warmup").start()

    def embed_texts(self, texts: list[str], *, prompt: Literal["SearchQuery", "Document"]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), MAX_TEXTS):
            out += self._request("embed_text", texts=texts[i:i + MAX_TEXTS], prompt=prompt)["vectors"]
        return out

    def embed_images(self, paths: list[Path], *, wait_cold: bool = True) -> list[list[float]]:
        if not wait_cold and not self.is_warm():
            self.warm_up()
            raise MediaWorkerWarming("The image model is starting. Try again in a moment.")
        out: list[list[float]] = []
        for i in range(0, len(paths), MAX_PATHS):
            out += self._request("embed_image", paths=[str(p) for p in paths[i:i + MAX_PATHS]])["vectors"]
        return out

    def _image_request(self, cmd: str, wait_cold: bool, **fields: Any) -> dict:
        if not wait_cold and not self.is_warm():
            self.warm_up()
            raise MediaWorkerWarming("The image model is starting. Try again in a moment.")
        reply = self._request(cmd, **fields)
        return {k: v for k, v in reply.items() if k not in ("ok", "id")}

    def prepare_image(self, path: Path | str, out_dir: Path | str, *, wait_cold: bool = True) -> dict:
        """Strip metadata, make a thumbnail and a perceptual hash; results are files in ``out_dir``."""
        return self._image_request("prepare_image", wait_cold, path=str(path), out_dir=str(out_dir))

    def ocr_image(self, path: Path | str, *, wait_cold: bool = True) -> dict:
        """The text in an image: ``{"engine": ..., "text": ...}`` (engine ``none`` when none is installed)."""
        return self._image_request("ocr_image", wait_cold, path=str(path))

    def embed_query(self, text: str, *, wait_s: float = 0.3) -> list[float] | None:
        """For recall: a vector only if the worker is already warm and free; otherwise None."""
        if not self.is_warm():
            self.warm_up()
            return None
        if not self._lock.acquire(timeout=wait_s):
            return None
        try:
            return self._request_locked("embed_text", texts=[text], prompt="SearchQuery")["vectors"][0]
        except MediaWorkerError as exc:
            logger.info("image worker query failed: %s", exc)
            return None
        finally:
            self._lock.release()


# -- factory --------------------------------------------------------------------

_CLIENTS: dict[tuple[str, str, str, str], MediaWorkerClient] = {}
_CLIENTS_LOCK = threading.Lock()


def live_clients() -> list[MediaWorkerClient]:
    """The clients that already exist in this process; creates nothing and starts nothing."""
    with _CLIENTS_LOCK:
        return list(_CLIENTS.values())


def _shared_client(managed: Any, model_id: str, revision: str, role: str, *,
                   stop_hook: bool = True) -> MediaWorkerClient:
    """The one client per (folder, model, revision, role) in this process."""
    key = (str(managed.root), model_id, revision, role)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = _CLIENTS[key] = MediaWorkerClient(managed, model_id=model_id, revision=revision, role=role)
            if stop_hook:
                register_media_stop_hook(client.stop)
        return client


def media_embedder(*, env: Any = None, data_root: str | Path | None = None, model_id: str | None = None,
                   revision: str | None = None) -> MediaWorkerClient | None:
    """The shared client, or None unless images and documents are on and the environment is ready.

    Starts nothing: the worker starts on the first request.
    """
    if not media_enabled(data_root):
        return None
    from superlocalmemory.runtimes.media_env import media_env
    from superlocalmemory.runtimes.space_plan import current_space_plan

    managed = env or media_env(root=Path(data_root) / "runtimes" / "media" if data_root is not None else None)
    if managed.status().state != "ready":
        return None
    role = ""
    if model_id is None:  # the image model is the plan's: the paired vision model or the separate one
        plan = current_space_plan(data_root)
        model_id, revision = plan.image_model, plan.image_revision if revision is None else revision
        role = "image" if plan.mode == "paired" else ""
    return _shared_client(managed, model_id, "" if revision is None else revision, role)


def text_loadout_role(data_root: str | Path | None = None) -> str:
    """``""`` (the full model) while pictures are on, so text and pictures share one process; else ``"text"``."""
    return "" if media_enabled(data_root) else "text"


def text_embedder(*, env: Any, data_root: str | Path | None, model_id: str,
                  revision: str = "") -> MediaWorkerClient | None:
    """The shared client that makes text vectors, or None unless the environment is ready.

    Not gated on the pictures switch: the text provider is chosen in the embedding settings. The
    text-only loadout is used when pictures are off and registers no stop hook (turning pictures
    off has nothing of its own to stop). Starts nothing.
    """
    if env.status().state != "ready":
        return None
    role = text_loadout_role(data_root)
    return _shared_client(env, model_id, revision, role, stop_hook=role == "")


__all__ = ["MediaWorkerClient", "MediaWorkerError", "MediaWorkerWarming", "WORKER_PATH", "live_clients", "media_embedder",
           "text_embedder", "text_loadout_role"]
