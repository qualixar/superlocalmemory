# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A Python environment SLM builds and owns, separate from SLM's own.

Nothing that runs inside it may import superlocalmemory: scripts are standalone
files launched by path (``run_script``). The package list is a hashed lock file
per platform and Python, installed with ``--require-hashes --no-deps``; a
computer without a lock is reported as unsupported rather than guessed at.
Failure text shown to people is plain language: no paths, no tracebacks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from superlocalmemory.core import laya_process
from superlocalmemory.infra.instance_lock import InstanceLock
from superlocalmemory.runtimes.model_sources import (
    HuggingFaceSource, LocalDirSource, ModelSource, classify_error,
)

logger = logging.getLogger(__name__)

LOCKS_DIR = Path(__file__).resolve().parent / "locks"
STATE_FILE = "state.json"
_PIP_TIMEOUT_S = 1800.0
_VENV_TIMEOUT_S = 180.0

NOT_INSTALLED, INSTALLING, READY, FAILED, UNSUPPORTED = (
    "not_installed", "installing", "ready", "failed", "unsupported")

_MESSAGES = {
    "network": "Can't reach the download server. Check the connection, then try again.",
    "stalled": "The download stopped making progress, so it was stopped. Try again.",
    "disk": "There isn't enough free disk space. Free some space, then try again.",
    "pip": "Installing the packages didn't finish. Try again.",
    "canary": "The installed parts didn't pass their self-check. Try again, or remove and set up again.",
    "cancelled": "Setup was cancelled. Start again to continue; what was downloaded is kept.",
    "unpinned": "The media model version is not pinned in this build yet",
    "other": "Setup didn't finish. Try again.",
}
STEP_STOPPED = "Setup stopped before it finished. Try again."
STEP_REPAIR = "Needs repair: the Python this was built with changed"
STEP_NO_LOCK = "This build has no media package list for this computer yet"
STEP_NO_VEC = "This Python cannot load the vector extension"
STEP_BAD_PYTHON = "This Python version is not supported"
STEP_INTEL_MAC = "Images and documents are not available on Intel Macs"


@dataclass(frozen=True)
class EnvSpec:
    name: str
    requirements: tuple[str, ...]
    model_source: ModelSource
    min_free_disk_bytes: int
    canary: Callable[[Path], bool] | None = None
    expected_download_bytes: int = 0
    lock_prefix: str = "media"


@dataclass(frozen=True)
class EnvStatus:
    state: str
    progress: float
    step: str
    error_kind: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- platform facts (module-level so tests can replace them) -----------------

def _python_supported() -> bool:
    return (3, 12) <= sys.version_info[:2] < (3, 15)


def _platform_tag() -> str:
    system = {"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux")
    return f"{system}-{platform.machine().lower()}"


def _base_python() -> tuple[str, str]:
    return getattr(sys, "_base_executable", None) or sys.executable, platform.python_version()


def _sqlite_vec_ok() -> bool:
    import sqlite3

    try:
        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            return True
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - any failure means "cannot load it"
        return False


def lock_name(prefix: str = "media") -> str:
    return f"{prefix}-{_platform_tag()}-py{sys.version_info[0]}{sys.version_info[1]}.txt"


def _ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("p", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("rest", ctypes.c_ulonglong * 5)]

        mem = _Mem()
        mem.l = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))  # type: ignore[attr-defined]
        return int(mem.total)
    except Exception:  # noqa: BLE001 - unknown RAM is reported as 0
        return 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock_index_args(text: str) -> list[str]:
    """The ``--index-url X`` / ``--extra-index-url X`` lines recorded in a lock header (commented or not)."""
    args: list[str] = []
    for line in text.splitlines():
        body = line.lstrip("# ").strip()
        if body.startswith(("--index-url ", "--extra-index-url ")):
            args += body.split(None, 1)
    return args


class ManagedEnv:
    def __init__(self, spec: EnvSpec, *, root: Path | None = None, runner: Callable | None = None) -> None:
        self.spec = spec
        self._root_arg = Path(root) if root is not None else None
        self._runner = runner
        self._cancel: threading.Event | None = None
        self._mutex = threading.Lock()

    # -- layout ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        if self._root_arg is not None:
            return self._root_arg
        from superlocalmemory.infra.data_root import canonical_data_root

        return canonical_data_root() / "runtimes" / self.spec.name

    def python(self) -> Path:
        sub = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        return self.root / "venv" / sub

    def weights_dir(self) -> Path:
        return self.root / "weights"

    def _lock_path(self) -> Path:
        return LOCKS_DIR / lock_name(self.spec.lock_prefix)

    # -- state file -----------------------------------------------------------
    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads((self.root / STATE_FILE).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _update(self, **changes: Any) -> None:
        with self._mutex:
            record = {**self._read(), **changes, "updated_at": _now()}
            self.root.mkdir(parents=True, exist_ok=True)
            # mkstemp: a unique, owner-only (0600) file in the same folder, so the replace is atomic.
            fd, tmp = tempfile.mkstemp(prefix=f"{STATE_FILE}.", suffix=".tmp", dir=str(self.root))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(record))
                os.replace(tmp, self.root / STATE_FILE)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

    def _mark(self, step: str, value: str) -> None:
        self._update(markers={**self._read().get("markers", {}), step: value})

    def _clear_markers(self, *steps: str) -> None:
        kept = {k: v for k, v in self._read().get("markers", {}).items() if k not in steps}
        self._update(markers=kept)

    def _marker(self, step: str) -> str:
        return str(self._read().get("markers", {}).get(step, ""))

    # -- status ---------------------------------------------------------------
    def status(self) -> EnvStatus:
        rec = self._read()
        state = str(rec.get("state", NOT_INSTALLED))
        stamp = str(rec.get("updated_at", ""))
        if state == INSTALLING and not self._install_running():
            return EnvStatus(FAILED, float(rec.get("progress", 0.0)), STEP_STOPPED, "other", stamp)
        if state == READY and self._base_changed(rec):
            return EnvStatus(FAILED, 1.0, STEP_REPAIR, "other", stamp)
        if state in (NOT_INSTALLED, FAILED) and (why := self._unsupported_reason()):
            return EnvStatus(UNSUPPORTED, 0.0, why, "", stamp)
        return EnvStatus(state, float(rec.get("progress", 0.0)), str(rec.get("step", "")),
                         str(rec.get("error_kind", "")), stamp)

    def _install_running(self) -> bool:
        probe = InstanceLock(self.root / "install.lock")
        if probe.try_acquire():
            probe.release()
            return False
        return True

    def _base_changed(self, rec: dict[str, Any]) -> bool:
        path, version = _base_python()
        return (rec.get("base_python"), rec.get("base_version")) != (path, version) or not self.python().exists()

    def _unsupported_reason(self) -> str:
        if not _python_supported():
            return STEP_BAD_PYTHON
        if _platform_tag() == "darwin-x86_64":
            return STEP_INTEL_MAC
        if not _sqlite_vec_ok():
            return STEP_NO_VEC
        if not self._lock_path().is_file():
            return STEP_NO_LOCK
        return ""

    def precheck(self) -> dict[str, Any]:
        probe = next((p for p in (self.root, *self.root.parents) if p.exists()), Path.cwd())
        try:
            free = shutil.disk_usage(probe).free
        except OSError:
            free = 0
        ram = _ram_bytes()
        return {"disk_ok": free >= self.spec.min_free_disk_bytes, "free_bytes": free,
                "ram_bytes": ram,
                "python_ok": _python_supported(), "python": platform.python_version(),
                "sqlite_vec_ok": _sqlite_vec_ok()}

    # -- running things -------------------------------------------------------
    def _run(self, cmd: list[str], *, timeout_s: float, env: dict[str, str] | None = None):
        runner = self._runner or self._cancelable_run
        # Nothing from the caller's shell may steer pip or Python inside the environment.
        clean = {k: v for k, v in (env if env is not None else os.environ).items()
                 if not k.startswith(("PIP_", "PYTHON"))}
        return runner(cmd, timeout_s=timeout_s, env=clean, classify=classify_error)

    def _cancelable_run(self, cmd, *, timeout_s, env, classify):
        cancel = self._cancel
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        except OSError as exc:
            return False, "other", str(exc)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                out, err = proc.communicate(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    laya_process.stop(proc)
                    return False, "cancelled", "stopped by the person"
                if time.monotonic() >= deadline:
                    laya_process.stop(proc)
                    return False, laya_process.KIND_TIMEOUT, f"ran past {timeout_s:.0f} s"
        if proc.returncode == 0:
            return True, "", ""
        detail = (err or out or "")[-2000:]
        return False, classify(detail), detail

    def _create_venv(self) -> tuple[bool, str, str]:
        return self._run([sys.executable, "-m", "venv", str(self.root / "venv")], timeout_s=_VENV_TIMEOUT_S)

    def run_script(self, path: str | Path, args: list[str] | tuple[str, ...] = (), *,
                   timeout_s: float = 300.0, extra_env: dict[str, str] | None = None,
                   input_text: str | None = None) -> subprocess.CompletedProcess:
        """Run a standalone script in isolated mode with the env root as cwd and no PYTHONPATH."""
        env = {**os.environ, **(extra_env or {})}
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
            env.pop(name, None)
        return subprocess.run([str(self.python()), "-I", str(path), *map(str, args)], cwd=str(self.root),
                              env=env, capture_output=True, text=True, timeout=timeout_s, input=input_text)

    # -- install --------------------------------------------------------------
    def install(self, *, on_progress: Callable[[float, str], None] | None = None,
                cancel: threading.Event | None = None) -> EnvStatus:
        """Build the environment end to end. Never raises; failures end as state failed."""
        if why := self._unsupported_reason():
            return EnvStatus(UNSUPPORTED, 0.0, why)
        self.root.mkdir(parents=True, exist_ok=True)
        lock = InstanceLock(self.root / "install.lock")
        if not lock.try_acquire():
            return self.status()
        self._cancel = cancel
        try:
            self._drop_stale_venv()
            return self._install_locked(on_progress or (lambda f, s: None), cancel)
        except Exception:  # noqa: BLE001 - install never raises
            logger.exception("environment install failed unexpectedly")
            return self._fail("other", self._progress_now())
        finally:
            self._cancel = None
            lock.release()

    def _drop_stale_venv(self) -> None:
        """A venv built by a different base Python is rebuilt, and its packages are reinstalled."""
        rec = self._read()
        recorded = (rec.get("base_python"), rec.get("base_version"))
        if recorded[0] and recorded != _base_python():
            shutil.rmtree(self.root / "venv", ignore_errors=True)
            self._clear_markers("venv", "pip")

    def _progress_now(self) -> float:
        return float(self._read().get("progress", 0.0))

    def _install_locked(self, report: Callable[[float, str], None], cancel) -> EnvStatus:
        top = [0.0]

        def say(fraction: float, step: str) -> None:
            top[0] = max(top[0], fraction)
            self._update(state=INSTALLING, progress=top[0], step=step, error_kind="")
            report(top[0], step)

        def stopped() -> bool:
            return cancel is not None and cancel.is_set()

        source = self.spec.model_source
        if isinstance(source, HuggingFaceSource) and not source.revision:
            return self._fail("unpinned", 0.0)
        say(0.02, "Checking this computer")
        if not self.precheck()["disk_ok"]:
            return self._fail("disk", top[0])
        for run_step in (self._step_venv, self._step_packages, self._step_weights, self._step_canary):
            if stopped():
                return self._fail("cancelled", top[0])
            kind = run_step(say)
            if kind:
                return self._fail(kind, top[0])
        if stopped():
            return self._fail("cancelled", top[0])
        path, version = _base_python()
        self._update(state=READY, progress=1.0, step="Ready", error_kind="", base_python=path, base_version=version)
        report(1.0, "Ready")
        return self.status()

    def _fail(self, kind: str, progress: float) -> EnvStatus:
        step = _MESSAGES.get(kind, _MESSAGES["other"])
        self._update(state=FAILED, progress=progress, step=step, error_kind=kind)
        return EnvStatus(FAILED, progress, step, kind)

    def _step_venv(self, say) -> str:
        if self._marker("venv") and self.python().exists():
            return ""
        say(0.05, "Creating the environment")
        shutil.rmtree(self.root / "venv", ignore_errors=True)
        self._clear_markers("venv", "pip")
        ok, kind, detail = self._create_venv()
        if not ok:
            logger.warning("venv creation failed: %s", detail)
            return kind if kind in ("cancelled", "disk") else "other"
        self._mark("venv", "1")
        return ""

    def _step_packages(self, say) -> str:
        if not self.spec.requirements:
            return ""
        lock = self._lock_path()
        text = lock.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self._marker("pip") == digest:
            return ""
        say(0.15, "Installing the packages")
        cmd = [str(self.python()), "-m", "pip", "install", "--disable-pip-version-check", "--isolated",
               "--only-binary=:all:", "--require-hashes", "--no-deps", *_lock_index_args(text), "-r", str(lock)]
        ok, kind, detail = self._run(cmd, timeout_s=_PIP_TIMEOUT_S, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
        if not ok:
            logger.warning("package install failed (%s): %s", kind, detail)
            return kind if kind in ("network", "cancelled", "disk", "stalled") else "pip"
        self._mark("pip", digest)
        return ""

    def _step_weights(self, say) -> str:
        source = self.spec.model_source
        wanted = f"{source.model_id}@{source.revision}"
        if self._marker("weights") == wanted and self.weights_dir().is_dir():
            return ""

        def progress(fraction: float, step: str) -> None:
            say(0.35 + 0.55 * min(max(fraction, 0.0), 1.0), step)

        say(0.35, "Getting the model")
        ok, kind, detail = source.fetch(self.python(), self.weights_dir(), progress=progress, cancel=self._cancel)
        if not ok:
            logger.warning("model fetch failed (%s): %s", kind, detail)
            return kind if kind in _MESSAGES else "other"
        self._mark("weights", wanted)
        return ""

    def _step_canary(self, say) -> str:
        if self.spec.canary is None:
            return ""
        say(0.92, "Checking that everything works")
        try:
            healthy = bool(self.spec.canary(self.python()))
        except Exception:  # noqa: BLE001 - a crashing check is a failed check
            logger.exception("self-check crashed")
            healthy = False
        return "" if healthy else "canary"

    # -- removal --------------------------------------------------------------
    def remove(self, *, keep_weights: bool) -> EnvStatus:
        """Delete the environment (and the weights unless kept). Refused while an install runs."""
        lock = InstanceLock(self.root / "install.lock")
        if not lock.try_acquire():
            return EnvStatus(INSTALLING, self._progress_now(), "Setup is still running. Cancel it first.")
        try:
            shutil.rmtree(self.root / "venv", ignore_errors=True)
            keep = {"weights": self._marker("weights")} if keep_weights and self._marker("weights") else {}
            if not keep_weights:
                shutil.rmtree(self.weights_dir(), ignore_errors=True)
            self._update(state=NOT_INSTALLED, progress=0.0, step="", error_kind="", markers=keep,
                         base_python="", base_version="")
        finally:
            lock.release()
        return self.status()


__all__ = ["EnvSpec", "EnvStatus", "HuggingFaceSource", "LocalDirSource", "ManagedEnv", "ModelSource",
           "lock_name"]
