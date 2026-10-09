# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Where a managed environment gets its model weights from.

``HuggingFaceSource`` downloads with the environment's own Python (the same
pattern the answer-check model uses, including its stall watch).
``LocalDirSource`` copies a folder that is already on disk: offline installs
and tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from superlocalmemory.core import laya_process

ProgressFn = Callable[[float, str], None]
FetchResult = tuple[bool, str, str]  # (ok, error_kind, detail for logs only)

_DOWNLOAD_TIMEOUT_S = 3600.0
_NETWORK_HINTS = ("timeout", "timed out", "connection", "network", "resolve", "unreachable",
                  "ssl", "certificate", "name or service not known", "max retries exceeded")


def classify_error(text: str) -> str:
    lowered = (text or "").lower()
    return "network" if any(h in lowered for h in _NETWORK_HINTS) else "other"


class ModelSource(Protocol):
    model_id: str
    revision: str

    def fetch(self, python: Path, dest: Path, *, progress: ProgressFn,
              cancel: threading.Event | None = None) -> FetchResult: ...


class LocalDirSource:
    """Copies a local folder into place; never touches the network."""

    def __init__(self, path: str | Path, model_id: str = "local", revision: str = "local") -> None:
        self.path, self.model_id, self.revision = Path(path), model_id, revision

    def fetch(self, python: Path, dest: Path, *, progress: ProgressFn,
              cancel: threading.Event | None = None) -> FetchResult:
        staging = dest.with_name(dest.name + ".partial")
        try:
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            for file in sorted(p for p in self.path.rglob("*") if p.is_file()):
                if cancel is not None and cancel.is_set():
                    shutil.rmtree(staging, ignore_errors=True)
                    return False, "cancelled", "stopped by the person"
                target = staging / file.relative_to(self.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
            shutil.rmtree(dest, ignore_errors=True)
            staging.replace(dest)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            return False, "disk" if getattr(exc, "errno", None) == 28 else "other", str(exc)
        progress(1.0, "Copying the model files")
        return True, "", ""


class HuggingFaceSource:
    """Downloads one pinned revision with the environment's Python."""

    def __init__(self, model_id: str, revision: str, expected_bytes: int = 0) -> None:
        self.model_id, self.revision, self.expected_bytes = model_id, revision, expected_bytes

    def fetch(self, python: Path, dest: Path, *, progress: ProgressFn,
              cancel: threading.Event | None = None) -> FetchResult:
        dest.mkdir(parents=True, exist_ok=True)
        script = ("import sys; from huggingface_hub import snapshot_download; "
                  "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]); print('OK')")
        env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        env["TOKENIZERS_PARALLELISM"] = "false"
        # stderr goes to a file: a pipe nobody reads fills up and freezes the download.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
            try:
                proc = subprocess.Popen([str(python), "-I", "-c", script, self.model_id, self.revision, str(dest)],
                                        stdout=subprocess.DEVNULL, stderr=err, text=True, env=env)
            except OSError as exc:
                return False, "other", str(exc)

            def measure() -> int:
                if cancel is not None and cancel.is_set():
                    laya_process.stop(proc)
                return laya_process.folder_size(dest)

            def report(size: int) -> None:
                if self.expected_bytes:
                    progress(min(size / self.expected_bytes, 1.0),
                             f"Downloading the model ({size // (1024 * 1024)} MB)")

            stopped = laya_process.watch_download(
                proc, measure, timeout_s=_DOWNLOAD_TIMEOUT_S, stall_s=laya_process.DEFAULT_STALL_S,
                on_size=report)
            if stopped is not None:
                return stopped
            err.seek(0)
            stderr = err.read()[-2000:]
        if cancel is not None and cancel.is_set():
            return False, "cancelled", "stopped by the person"
        if proc.returncode == 0:
            return True, "", ""
        return False, classify_error(stderr), stderr
