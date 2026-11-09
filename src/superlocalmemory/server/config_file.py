"""Process-safe, durable read/modify/write access to ``config.json``."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable

from superlocalmemory.infra.config_lock import config_file_lock

# The lock moved to infra (4.2.14) so SLMConfig.save() can take the same one.
# The old private name stays: server/backup_bundle.py imports it.
_file_lock = config_file_lock


def _read_unlocked(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config root must be a JSON object")
    return value


def read_config(path: Path) -> dict:
    """Return one coherent config snapshot under the interprocess lock."""
    with _file_lock(path):
        return _read_unlocked(path)


def update_config(path: Path, update: Callable[[dict], None]) -> dict:
    """Atomically update and durably replace a config JSON object."""
    with _file_lock(path):
        data = _read_unlocked(path)
        update(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, path)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)
        return data


__all__ = ("read_config", "update_config")
