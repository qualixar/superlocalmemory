# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Plain-words reasons for "images and documents are on but not usable yet".

Turned off and not ready are different answers: the first is a choice, the
second is set-up still running, impossible here, or unfinished.
"""

from __future__ import annotations

from pathlib import Path

_PENDING = "Images and documents are still being set up. Try again in a few minutes (see: slm media status)."
_UNSUPPORTED = "Images and documents can't be set up on this computer yet (see: slm media status)."
_FAILED = "Setting up images and documents did not finish. See: slm media status."


def setup_message(state: str, step: str = "") -> str:
    """The reason to show for an environment that is not ready; ``step`` is not shown."""
    if state in ("not_installed", "installing"):
        return _PENDING
    if state == "unsupported":
        return _UNSUPPORTED
    return _FAILED


def media_refusal(data_root: str | Path | None = None) -> str | None:
    """The set-up message when the feature is on but its environment is not ready, else None.

    None also covers "turned off": that text belongs to the caller.
    """
    from superlocalmemory.runtimes.features import media_enabled

    if not media_enabled(data_root):
        return None
    from superlocalmemory.runtimes.media_env import media_env

    root = Path(data_root) / "runtimes" / "media" if data_root is not None else None
    status = media_env(root=root).status()
    if status.state == "ready":
        return None
    return setup_message(status.state, getattr(status, "step", ""))
