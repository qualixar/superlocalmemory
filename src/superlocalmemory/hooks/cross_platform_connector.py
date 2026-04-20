# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-05 §8.2

"""Cross-platform adapter orchestrator (``slm connect``).

LLD-05 §8.2. Holds the active set of adapters, detects which are installed,
runs a one-shot sync over every active adapter, and flips an individual
adapter off. Kept in its own module so it can be covered in isolation from
the legacy ``IDEConnector`` shim that still lives in ``ide_connector.py``
for backward compatibility with existing SLM users.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from superlocalmemory.hooks.adapter_base import Adapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdapterStatus:
    """Return record for ``CrossPlatformConnector.detect``."""
    name: str
    active: bool
    target_path: str


class CrossPlatformConnector:
    """LLD-05 §8.2 orchestrator.

    A thin coordinator: every adapter knows how to detect itself, sync
    itself, and disable itself. This class just iterates, catches, and
    reports. Errors never abort the iteration (A8).
    """

    def __init__(self, adapters: Iterable[Adapter]) -> None:
        self._adapters: list[Adapter] = list(adapters)

    @property
    def adapters(self) -> list[Adapter]:
        return list(self._adapters)

    def detect(self) -> list[AdapterStatus]:
        out: list[AdapterStatus] = []
        for a in self._adapters:
            try:
                active = a.is_active()
            except Exception:
                active = False
            try:
                target = str(a.target_path)
            except Exception:
                target = "?"
            out.append(AdapterStatus(name=a.name, active=active,
                                     target_path=target))
        return out

    def connect(self) -> dict[str, str]:
        """One-shot sync over every active adapter."""
        results: dict[str, str] = {}
        for a in self._adapters:
            try:
                if not a.is_active():
                    results[a.name] = "inactive"
                    continue
                wrote = a.sync()
                results[a.name] = "wrote" if wrote else "skipped"
            except Exception as exc:
                logger.warning("adapter %s failed: %s", a.name, exc)
                results[a.name] = f"error:{type(exc).__name__}"
        return results

    def disable(self, name: str) -> bool:
        for a in self._adapters:
            if a.name == name:
                try:
                    a.disable()
                except Exception as exc:
                    logger.warning("disable %s failed: %s", name, exc)
                    return False
                return True
        return False


__all__ = ("AdapterStatus", "CrossPlatformConnector")
