# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Connect folder sources to the running daemon: the writer, the erasure path and the remote check."""

from __future__ import annotations

from typing import Any

from superlocalmemory import sources
from superlocalmemory.server.remote_access_state import remote_access_configured
from superlocalmemory.sources.scanner import SERVICE_NAME, SourceScanService


def _engine(state: Any) -> Any:
    return getattr(state, "engine", None)


def build_host(application: Any) -> sources.SourceHost:
    state = application.state

    def runtime() -> Any:
        found = getattr(state, "canonical_remember_runtime", None)
        return found if found is not None and getattr(found, "ready", False) else None

    def eraser(profile_id: str, fact_ids: list[str], subject_id: str) -> dict:
        from superlocalmemory.compliance.gdpr import GDPRCompliance

        engine = _engine(state)
        return GDPRCompliance(engine._db, engine=engine).forget_facts(
            fact_ids, profile_id, subject_id=subject_id)

    def actor_id() -> str:
        from superlocalmemory.server import unified_daemon

        return unified_daemon._materializer_actor_id()

    return sources.SourceHost(
        remote_check=remote_access_configured, runtime=runtime,
        config=lambda: getattr(_engine(state), "_config", None), eraser=eraser,
        profile=lambda: _engine(state)._profile_id, actor_id=actor_id)


def start_source_scanner(application: Any, registry: Any) -> None:
    """Register (once) and start the scan service; it stays idle while folder sources are off."""
    if registry.get(SERVICE_NAME) is None:
        host = build_host(application)
        service = SourceScanService(host)
        host.wake = service.wake
        sources.configure(host)
        registry.register(service)
    registry.start(SERVICE_NAME)


def stop_source_scanner(registry: Any, timeout_s: float = 5.0) -> bool:
    sources.configure(None)
    return registry.stop(SERVICE_NAME, timeout_s)
