"""Lifecycle contracts for the standalone REST application."""

from __future__ import annotations

import warnings


def test_create_app_registers_startup_without_deprecated_event_hooks() -> None:
    from superlocalmemory.server.api import create_app

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        application = create_app()

    assert application.router.lifespan_context is not None
