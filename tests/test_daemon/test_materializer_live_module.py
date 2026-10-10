# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The materializer reads the engine the serving module published.

The daemon runs as ``python -m ...unified_daemon`` (module ``__main__``) while
uvicorn imports the app by name, so the lifespan sets ``_engine`` on the
imported module. A materializer built in ``__main__`` must read that one.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

MODULE = "superlocalmemory.server.unified_daemon"


def test_materializer_reads_engine_of_the_imported_module():
    live = importlib.import_module(MODULE)
    spec = importlib.util.spec_from_file_location("_daemon_as_main", live.__file__)
    main_copy = importlib.util.module_from_spec(spec)
    sys.modules["_daemon_as_main"] = main_copy
    try:
        spec.loader.exec_module(main_copy)
        engine_before, runtime_before = live._engine, live._profile_runtime
        live._engine, live._profile_runtime = object(), object()
        try:
            service = main_copy._pending_materializer()
            assert service.engine_supplier() is live._engine
            assert service.runtime_supplier() is live._profile_runtime
        finally:
            live._engine, live._profile_runtime = engine_before, runtime_before
    finally:
        sys.modules.pop("_daemon_as_main", None)
