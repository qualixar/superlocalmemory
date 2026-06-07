"""Module-level accessor for OptimizeConfig (INTERFACE-CONTRACT §2).

LLD-01 (and all other Optimize LLDs) import get_optimize_config() from this
module. They NEVER construct ConfigStore themselves or read optimize.json directly.
"""

from __future__ import annotations

from superlocalmemory.optimize.config.schema import OptimizeConfig

_store: "ConfigStore | None" = None


def get_optimize_config() -> OptimizeConfig:
    """Return the current active OptimizeConfig (thread-safe, no I/O)."""
    if _store is None:
        from superlocalmemory.optimize.config.defaults import DEFAULT_OPTIMIZE_CONFIG
        return DEFAULT_OPTIMIZE_CONFIG
    return _store.get()


def load_optimize_config() -> OptimizeConfig:
    """Alias of get_optimize_config() — INTERFACE-CONTRACT v2 §2."""
    return get_optimize_config()


def _set_config_store(store: "ConfigStore") -> None:
    global _store
    _store = store


def _reset_config_store() -> None:
    """Reset the module-level store (testing only)."""
    global _store
    _store = None
