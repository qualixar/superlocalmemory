# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph Module

"""Configuration for the CodeGraph module.

Frozen dataclass with all tunables. Sensible defaults for typical repos.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from superlocalmemory.infra.data_root import state_path

# Languages supported out of the box (tree-sitter grammar names)
DEFAULT_LANGUAGES: frozenset[str] = frozenset({
    "python", "typescript", "tsx", "javascript", "jsx",
})

# File extensions → language mapping
DEFAULT_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
}

# Directories to always skip
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "coverage", ".code-review-graph",
    "vendor", ".eggs", "*.egg-info",
})


@dataclass(frozen=True)
class CodeGraphConfig:
    """Configuration for the CodeGraph module.

    All paths are absolute. File paths stored in the DB are relative
    to repo_root (HR-02).
    """

    # --- Core ---
    enabled: bool = False                 # Feature flag (default off for backward compat)
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    db_path: Path | None = None           # If None, derived from SLM base_dir

    # --- Parser ---
    languages: frozenset[str] = DEFAULT_LANGUAGES
    extension_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EXTENSION_MAP))
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS
    exclude_patterns: tuple[str, ...] = ()  # Additional glob patterns to skip
    max_file_size_bytes: int = 1_000_000  # Skip files > 1MB
    parse_timeout_seconds: float = 30.0   # Per-file parse timeout
    parallel_workers: int = 4             # ProcessPoolExecutor workers

    # --- Graph ---
    batch_size: int = 450                 # SQLite variable limit workaround
    max_depth_blast_radius: int = 2       # BFS depth for impact analysis
    max_nodes_blast_radius: int = 500     # Cap on blast radius nodes

    # --- Search ---
    rrf_k: int = 60                       # RRF fusion constant
    search_limit: int = 20                # Default search result limit

    # --- Resolution ---
    heuristic_confidence: float = 0.6     # Confidence for heuristic name matches

    # --- Bridge ---
    bridge_enabled: bool = False          # Bridge feature flag (default off)

    # --- Watch ---
    watch_debounce_ms: int = 300          # Debounce interval for file watcher

    def get_db_path(self, slm_base_dir: Path | None = None) -> Path:
        """Resolve the database path.

        Priority: explicit db_path > slm_base_dir/code_graph.db > ~/.superlocalmemory/code_graph.db
        """
        if self.db_path is not None:
            return self.db_path
        if slm_base_dir is not None:
            return slm_base_dir / "code_graph.db"
        return state_path("code_graph.db")

    @classmethod
    def load(cls, **overrides: object) -> CodeGraphConfig:
        """Build a config from ``code_graph_config.json``, then apply overrides.

        WHY THIS EXISTS (4.0.7). ``cli/setup_wizard.py`` has always written
        ``code_graph_config.json`` with ``enabled`` and ``bridge_enabled``, and
        until now **nothing read it**. There was no loader on this class at all,
        and every call site constructed ``CodeGraphConfig(enabled=True)`` with
        hardcoded defaults. So a user could answer "yes, enable the code graph"
        in setup, get ``bridge_enabled: true`` written to disk, and have it
        affect nothing — silently, with no way to tell from the outside.

        Unknown keys in the file are ignored rather than raising: the file is
        user-editable, and a stray key should not stop the code graph from
        loading. Malformed JSON falls back to defaults with a warning, because
        failing closed here would disable a working code graph over a typo.
        """
        import json
        import logging

        data: dict[str, object] = {}
        path = state_path("code_graph_config.json")
        try:
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
                else:
                    logging.getLogger(__name__).warning(
                        "%s does not contain a JSON object; using defaults", path,
                    )
        except (OSError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).warning(
                "could not read %s (%s); using defaults", path, exc,
            )

        data.update(overrides)

        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            logging.getLogger(__name__).debug(
                "ignoring unknown code_graph config keys: %s", ", ".join(unknown),
            )

        kwargs = {k: v for k, v in data.items() if k in valid}

        # JSON has no frozenset/Path; coerce the fields that need it.
        if isinstance(kwargs.get("languages"), list):
            kwargs["languages"] = frozenset(kwargs["languages"])
        if isinstance(kwargs.get("exclude_dirs"), list):
            kwargs["exclude_dirs"] = frozenset(kwargs["exclude_dirs"])
        for key in ("repo_root", "db_path"):
            if isinstance(kwargs.get(key), str):
                kwargs[key] = Path(kwargs[key])

        try:
            return cls(**kwargs)  # type: ignore[arg-type]
        except TypeError as exc:
            logging.getLogger(__name__).warning(
                "code_graph config rejected (%s); using defaults", exc,
            )
            return cls()
