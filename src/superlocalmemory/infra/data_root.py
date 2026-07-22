# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Single authority for SuperLocalMemory runtime-state paths.

The resolver is intentionally stdlib-only so CLI, daemon, MCP, and hook entry
points can use it before importing the engine. Environment selection is an
explicit process contract and always wins over persisted configuration.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

DATA_ROOT_ALIASES = ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")
_DURABLE_IDENTITY_NAMES = frozenset(
    {
        ".install_token",
        ".signer_key",
        "api_key",
        "credentials.json",
    },
)

logger = logging.getLogger(__name__)


class DataRootConflictError(RuntimeError):
    """Raised when two state-bearing roots make startup ambiguous."""


def _canonical_path(value: str | Path) -> Path:
    expanded = Path(value).expanduser().resolve(strict=False)
    return Path(os.path.normcase(str(expanded)))


def environment_data_root() -> Path | None:
    """Return the winning environment root, or ``None`` when none is set."""
    for name in DATA_ROOT_ALIASES:
        value = os.environ.get(name, "").strip()
        if value:
            return _canonical_path(value)
    return None


def _legacy_configured_root(default_root: Path) -> Path | None:
    """Read the bounded legacy ``config.json:base_dir`` relocation hint.

    Only an absolute path is accepted. Relative paths historically depended on
    the caller's working directory and cannot safely identify a process
    namespace. Corrupt or unreadable configuration never changes the root.
    """
    config_path = default_root / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        value = payload.get("base_dir") if isinstance(payload, dict) else None
        if not isinstance(value, str) or not value.strip():
            return None
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            return None
        return _canonical_path(candidate)
    except (OSError, ValueError, TypeError):
        return None


def canonical_data_root(
    configured_base_dir: str | Path | None = None,
    *,
    home: str | Path | None = None,
) -> Path:
    """Resolve the one runtime-state namespace for the current process.

    Precedence:
      1. ``SLM_DATA_DIR``
      2. ``SL_MEMORY_PATH`` (legacy input shim)
      3. ``SLM_HOME`` (legacy input shim)
      4. explicit configured base directory
      5. legacy default-root ``config.json:base_dir``
      6. ``~/.superlocalmemory``

    The function performs no writes and never moves data.
    """
    selected = environment_data_root()
    if selected is not None:
        return selected
    if configured_base_dir is not None:
        return _canonical_path(configured_base_dir)
    home_path = _canonical_path(home if home is not None else Path.home())
    default_root = _canonical_path(home_path / ".superlocalmemory")
    return _legacy_configured_root(default_root) or default_root


def _durable_markers(root: Path) -> tuple[Path, ...]:
    """Return bounded evidence that ``root`` contains durable SLM state.

    The startup guard deliberately ignores ``config.json`` on its own because
    the default-root file is the compatibility locator for legacy custom
    roots. Database sidecars count even if the base file was lost.
    """
    try:
        children = tuple(root.iterdir())
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise DataRootConflictError(
            f"SuperLocalMemory cannot inspect possible state root {root}: {exc}. "
            "Startup is blocked until the root is readable or explicitly resolved."
        ) from exc
    markers = []
    for path in children:
        name = path.name
        if (
            name in _DURABLE_IDENTITY_NAMES
            or name.endswith((".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3"))
        ):
            markers.append(path)
    return tuple(sorted(markers))


def assert_no_durable_root_conflict(
    *,
    home: str | Path | None = None,
) -> None:
    """Refuse *ambiguous* startup when two state roots make the live namespace unclear.

    The root actually selected for this process is always inspected: an
    unreadable selected root fails closed. Beyond that, a conflict is only raised
    when the selection was *implicit* — resolved from the legacy
    ``config.json:base_dir`` relocation hint — and a separately-populated default
    root leaves it genuinely ambiguous which namespace is live.

    An explicit environment selection (``SLM_DATA_DIR`` / ``SL_MEMORY_PATH`` /
    ``SLM_HOME``) is an unambiguous operator contract: a separately-populated
    default root is then a deliberate multi-root / per-team / second-instance
    layout, not an ambiguity, so startup proceeds. If the explicitly chosen root
    is empty while the old default still holds data, that likely-mistyped path is
    surfaced as a warning rather than a hard block.

    This check never writes, copies, or deletes data.
    """
    home_path = _canonical_path(home if home is not None else Path.home())
    default_root = _canonical_path(home_path / ".superlocalmemory")
    selected_root = canonical_data_root(home=home_path)
    if selected_root == default_root:
        return

    # The root about to be used must be inspectable regardless of how it was
    # chosen; an unreadable selected root fails closed inside _durable_markers.
    selected_markers = _durable_markers(selected_root)

    if environment_data_root() is not None:
        # Explicit selection wins; a populated default root is a deliberate
        # multi-root layout, not an ambiguity. Only warn on the "empty new root
        # while the old default still holds data" case so a wrong SLM_DATA_DIR
        # stays visible. Inspection of the unused default never blocks startup.
        if not selected_markers:
            try:
                default_has_data = bool(_durable_markers(default_root))
            except DataRootConflictError:
                default_has_data = False
            if default_has_data:
                logger.warning(
                    "SLM_DATA_DIR selects an empty state root (%s) while the "
                    "default root (%s) still holds data; starting with the empty "
                    "root as explicitly requested.",
                    selected_root,
                    default_root,
                )
        return

    default_markers = _durable_markers(default_root)
    if not selected_markers or not default_markers:
        return
    raise DataRootConflictError(
        "Conflicting SuperLocalMemory state roots detected: "
        f"selected={selected_root} and default={default_root}. "
        "SuperLocalMemory will not merge or move either root automatically. "
        "Stop both namespaces, back them up, and choose an explicit migration "
        "or one SLM_DATA_DIR before starting the daemon."
    )


def state_path(
    *parts: str | Path,
    configured_base_dir: str | Path | None = None,
    home: str | Path | None = None,
) -> Path:
    """Return a canonical child path without permitting root escape."""
    relative = Path()
    for raw_part in parts:
        part = Path(raw_part)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError(f"state path must remain relative: {raw_part!s}")
        relative /= part
    root = canonical_data_root(configured_base_dir, home=home)
    destination = _canonical_path(root / relative)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("state path escapes the canonical data root") from exc
    return destination


class DynamicStatePath(os.PathLike[str]):
    """Backward-compatible path proxy that never freezes the active root."""

    __slots__ = ("_parts",)

    def __init__(self, *parts: str | Path) -> None:
        self._parts = parts

    def _resolve(self) -> Path:
        return state_path(*self._parts)

    def __fspath__(self) -> str:
        return str(self._resolve())

    def __truediv__(self, other: str | Path) -> Path:
        return self._resolve() / other

    def __str__(self) -> str:
        return str(self._resolve())

    def __repr__(self) -> str:
        return f"DynamicStatePath({self._parts!r})"

    def __eq__(self, other: object) -> bool:
        try:
            return self._resolve() == Path(other)  # type: ignore[arg-type]
        except TypeError:
            return False

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)
