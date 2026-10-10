"""Layering debt may shrink but never grow.

``core`` must not import ``server`` or ``cli``, and ``infra`` must not import
``server``. Existing violations are frozen in the baselines below (and mirrored
in ``.importlinter``); a new one fails here on every platform, without any extra
tool. Function-level imports count: they are in the import graph too.
"""

from __future__ import annotations

import ast
import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_PKG = _SRC / "superlocalmemory"
_CONFIG = _REPO / ".importlinter"

_RATCHETS = {
    "core-to-server-ratchet": ("core", "superlocalmemory.server"),
    "core-to-cli-ratchet": ("core", "superlocalmemory.cli"),
    "infra-to-server-ratchet": ("infra", "superlocalmemory.server"),
}

BASELINES: dict[str, frozenset[tuple[str, str]]] = {
    "core-to-server-ratchet": frozenset({
        ("superlocalmemory.core.admission", "superlocalmemory.server.profile_runtime"),
        ("superlocalmemory.core.answer_check_state", "superlocalmemory.server.config_file"),
        ("superlocalmemory.core.embedding_reindex", "superlocalmemory.server.profile_runtime"),
        ("superlocalmemory.core.embedding_reindex_activate", "superlocalmemory.server.profile_runtime"),
        ("superlocalmemory.core.ops_remediation", "superlocalmemory.server.unified_daemon"),
        ("superlocalmemory.core.recall_worker", "superlocalmemory.server.recall_serializer"),
        ("superlocalmemory.core.remote_mode", "superlocalmemory.server.loopback"),
    }),
    "core-to-cli-ratchet": frozenset({
        ("superlocalmemory.core.component_healer", "superlocalmemory.cli.setup_wizard"),
        ("superlocalmemory.core.component_registry", "superlocalmemory.cli.setup_wizard"),
        ("superlocalmemory.core.config", "superlocalmemory.cli._lazy_init"),
        ("superlocalmemory.core.engine", "superlocalmemory.cli.pending_store"),
        ("superlocalmemory.core.maintenance_scheduler", "superlocalmemory.cli.pending_store"),
        ("superlocalmemory.core.mcp_embedder_proxy", "superlocalmemory.cli.daemon"),
    }),
    "infra-to-server-ratchet": frozenset({
        ("superlocalmemory.infra.auth_middleware", "superlocalmemory.server.loopback"),
        ("superlocalmemory.infra.event_bus", "superlocalmemory.server.routes.helpers"),
    }),
}


def _module_name(path: Path) -> tuple[str, bool]:
    rel = path.relative_to(_SRC).with_suffix("")
    parts = list(rel.parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts.pop()
    return ".".join(parts), is_pkg


def _targets(node: ast.AST, module: str, is_pkg: bool) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    base = node.module or ""
    if node.level:
        package = module.split(".") if is_pkg else module.split(".")[:-1]
        package = package[: len(package) - (node.level - 1)]
        base = ".".join(package + ([node.module] if node.module else []))
    return [base] + [f"{base}.{alias.name}" for alias in node.names]


def _scan(layer: str, forbidden: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted((_PKG / layer).rglob("*.py")):
        module, is_pkg = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for target in _targets(node, module, is_pkg):
                if target == forbidden or target.startswith(forbidden + "."):
                    found.add((module, _owning_module(target)))
    return found


def _owning_module(target: str) -> str:
    """Reduce ``pkg.mod.name`` to the module that exists on disk."""
    parts = target.split(".")
    while parts:
        base = _SRC.joinpath(*parts)
        if base.with_suffix(".py").exists() or (base / "__init__.py").exists():
            return ".".join(parts)
        parts.pop()
    return target


def _ini_pairs(section: str) -> set[tuple[str, str]]:
    parser = configparser.ConfigParser()
    parser.read(_CONFIG, encoding="utf-8")
    raw = parser.get(f"importlinter:contract:{section}", "ignore_imports", fallback="")
    pairs = set()
    for line in raw.splitlines():
        if "->" in line:
            left, right = (side.strip() for side in line.split("->"))
            pairs.add((left, right))
    return pairs


@pytest.mark.parametrize("contract", sorted(_RATCHETS))
def test_no_new_upward_imports(contract: str) -> None:
    layer, forbidden = _RATCHETS[contract]
    new = _scan(layer, forbidden) - BASELINES[contract]
    assert not new, (
        f"{layer} must not import {forbidden}; move the shared code down or "
        f"inject it. New imports: {sorted(new)}"
    )


@pytest.mark.parametrize("contract", sorted(_RATCHETS))
def test_ratchet_only_shrinks(contract: str) -> None:
    layer, forbidden = _RATCHETS[contract]
    scanned = _scan(layer, forbidden)
    stale = BASELINES[contract] - scanned
    assert not stale, (
        f"{sorted(stale)} no longer exist; remove them from BASELINES and "
        f".importlinter so the list stays honest"
    )
    ini = _ini_pairs(contract)
    assert ini == set(BASELINES[contract]), (
        f".importlinter ignore_imports for {contract} must equal the baseline"
    )


def test_importlinter_config_parses() -> None:
    parser = configparser.ConfigParser()
    assert parser.read(_CONFIG, encoding="utf-8"), ".importlinter is missing"
    assert parser.get("importlinter", "root_package") == "superlocalmemory"
    checked = 0
    for section in parser.sections():
        if not section.startswith("importlinter:contract:"):
            continue
        for key in ("source_modules", "forbidden_modules"):
            for name in parser.get(section, key, fallback="").split():
                if not name.startswith("superlocalmemory."):
                    continue
                base = _SRC.joinpath(*name.split("."))
                assert base.is_dir() or base.with_suffix(".py").exists(), (
                    f"{section}: {name} does not exist under src/"
                )
                checked += 1
    assert checked


def test_lint_imports_contracts_kept() -> None:
    pytest.importorskip("importlinter")
    exe = Path(sys.executable).parent / "lint-imports"
    command = [str(exe)] if exe.exists() else (
        [shutil.which("lint-imports")] if shutil.which("lint-imports") else None
    )
    if command is None:
        pytest.skip("lint-imports console script not found")
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    result = subprocess.run(
        command, cwd=_REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
