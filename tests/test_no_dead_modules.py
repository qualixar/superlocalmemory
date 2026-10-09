"""Guard against shipping modules that exist, are tested, and are never wired.

WHY THIS EXISTS
---------------
`code_graph/resolver.py` shipped fully written, with a passing test file, and
was never imported by any production module. The parser therefore emitted
placeholder edges that no resolution step ever cleaned up, they reached
GraphStore with endpoints absent from graph_nodes, and every `build_code_graph`
call died with "FOREIGN KEY constraint failed". The code graph was empty from
2026-07-24 until 4.0.6.

CI was green the whole time, because "has tests" was being mistaken for
"is wired". This test closes that gap: a src module imported ONLY by its own
test is dead in production, no matter how well tested it is.

Audit at the time this guard was added found ~1,650 such lines:
  mcp/cli_fallback.py                (602)  imported by NOTHING, not even a test
  core/fact_consolidator.py          (598)  only its own test
  code_graph/git_hooks.py            (226)  only its own test
  integrations/bounded_loops_v051.py (236)  only its own test, superseded by
                                            integrations/bounded_loops_mcp.py

Migrations are excluded: storage/migrations/M0*.py are loaded dynamically via
importlib.util.spec_from_file_location in storage/migrations/__init__.py, so a
static import scan cannot see them and would report every one as dead.

THE HOLE THIS GUARD SHIPPED WITH (closed after 4.0.6)
-----------------------------------------------------
The module-level scan below skips every `__init__.py` when choosing what to
CHECK, but counts `__init__.py` files when deciding what has an IMPORTER. So a
package whose `__init__.py` re-exports its own submodules made every one of
those submodules look wired — by itself — while the package as a whole was
imported by nothing. The package was then invisible from both directions.

That is not hypothetical. `summaries/` shipped in 4.0.6 in exactly that shape:
three generators written for issue #113, seventeen passing tests, an
`__init__.py` re-exporting all three, and no command, tool or endpoint anywhere
that calls the package. It passed this guard for the whole release.

`test_no_new_dead_packages` closes it: a package must be imported by production
code OUTSIDE itself. Same rule as a module, one level up.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "superlocalmemory"
_TESTS = _REPO / "tests"

#: Modules known to be dead, kept passing so the guard can be introduced without
#: a mass deletion mid-release. Each MUST be either wired up or deleted, and its
#: entry removed. Shrinking this list is the point; adding to it needs a reason.
#: RATCHET MODE. Seeded 4.0.6 with everything the static scan currently flags,
#: so the guard passes today and its real job is blocking NEW dead modules from
#: this point on — which is exactly what would have stopped resolver.py.
#:
#: Entries here are NOT all confirmed dead. This scan sees only static imports,
#: so a module invoked dynamically looks identical to one nobody calls. Verified
#: examples of each:
#:   ALIVE  core/embedding_worker.py, core/recall_worker.py — spawned by
#:          infra/self_heal.py via the strings "embedding_worker.py" /
#:          "recall_worker.py"
#:   DEAD   mcp/cli_fallback.py (602 loc, no importer at all, not even a test)
#:          integrations/bounded_loops_v051.py (superseded by
#:          integrations/bounded_loops_mcp.py, the live slm-bridge/v1 path)
#:   FIXED  core/fact_consolidator.py — was listed here as DEAD (598 loc, only
#:          its own test). Wired into core/maintenance.py in 4.0.6 for issue
#:          #113, so its entry is gone from the dict above. This guard did its
#:          job: it named the module, and the module got wired rather than
#:          quietly shipping unreferenced for another release.
#:
#: This list may only SHRINK. test_known_dead_list_has_no_stale_entries forces
#: an entry out as soon as the module gains a production importer.
#: TODO(4.0.7): triage the seeded entries — teach the scan about dynamic
#: invocation (subprocess "-m", route registries), then wire or delete the rest.
_KNOWN_DEAD: dict[str, str] = {
    "code_graph/git_hooks.py": "seeded 4.0.6 — triage: wire or delete",
    "code_graph/incremental.py": "seeded 4.0.6 — triage",
    "code_graph/watcher.py": "seeded 4.0.6 — triage",
    "core/embedding_worker.py": "ALIVE — subprocess-spawned by infra/self_heal.py",
    "core/engine_lock.py": "seeded 4.0.6 — triage",
    "core/recall_worker.py": "ALIVE — subprocess-spawned by infra/self_heal.py",
    "core/reranker_worker.py": "ALIVE — subprocess reranker backend",
    "core/laya_worker.py": "ALIVE — run by file path in Laya's own Python, which cannot "
                           "import this package (retrieval/sufficiency.py, core/laya_runtime.py)",
    "runtimes/multimodal_worker.py": "ALIVE — run by file path inside the managed media environment, "
                                     "which cannot import this package (runtimes/worker_client.py)",
    "dynamics/activation_guided_quantization.py": "seeded 4.0.6 — triage",
    "ingestion/calendar_adapter.py": "seeded 4.0.6 — triage",
    "ingestion/gmail_adapter.py": "seeded 4.0.6 — triage",
    "ingestion/transcript_adapter.py": "seeded 4.0.6 — triage",
    "integrations/bounded_loops_v051.py": "DEAD — superseded by bounded_loops_mcp.py",
    "learning/bootstrap.py": "seeded 4.0.6 — triage",
    "learning/project_context.py": "seeded 4.0.6 — triage",
    "mcp/cli_fallback.py": "DEAD — no importer at all, not even a test",
    "mcp/tools.py": "seeded 4.0.6 — triage",
    "optimize/metrics/exporters.py": "seeded 4.0.6 — triage",
    "server/ui.py": "seeded 4.0.6 — triage",
    "storage/migration_v33.py": "seeded 4.0.6 — triage",
}

#: Packages imported by nothing outside themselves. Same ratchet as
#: _KNOWN_DEAD: seeded with what the scan finds today so the guard passes, and
#: its real job is blocking the NEXT one. Every entry states why it is here.
#: TODO(4.0.7): wire or delete all four.
_KNOWN_DEAD_PACKAGES: dict[str, str] = {
    "attribution": "seeded 4.0.6 — triage: provenance signer/watermark, no caller",
    # code_graph/bridge: REMOVED in 4.0.7 — wired into core/maintenance.py as a
    # background pass. This guard is what forced the entry out: the ratchet test
    # failed the moment the package gained a production importer, which is
    # exactly the behaviour it was added for.
    "evaluation": "INTENTIONAL — calibration artifacts, documented as isolated "
                  "from production retrieval paths. Not a defect.",
    # summaries: REMOVED in 4.0.7 — reachable via `slm summary`
    # (cli/summary_cmd.py). Seeded one release earlier precisely because the
    # generators shipped with no caller; the ratchet forced the entry out as soon
    # as a real surface existed, which is what it is for.
}

#: Dynamically loaded, invisible to a static scan.
_DYNAMIC_PREFIXES = ("storage/migrations/",)

#: Interpreter entry points. `__main__.py` is executed by `python -m pkg` and is
#: never imported by anything — that is the idiom, not dead code. an earlier stage's
#: compliance CLI gate invokes `python -m superlocalmemory.cli` directly.
_ENTRYPOINT_NAMES = ("__main__.py",)


def _module_names_imported_by(paths: list[pathlib.Path]) -> set[str]:
    """Collect every module name referenced by an import statement in *paths*."""
    seen: set[str] = set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    seen.update(alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    seen.update(node.module.split("."))
                for alias in node.names:
                    seen.add(alias.name)
    return seen


def _src_modules() -> list[pathlib.Path]:
    out = []
    for path in _SRC.rglob("*.py"):
        if path.name in ("__init__.py", *_ENTRYPOINT_NAMES) or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_SRC).as_posix()
        if rel.startswith(_DYNAMIC_PREFIXES):
            continue
        out.append(path)
    return sorted(out)


@pytest.fixture(scope="module")
def _importers_by_stem() -> dict[str, set[pathlib.Path]]:
    """Map module stem -> set of production files importing it.

    Built in ONE pass. The earlier per-module rescan was O(n^2) and took six
    minutes on this repo, which is slow enough that nobody would run it.
    """
    index: dict[str, set[pathlib.Path]] = {}
    for path in _SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for name in _module_names_imported_by([path]) | _routers_loaded_by_name(path):
            index.setdefault(name, set()).add(path)
    return index


def _routers_loaded_by_name(path: pathlib.Path) -> set[str]:
    """Route modules a daemon imports by name in its optional-router loop
    (``for _mod_name in ("learning", ..., "integrity"): __import__(...)``).
    A module named there IS wired; one that is missing from the loop is not."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "superlocalmemory.server.routes.{_mod_name}" not in text:
        return set()
    import re

    names: set[str] = set()
    for group in re.findall(r"for _mod_name in \(([^)]*)\)", text):
        names.update(re.findall(r'"([a-z_]+)"', group))
    return names


def test_no_new_dead_modules(_importers_by_stem) -> None:
    """A src module must be imported by production code, not only by its test."""
    dead: list[str] = []
    for path in _src_modules():
        importers = _importers_by_stem.get(path.stem, set()) - {path}
        if importers:
            continue
        dead.append(path.relative_to(_SRC).as_posix())

    unexpected = sorted(set(dead) - set(_KNOWN_DEAD))
    assert not unexpected, (
        "These src modules are imported by NO production code — they are dead at "
        "runtime no matter how well tested:\n  "
        + "\n  ".join(unexpected)
        + "\n\nThis is the defect class that left code_graph/resolver.py unwired "
        "and the code graph empty for six weeks with CI green. Wire the module "
        "into the code path that needs it, delete it, or (last resort, with a "
        "reason) add it to _KNOWN_DEAD in this file."
    )


def _packages() -> list[pathlib.Path]:
    """Every package directory under src, excluding the root package itself."""
    return sorted(
        p.parent
        for p in _SRC.rglob("__init__.py")
        if p.parent != _SRC and "__pycache__" not in p.parts
    )


@pytest.fixture(scope="module")
def _imports_by_file() -> dict[pathlib.Path, set[str]]:
    """Parse every src file ONCE: path -> names it imports.

    Deliberately a fixture and not a per-package rescan. The module-level scan
    above already paid for that lesson — an O(n^2) rglob-and-parse per target
    took six minutes and nobody ran it. One pass, then set lookups.
    """
    return {
        f: _module_names_imported_by([f])
        for f in _SRC.rglob("*.py")
        if "__pycache__" not in f.parts
    }


def _package_importers(
    pkg: pathlib.Path, imports_by_file: dict[pathlib.Path, set[str]]
) -> set[pathlib.Path]:
    """Production files OUTSIDE *pkg* that name it in an import statement.

    "Outside" is what makes this different from the module scan: a package's
    own ``__init__.py`` re-exporting its submodules is not evidence that
    anything uses the package. That self-reference is the hole this closes.
    """
    return {
        f for f, names in imports_by_file.items()
        if pkg.name in names and pkg not in f.parents and f.parent != pkg
    }


def test_no_new_dead_packages(_imports_by_file) -> None:
    """A package must be imported by production code outside itself."""
    dead = [
        pkg.relative_to(_SRC).as_posix()
        for pkg in _packages()
        if not _package_importers(pkg, _imports_by_file)
    ]

    unexpected = sorted(set(dead) - set(_KNOWN_DEAD_PACKAGES))
    assert not unexpected, (
        "These packages are imported by NO production code outside themselves. "
        "Their submodules look wired only because the package's own "
        "__init__.py re-exports them:\n  "
        + "\n  ".join(unexpected)
        + "\n\nThis is how summaries/ shipped in 4.0.6 unreachable — three "
        "generators, seventeen passing tests, and no way for a user to call "
        "any of them. Give the package a caller (command, tool, route, or "
        "scheduled job), delete it, or add it to _KNOWN_DEAD_PACKAGES with a "
        "reason."
    )


def test_known_dead_packages_has_no_stale_entries(_imports_by_file) -> None:
    """Once a known-dead package gains an outside caller, force the entry out."""
    now_wired = [
        rel for rel in sorted(_KNOWN_DEAD_PACKAGES)
        if (_SRC / rel).exists()
        and _package_importers(_SRC / rel, _imports_by_file)
    ]
    assert not now_wired, (
        "These packages are now imported by production code — remove them from "
        "_KNOWN_DEAD_PACKAGES so the guard stays meaningful:\n  "
        + "\n  ".join(now_wired)
    )


def test_known_dead_list_has_no_stale_entries(_importers_by_stem) -> None:
    """Once a known-dead module gets wired up, force its entry to be removed.

    Keeps the allowlist honest — otherwise it silently becomes permanent.
    """
    now_wired = []
    for rel in sorted(_KNOWN_DEAD):
        path = _SRC / rel
        if not path.exists():
            continue  # deleted — the good outcome
        if _importers_by_stem.get(path.stem, set()) - {path}:
            now_wired.append(rel)
    assert not now_wired, (
        "These modules are now imported by production code — remove them from "
        "_KNOWN_DEAD so the guard stays meaningful:\n  " + "\n  ".join(now_wired)
    )
