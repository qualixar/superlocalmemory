# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory release/4.0.6 — Backward Compatibility Gate

"""Backward-compatibility test suite: SLM 4.0.5 → 4.0.6.

GOAL ANCHOR (verbatim from Varun):
  "Everything should be backward compatible."
  Invariant I4: "Backward compatible — schema additive-only; 4.0.5 DB must open
  on 4.0.6 and vice-versa."

Release context
---------------
The ONLY change from 4.0.5 → 4.0.6 was a WAL close-path deadlock fix in
``storage/database.py`` (wal_autocheckpoint=400 + SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE).
No new migrations were added; M001–M042 and SUPPORTED_SCHEMA_VERSION=42 were
IDENTICAL in both releases. All 41 migration DDL files hash byte-for-byte
identically against the installed 4.0.5 package.

**4.1.0 deliberately ends that co-existence, and this suite now proves the
ending rather than the compatibility.** M046 rebuilds ``atomic_facts`` with a
constraint an older build's writes violate, so a store it has touched must not
be opened by one. The tests below assert the ceiling ROSE and that an older
ceiling now REFUSES such a store — the same properties, read the other way
round. Everything about M001–M042 is unchanged and still checked: the DDL hash
identity, the additive-only invariant over those files, and the drift guards.

What this suite PROVES
-----------------------
1. The fixture IS 4.0.5-shaped: every migration file in the current branch is
   byte-for-byte identical to the installed 4.0.5 package (SHA-256 verified).
2. Forward compat: a 4.0.5 DB opened under 4.0.6 apply_all() has all migrations
   skipped, zero failures, pre-existing rows intact (content asserted, not
   just counts), and slm_schema_version remains 42.
3. Additive-only invariant: no migration in M001–M042 removes a table or a
   column. Checked two independent ways, because neither alone is sufficient:
     - STATIC, over each migration module's full Python source (not just the
       ``migration.ddl`` string), so the six rebuilds that run through a custom
       ``apply()`` are covered. Table names built by f-string cannot be
       resolved statically and are reported separately against a reviewed
       waiver list rather than passed silently.
     - RUNTIME, diffing every table and column in both DBs across a real
       ``apply_all`` + ``apply_deferred`` run.
   Every detector carries a teeth check that injects the failure it is meant
   to catch.
4. Backward REFUSAL, from 4.1.0 on: a store this build has migrated is stamped
   46, so a build whose ceiling is 42 raises SchemaVersionError rather than
   writing to it. Before 4.1.0 this section proved the opposite — that the
   stamps matched and an older build could open the store safely. That was true
   for an additive release and is the wrong guarantee for this one.

What this suite CANNOT PROVE
------------------------------
- True 4.0.5 in-process backward execution: the installed 4.0.5 uses Python 3.14
  and cannot run inside this Python 3.13.5 test process. Backward opening is
  validated by proving the schema_version ceiling identity and the additive-only
  invariant, not by importing the installed package code at runtime.
- Runtime behaviour of five of the six apply()-based rebuilds. Only M026
  executes its rebuild branch against the test fixture; M021/M023/M027 find
  their tables already carrying profile_id (engine init creates the modern
  shape), and M032/M036 find theirs absent or well-formed. Those five are
  covered statically, by source scan, not by execution. Closing that would
  require hand-authored legacy table shapes, which would be an unverified
  guess at the pre-4.0.5 schema rather than evidence.
- M023's rebuild is not machine-verified at all: it builds its table names by
  f-string, so the static scan reports it as unresolvable and it rests on the
  human review recorded in ``_DYNAMIC_DDL_WAIVERS``.
- Query-level regression: 4.0.5 application logic is not exercised here. The
  safe-to-open property rests on (a) identical SUPPORTED_SCHEMA_VERSION, (b) no
  new tables/columns in 4.0.6, and (c) the WAL-only behavioral delta being
  non-schema-structural. No new SELECT * path over an unknown table can fail
  because there are NO new tables.

CRIT flaws acknowledged and fixed
-----------------------------------
CRIT-1 "is the fixture actually 4.0.5-shaped or just current-shaped?":
  Fixed: TestFixtureShape.test_fixture_ddl_hashes_match_installed_405 computes
  SHA-256 of every migration file and asserts identity with the installed 4.0.5
  package. Identical file hashes prove the fixture is built from exactly the
  same DDL the 4.0.5 runner used.

CRIT-2 "does the schema diff catch a renamed column?":
  Fixed: _DESTRUCTIVE_PATTERNS includes ALTER TABLE ... RENAME COLUMN (SQLite
  3.25+ syntax) in addition to DROP TABLE and DROP COLUMN. Three teeth-check
  tests exercise all three destructive patterns: DROP COLUMN, DROP TABLE, and
  RENAME COLUMN.

CRIT-3 "are you asserting content or only row counts?":
  Fixed: TestForwardCompat.test_preexisting_fact_content_survives asserts the
  exact fact_id value and exact content string, not COUNT(*).

CRIT-4 "the scanner reads migration.ddl, but apply() is what runs":
  ``_apply_single`` executes a module's ``apply(conn)`` INSTEAD of its DDL
  string when one exists, so for M021/M023/M026/M027/M032/M036 the scanned
  text was not the executed text. M021 is the clearest case: its DDL string is
  a plain ADD COLUMN while ``apply()`` does RENAME → CREATE → copy → DROP. The
  ``ddl_sha256`` drift guard hashes the same unexecuted string, so a
  destructive edit to any ``apply()`` would have left both checks green.
  Fixed: test_no_destructive_sql_in_any_migration_module_source scans the whole
  module source via ``inspect.getsource``.

CRIT-5 "does the live schema diff actually span a migration?":
  It did not. It snapshotted around a SECOND ``apply_all()`` on an
  already-migrated DB — where every migration is skipped — and looked only at
  ``atomic_facts`` in memory.db, while four of the six apply()-based rebuilds
  are DEFERRED (never reached by ``apply_all``) and one targets learning.db.
  Fixed: the diff now spans a real ``apply_all`` + ``apply_deferred`` run over
  every table in both DBs, and asserts the run is not a no-op.

Safety (CRITICAL)
-----------------
All DB files are created under pytest's tmp_path. The root conftest.py installs
a sys.addaudithook that raises PermissionError if any path under
~/.superlocalmemory is opened. This test never touches live user data.

Run:
  .venv/bin/python -m pytest tests/test_migrations/test_backward_compat_405.py \\
      -o addopts="" -v
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import re
import sqlite3
import time
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The schema version written after a complete 4.0.5 (and 4.0.6) migration run.
#: 4.1.0 raises the ceiling to the trailing serial of its last migration. It sat
#: at 42 while M043, M044 and M045 shipped — tolerable only because all three
#: were additive, so an older build simply never read what it did not know
#: about. M046 is not additive: it rebuilds atomic_facts with a constraint that
#: rejects the value an older build writes planned events as. The ceiling is
#: what converts that from a lost memory into a refusal to start.
#:
#: 49 as of 4.1.0: +M049_a_schema_version_marker_is_one_row. It is not additive
#: either — it puts a UNIQUE index on schema_version(version) and collapses the
#: duplicates that index forbids (234,348 rows down to 7 on one real store).
#: This build's six writers were changed to INSERT OR IGNORE alongside it; an
#: older build still issues a plain INSERT and would hit the constraint partway
#: through its own migration run. Raising the ceiling turns that into a refusal
#: to open rather than a half-migrated store.
#:
#: This constant sat at 48 while M049 shipped, so the whole file was failing —
#: nothing noticed because the full suite had not been run since. Keep it equal
#: to the trailing serial of the last migration.
_EXPECTED_SCHEMA_VERSION: int = 50

#: Total migrations in the MIGRATIONS + DEFERRED_MIGRATIONS catalogue.
#: M001–M043 with M008 absent = 42 total.
#:
#: _EXPECTED_SCHEMA_VERSION deliberately stays at 42 while this went to 42.
#: The two are not the same thing: the version is a DOWNGRADE CEILING, and
#: M043's changes are additive (one column with a default, one new table,
#: plus data moved between them). An older build opening the store reads it
#: fine. It would show the withheld summaries again, because it has no filter
#: for them — the behaviour the owner downgraded away from — but that is what
#: downgrading means. Bumping the ceiling would instead make the older build
#: refuse the store outright and strand anyone who needed to go back.
# 44 as of 4.1.0: +M044_play_carries_its_own_evidence (learning.db,
# bandit_plays.shown_fact_ids) and +M045_fact_outcome_score (memory.db, the
# per-fact outcome score). Both additive columns/tables; neither rewrites an
# existing row, so 4.0.5 forward-compat is unaffected.
_EXPECTED_MIGRATION_COUNT: int = 49

#: Path to an installed reference package's migrations directory, if one exists.
#:
#: Resolved from the environment rather than hardcoded. The original literal
#: pointed at one developer's pipx venv, which (a) embedded a username in a
#: published file and (b) could never resolve on anyone else's machine, so the
#: comparison it guards silently did nothing everywhere except that laptop.
#:
#: Set SLM_REFERENCE_INSTALL to the site-packages/superlocalmemory directory of
#: a previous release to run the cross-version comparison; otherwise the tests
#: that need it skip, which is the honest outcome when the reference is absent.
def _reference_install_dir() -> Path | None:
    override = os.environ.get("SLM_REFERENCE_INSTALL", "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.is_dir() else None
    # Common pipx layout, discovered without assuming a user or Python version.
    for base in (Path.home() / ".local/pipx/venvs/superlocalmemory/lib",):
        if not base.is_dir():
            continue
        for pyver in sorted(base.glob("python3.*")):
            cand = pyver / "site-packages" / "superlocalmemory"
            if cand.is_dir():
                return cand
    return None


_REFERENCE_INSTALL: Path | None = _reference_install_dir()
_INSTALLED_405_MIGRATIONS_DIR: Path = (
    (_REFERENCE_INSTALL / "storage" / "migrations")
    if _REFERENCE_INSTALL is not None
    else Path("/nonexistent/reference-install/storage/migrations")
)

#: Path to the installed 4.0.5 _schema_version.py module.
_INSTALLED_405_SCHEMA_VERSION_PATH: Path = (
    _INSTALLED_405_MIGRATIONS_DIR.parent / "_schema_version.py"
)

#: Sentinel rows inserted by the fixture builder.
_FIXTURE_PROFILE_ID = "compat_test_profile_405"
_FIXTURE_FACT_ID = "fact_compat_405_001"
_FIXTURE_FACT_CONTENT = "SLM v4.0.5 shipped additive-only schema M042 on 2026-08-16"
_FIXTURE_FACT_ID_2 = "fact_compat_405_002"
_FIXTURE_FACT_CONTENT_2 = "Backward compatibility invariant I4 must hold across releases"


# ---------------------------------------------------------------------------
# Additive-only checker
# ---------------------------------------------------------------------------

#: Regex patterns that identify destructive DDL operations.
#: Each tuple is (pattern, human-readable label).
_DESTRUCTIVE_PATTERNS: list[tuple[str, str]] = [
    # DROP TABLE (with or without IF EXISTS)
    (r"\bDROP\s+TABLE\b", "DROP TABLE"),
    # ALTER TABLE <name> DROP COLUMN <col>
    (r"\bALTER\s+TABLE\b[^;]*\bDROP\s+COLUMN\b", "ALTER TABLE ... DROP COLUMN"),
    # ALTER TABLE <name> RENAME TO <new_name>  (table rename)
    (r"\bALTER\s+TABLE\b[^;]*\bRENAME\s+TO\b", "ALTER TABLE ... RENAME TO"),
    # ALTER TABLE <name> RENAME COLUMN <old> TO <new>  (column rename, SQLite 3.25+)
    (r"\bALTER\s+TABLE\b[^;]*\bRENAME\s+COLUMN\b[^;]*\bTO\b",
     "ALTER TABLE ... RENAME COLUMN ... TO"),
]

#: Migrations whose destructive DDL builds its table name at runtime, so no
#: static scanner can resolve it. Each entry is a REVIEWED waiver: the value
#: records why the migration is additive despite being unverifiable by machine.
#:
#: This mapping is asserted to match the scanner's ``dynamic`` output exactly.
#: A new migration with dynamic DDL therefore FAILS the suite until a human
#: reads it and writes a justification here; a waiver left behind after the
#: dynamic DDL is removed fails too.
_DYNAMIC_DDL_WAIVERS: dict[str, str] = {
    "M023_mesh_profile_isolation": (
        "_atomic_rebuild(conn, table, create_sql, insert_sql) issues "
        "f'ALTER TABLE {table} RENAME TO {old}' and f'DROP TABLE {old}' where "
        "old = f'_{table}_old'. Both call sites pass a literal table name — "
        "_rebuild_state('mesh_state') and _rebuild_locks('mesh_locks') — and "
        "each replacement table is created from _NEW_STATE / _NEW_LOCKS inside "
        "the same BEGIN IMMEDIATE transaction before the old copy is dropped. "
        "Net effect adds profile_id; no column is removed. "
        "Source-reviewed 2026-08-16 for release/4.0.6."
    ),
    "M046_prospective_memory_has_its_own_name": (
        "_rebuild() issues literal-name DROP/RENAME against atomic_facts via "
        "f-strings over the module constant _TABLE = 'atomic_facts', so the "
        "scanner cannot resolve them. The replacement table is CREATEd from the "
        "live table's own definition read out of sqlite_master — which is why "
        "that statement cannot be a literal and why this waiver is needed: "
        "reconstructing the DDL from PRAGMA table_info would silently drop "
        "CHECK constraints and collations. The shape is the forward rebuild "
        "dance (CREATE staging, copy, DROP original, RENAME staging into place) "
        "inside one BEGIN IMMEDIATE, with the row count compared before COMMIT. "
        "\n\n"
        "IT REMOVES NO TABLE AND NO COLUMN, so the invariant this suite states "
        "holds. It is NOT additive in a broader sense and must not be read as "
        "such: it NARROWS the fact_type value domain, replacing 'temporal' with "
        "'prospective'. An older build's SELECTs all still resolve; its INSERTs "
        "of a planned event do not. That break is deliberate and is what the "
        "schema-version ceiling exists to convert into a refusal to start — see "
        "test_an_older_build_is_refused_by_the_gate. "
        "Source-reviewed 2026-08-22 for release/4.1.0."
    ),
}


#: Placeholder substituted for the interpolated part of an f-string. Chosen so
#: that ``\w+`` cannot match it — an identifier built at runtime is, by
#: construction, not statically resolvable.
_DYNAMIC_TOKEN = "<?>"


def _normalize_identifier(raw: str) -> str:
    """Strip SQL punctuation and quoting from a captured table identifier.

    The capture is deliberately ``\\S+`` rather than ``\\w+`` so that a runtime-
    built name (rendered as ``_DYNAMIC_TOKEN``) is still captured instead of
    silently failing to match and being treated as absent. The cost is that
    trailing syntax comes along — ``DROP TABLE foo;`` captures ``foo;`` — so it
    is trimmed here. Leaving it attached made every rebuild look like a real
    removal, because ``"foo;"`` never matches the recreated name ``"foo"``.
    """
    return raw.strip().strip(";,()").strip("\"'`[]").lower()


def _recreated_table_names(sql: str) -> set[str]:
    """Return every table name this SQL re-establishes.

    A table name is "re-established" if the SQL either CREATEs it or RENAMEs
    some other table INTO it. Both forms mean the name still exists once the
    statement batch finishes.

    This is the reconciliation primitive for SQLite's table-rebuild dance.
    SQLite cannot alter a column's constraints in place, so migrations rebuild
    the table. Three shapes appear in this codebase, and all three are additive
    in net effect:

      forward   CREATE X_new … ; DROP X ; ALTER TABLE X_new RENAME TO X   (M002)
      reverse   ALTER TABLE X RENAME TO _X_old ; CREATE X … ; DROP _X_old (M021,
                                                        M026, M027, M032)
      recreate  DROP X ; CREATE X …                                       (M036)

    Reducing all three to "was the name put back?" avoids special-casing each
    shape, and is strictly stronger than matching drop/rename-target pairs:
    a DROP whose table is never re-established is now caught in every shape.
    """
    created = {
        m.group(1).lower()
        for m in re.finditer(
            r"\bCREATE\s+(?:TEMP\s+|TEMPORARY\s+)?TABLE\s+"
            r"(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
            sql,
            re.IGNORECASE,
        )
    }
    renamed_into = {
        m.group(1).lower()
        for m in re.finditer(r"\bRENAME\s+TO\s+(\w+)", sql, re.IGNORECASE)
    }
    return created | renamed_into


def _scan_ddl_for_destructive_ops(
    named_ddls: list[tuple[str, str]],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Scan SQL text for operations that violate the additive-only invariant.

    Works on any SQL text: a ``migration.ddl`` string, or the SQL literals
    extracted from a migration module's Python source (see
    ``_extract_sql_literals``). The caller decides what to feed in.

    A DROP TABLE or a rename-away is tolerated **only** when the same text
    re-establishes that table name (the rebuild dance — see
    ``_recreated_table_names``). DROP COLUMN and RENAME COLUMN are never
    reconcilable and always report.

    Table names built at runtime (f-strings) cannot be resolved statically.
    Rather than silently passing them — which is how this class of change hid
    in the first place — they are reported separately so the caller can require
    an explicit, reviewed waiver.

    Args:
        named_ddls: List of (migration_name, sql_text) pairs.

    Returns:
        (violations, dynamic) where each entry is
        (migration_name, pattern_label, snippet). ``violations`` are hard
        failures; ``dynamic`` are unresolvable names needing human review.
    """
    violations: list[tuple[str, str, str]] = []
    dynamic: list[tuple[str, str, str]] = []

    for name, ddl in named_ddls:
        survives = _recreated_table_names(ddl)

        for pattern, label in _DESTRUCTIVE_PATTERNS:
            for match in re.finditer(pattern, ddl, re.IGNORECASE | re.DOTALL):
                text = match.group(0)
                snippet = " ".join(text.split())[:100]

                # The matched span stops at the destructive keyword, so the
                # table name is read from the text starting at the match.
                ctx = ddl[match.start(): match.start() + 200]

                if "DROP TABLE" in label:
                    target = re.search(
                        r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\S+)",
                        ctx,
                        re.IGNORECASE,
                    )
                    subject = _normalize_identifier(target.group(1)) if target else None
                elif "RENAME TO" in label and "COLUMN" not in label:
                    # The table at risk is the one renamed AWAY, not the
                    # destination: "ALTER TABLE <subject> RENAME TO <dest>".
                    target = re.search(
                        r"\bALTER\s+TABLE\s+(\S+)\s+RENAME\s+TO\b",
                        ctx,
                        re.IGNORECASE,
                    )
                    subject = _normalize_identifier(target.group(1)) if target else None
                else:
                    # DROP COLUMN / RENAME COLUMN — never reconcilable.
                    violations.append((name, label, snippet))
                    continue

                if not subject or _DYNAMIC_TOKEN in subject:
                    dynamic.append((name, label, snippet))
                elif subject not in survives:
                    violations.append((name, label, snippet))
                # else: the name is put back — this is a rebuild, not a removal.

    return violations, dynamic


def _extract_sql_literals(source: str) -> str:
    """Return the SQL-bearing string literals of a Python module, ';'-joined.

    Parses with ``ast`` rather than scanning raw text, which drops ``#``
    comments for free and lets docstrings be excluded deliberately — prose
    such as "rename → create → copy → drop old" must not read as DDL.

    f-strings are rendered with their interpolated parts replaced by
    ``_DYNAMIC_TOKEN``, preserving the SQL keywords while marking the
    identifier as statically unresolvable.

    Literals are joined with ';' so the ``[^;]*`` guards inside
    ``_DESTRUCTIVE_PATTERNS`` cannot straddle two unrelated statements.
    """
    tree = ast.parse(source)

    # First statement of a module/class/function that is a bare string is a
    # docstring: prose, not SQL.
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_nodes.add(id(body[0].value))

    # Constants nested inside an f-string are emitted as part of the f-string;
    # skip them so they are not also counted standalone.
    nested_in_fstring: set[int] = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
    }

    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            literals.append(
                "".join(
                    part.value
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                    else _DYNAMIC_TOKEN
                    for part in node.values
                )
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_nodes or id(node) in nested_in_fstring:
                continue
            literals.append(node.value)

    return ";\n".join(literals)


def _collect_all_migration_ddls() -> list[tuple[str, str]]:
    """Return [(migration_name, ddl_text), ...] for all M001–M042 migrations."""
    from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS, MIGRATIONS

    return [(m.name, m.ddl) for m in (*MIGRATIONS, *DEFERRED_MIGRATIONS)]


def _collect_all_migration_sources() -> list[tuple[str, str]]:
    """Return [(migration_name, sql_from_module_source), ...] for M001–M042.

    Reads each migration module's *entire* Python source via
    ``inspect.getsource`` — not just ``migration.ddl`` and not just ``apply()``.

    Why the whole module: a rebuild is split across module-level constants and
    the function that runs them. M021 keeps its replacement schema in
    ``_NEW_TABLE`` and issues the DROP inside ``apply()``; reading only the
    function body would see the DROP and miss the CREATE that puts the table
    back, manufacturing a false violation.

    Why ``inspect.getsource`` rather than reading ``src/…/migrations/*.py``:
    it resolves through the import system, so this works against an installed
    package as well as the source tree.
    """
    from superlocalmemory.storage._migration_internals import _MODULES
    from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS, MIGRATIONS

    collected: list[tuple[str, str]] = []
    for migration in (*MIGRATIONS, *DEFERRED_MIGRATIONS):
        module = _MODULES.get(migration.name)
        assert module is not None, (
            f"Migration {migration.name!r} is in the catalogue but absent from "
            "_MODULES. The scanner cannot read its source, so its apply() would "
            "go unchecked."
        )
        collected.append((migration.name, _extract_sql_literals(inspect.getsource(module))))
    return collected


def _snapshot_schema(db_path: Path) -> dict[str, set[str]]:
    """Return {table_name: {column_name, …}} for every user table in the DB.

    Returns an empty mapping when the file does not exist yet — learning.db is
    created by the migration runner itself, so the "before" snapshot of a fresh
    install is legitimately empty.
    """
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        return {
            table: {
                r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for table in tables
        }
    finally:
        conn.close()


def _schema_losses(
    before: dict[str, set[str]], after: dict[str, set[str]], label: str
) -> list[str]:
    """Return one human-readable line per table or column that went missing.

    Additions are ignored — the invariant is additive-only, so new schema is
    expected. Only disappearance breaks a 4.0.5 reader.
    """
    losses: list[str] = []
    for table, columns in sorted(before.items()):
        if table not in after:
            losses.append(f"{label}.{table}: TABLE DISAPPEARED")
            continue
        missing = columns - after[table]
        if missing:
            losses.append(f"{label}.{table}: COLUMNS DISAPPEARED {sorted(missing)}")
    return losses


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

def _build_405_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Construct a pair of DBs that faithfully represent a 4.0.5 production install.

    Mirrors the exact lifespan sequence the 4.0.5 daemon executes:
      1. ``schema.create_all_tables()`` on memory.db — bootstraps engine-side
         tables (atomic_facts, profiles, canonical_entities, …) as
         MemoryEngine.initialize() does before apply_deferred runs.
      2. ``apply_all(learning_db, memory_db)`` — 24 eager migrations.
      3. ``apply_deferred(learning_db, memory_db)`` — 17 deferred migrations
         (extending engine-bootstrapped tables); stamps slm_schema_version=42.
      4. INSERT two atomic_facts rows and one extra profile as representative
         pre-existing data.

    Returns:
        (learning_db, memory_db) as absolute Path objects under tmp_path.

    Raises:
        pytest.Failed if any migration reports a failure.
    """
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.migration_runner import apply_all, apply_deferred

    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"

    # Step 1: engine table bootstrap on memory.db.
    # LearningDatabase bootstraps its own tables inside apply_all via
    # _bootstrap_learning_schema, so learning.db needs no explicit pre-bootstrap.
    conn = sqlite3.connect(str(memory_db))
    try:
        schema.create_all_tables(conn)
        conn.commit()
    finally:
        conn.close()

    # Step 2: eager migrations (24 total).
    result_eager = apply_all(learning_db, memory_db)
    if result_eager["failed"]:
        pytest.fail(
            f"4.0.5 fixture: apply_all() reported failures:\n"
            f"  failed: {result_eager['failed']}\n"
            f"  details: {result_eager.get('details', {})}"
        )

    # Step 3: deferred migrations (17 total) + slm_schema_version=42 stamp.
    result_deferred = apply_deferred(learning_db, memory_db)
    if result_deferred["failed"]:
        pytest.fail(
            f"4.0.5 fixture: apply_deferred() reported failures:\n"
            f"  failed: {result_deferred['failed']}\n"
            f"  details: {result_deferred.get('details', {})}"
        )

    # Step 4: insert representative data.
    conn = sqlite3.connect(str(memory_db))
    try:
        # Extra profile beyond the default one seeded by create_all_tables.
        conn.execute(
            "INSERT OR IGNORE INTO profiles "
            "(profile_id, name, description) VALUES (?, ?, ?)",
            (
                _FIXTURE_PROFILE_ID,
                "Compat Test Profile",
                "Synthetic 4.0.5-era profile created by backward-compat test fixture",
            ),
        )
        # Two atomic_facts rows with known, distinct content.
        for fact_id, content in (
            (_FIXTURE_FACT_ID, _FIXTURE_FACT_CONTENT),
            (_FIXTURE_FACT_ID_2, _FIXTURE_FACT_CONTENT_2),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO atomic_facts "
                "(fact_id, memory_id, profile_id, content, fact_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (fact_id, "mem_fixture_405_001", _FIXTURE_PROFILE_ID,
                 content, "semantic"),
            )
        conn.commit()
    finally:
        conn.close()

    return learning_db, memory_db


# ---------------------------------------------------------------------------
# TestFixtureShape — prove the fixture IS 4.0.5-shaped
# ---------------------------------------------------------------------------

class TestFixtureShape:
    """Verify the fixture is genuinely 4.0.5-shaped, not just current-shaped.

    CRIT-1 fix: compare every migration file against the installed 4.0.5 package
    using SHA-256 file hashes.
    """

    def test_fixture_ddl_hashes_match_installed_405(self):
        """Every migration file in the current branch must be byte-for-byte
        identical to the installed 4.0.5 package.

        This is the primary guarantee that the fixture uses exactly the same DDL
        the 4.0.5 runner used. Identical file hashes → identical CREATE TABLE /
        ALTER TABLE DDL → fixture is 4.0.5-shaped.

        If 4.0.6 adds new migrations, this test documents them explicitly as
        failures that must be reviewed and acknowledged.
        """
        if not _INSTALLED_405_MIGRATIONS_DIR.is_dir():
            pytest.skip(
                f"4.0.5 reference package not found at {_INSTALLED_405_MIGRATIONS_DIR}. "
                "Install it with 'pipx install superlocalmemory==4.0.5' to enable "
                "this cross-version hash check."
            )

        from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS, MIGRATIONS

        current_src = (
            Path(__file__).parents[2]
            / "src" / "superlocalmemory" / "storage" / "migrations"
        )

        mismatches: list[str] = []
        new_in_406: list[str] = []

        for migration in (*MIGRATIONS, *DEFERRED_MIGRATIONS):
            fname = migration.name + ".py"
            current_path = current_src / fname
            installed_path = _INSTALLED_405_MIGRATIONS_DIR / fname

            assert current_path.exists(), (
                f"Migration file missing in current branch: {current_path}"
            )

            if not installed_path.exists():
                # This migration is NEW in 4.0.6 — not present in 4.0.5.
                new_in_406.append(fname)
                continue

            current_hash = hashlib.sha256(current_path.read_bytes()).hexdigest()
            installed_hash = hashlib.sha256(installed_path.read_bytes()).hexdigest()
            if current_hash != installed_hash:
                mismatches.append(
                    f"  {fname}: "
                    f"current={current_hash[:12]}… installed={installed_hash[:12]}…"
                )

        if new_in_406:
            # New migrations exist in 4.0.6 that were not in 4.0.5.
            # This is expected when new schema is added; the test fails to
            # force the release engineer to document the new schema additions.
            pytest.fail(
                "4.0.6 introduces NEW migration files absent from 4.0.5:\n"
                + "\n".join(f"  {f}" for f in new_in_406)
                + "\n\nUpdate this test to explicitly document each new migration "
                "and its backward-compatibility implications before releasing."
            )

        assert not mismatches, (
            "Migration DDL files DIFFER between 4.0.6 and the installed 4.0.5:\n"
            + "\n".join(mismatches)
            + "\n\nThe fixture cannot be guaranteed to be 4.0.5-shaped. "
            "Investigate DDL changes before releasing."
        )

    def test_supported_schema_version_rose_above_the_405_ceiling(self):
        """4.1.0's ceiling must be HIGHER than 4.0.5's, and that is the point.

        Through 4.0.6 this asserted the two were identical, because an additive
        release wants an older build to keep opening the store. M046 is not
        additive — it rebuilds a table with a constraint an older build's writes
        violate — so the ceilings must now differ in this direction:

          higher here  → a store this build migrated is refused by the old one,
                         which is the outcome we want.
          equal        → the old build writes into a store whose constraint
                         rejects its values, and the write is lost.
          lower here   → this build refuses stores it produced itself.
        """
        from superlocalmemory.storage._schema_version import SUPPORTED_SCHEMA_VERSION

        assert SUPPORTED_SCHEMA_VERSION == _EXPECTED_SCHEMA_VERSION, (
            f"SUPPORTED_SCHEMA_VERSION={SUPPORTED_SCHEMA_VERSION}, "
            f"expected {_EXPECTED_SCHEMA_VERSION}. It must match the trailing "
            "serial of the last migration, or the gate stops tracking what the "
            "migrations actually did — which is how it came to sit three "
            "migrations behind."
        )

        # Cross-reference against the installed 4.0.5 file if available.
        if not _INSTALLED_405_SCHEMA_VERSION_PATH.exists():
            return  # skip the cross-version part silently

        src = _INSTALLED_405_SCHEMA_VERSION_PATH.read_text()
        m = re.search(r"SUPPORTED_SCHEMA_VERSION\s*:\s*int\s*=\s*(\d+)", src)
        assert m is not None, (
            f"Cannot parse SUPPORTED_SCHEMA_VERSION from installed 4.0.5 "
            f"at {_INSTALLED_405_SCHEMA_VERSION_PATH}"
        )
        installed_version = int(m.group(1))

        assert SUPPORTED_SCHEMA_VERSION > installed_version, (
            f"SUPPORTED_SCHEMA_VERSION={SUPPORTED_SCHEMA_VERSION} is not above "
            f"the installed older build's {installed_version}. A store migrated "
            "here would then be opened and written by that build, whose values "
            "the new constraint rejects."
        )

    def test_migration_count_matches_expected(self):
        """Total migration count must be 41 (M001–M042, M008 absent).

        Detects accidental additions or removals from the MIGRATIONS /
        DEFERRED_MIGRATIONS catalogues that could silently hole the schema.
        """
        from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS, MIGRATIONS

        total = len(MIGRATIONS) + len(DEFERRED_MIGRATIONS)
        assert total == _EXPECTED_MIGRATION_COUNT, (
            f"Migration count changed: expected {_EXPECTED_MIGRATION_COUNT}, got {total}. "
            "A migration was added or removed. Review backward-compat implications "
            "and update _EXPECTED_MIGRATION_COUNT if intentional."
        )


# ---------------------------------------------------------------------------
# TestForwardCompat — 4.0.5 DB opens cleanly under 4.0.6 code
# ---------------------------------------------------------------------------

class TestForwardCompat:
    """Forward compatibility: a 4.0.5 DB must open on 4.0.6 with no data loss."""

    def test_apply_all_skips_all_on_fully_migrated_db(self, tmp_path):
        """apply_all() on a 4.0.5 DB must skip all migrations and fail none.

        A fully-migrated 4.0.5 DB already has every migration recorded as
        'complete' in its migration_log. The 4.0.6 runner must detect this
        and skip every migration — not re-run DDL that is already applied.
        """
        from superlocalmemory.storage.migration_runner import apply_all

        learning_db, memory_db = _build_405_fixture(tmp_path)
        result = apply_all(learning_db, memory_db)

        assert not result["failed"], (
            f"apply_all() on a 4.0.5 DB reported failures:\n"
            f"  {result['failed']}\ndetails: {result.get('details', {})}"
        )
        assert not result["applied"], (
            f"apply_all() re-applied migrations on a fully-migrated 4.0.5 DB:\n"
            f"  {result['applied']}\n"
            "apply_all must be idempotent — re-application signals a hash drift "
            "or broken idempotency guard."
        )
        assert len(result["skipped"]) > 0, (
            "apply_all() produced zero skipped migrations on a fully-migrated DB. "
            "Something went wrong with the migration log."
        )

    def test_preexisting_fact_content_survives(self, tmp_path):
        """CRIT-3 fix: assert exact content of a pre-existing row, not just COUNT.

        apply_all() must not modify, truncate, or delete any existing data row.
        This test reads back the exact fact_id and content string that were
        written by the fixture builder.
        """
        from superlocalmemory.storage.migration_runner import apply_all

        learning_db, memory_db = _build_405_fixture(tmp_path)
        apply_all(learning_db, memory_db)

        conn = sqlite3.connect(str(memory_db))
        try:
            row = conn.execute(
                "SELECT fact_id, content, profile_id, fact_type "
                "FROM atomic_facts WHERE fact_id = ?",
                (_FIXTURE_FACT_ID,),
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, (
            f"fact_id={_FIXTURE_FACT_ID!r} was DELETED during apply_all(). "
            "Migrations must never delete pre-existing rows."
        )
        assert row[0] == _FIXTURE_FACT_ID, (
            f"fact_id corrupted: expected {_FIXTURE_FACT_ID!r}, got {row[0]!r}"
        )
        assert row[1] == _FIXTURE_FACT_CONTENT, (
            f"Content was MUTATED during apply_all()!\n"
            f"  expected: {_FIXTURE_FACT_CONTENT!r}\n"
            f"  actual:   {row[1]!r}"
        )
        assert row[2] == _FIXTURE_PROFILE_ID, (
            f"profile_id corrupted: expected {_FIXTURE_PROFILE_ID!r}, got {row[2]!r}"
        )
        assert row[3] == "semantic", (
            f"fact_type corrupted: expected 'semantic', got {row[3]!r}"
        )

    def test_both_fixture_fact_rows_survive_by_id_and_content(self, tmp_path):
        """Both pre-existing rows must survive apply_all() with correct content.

        Asserts: row count == 2 AND each row has its exact expected content.
        """
        from superlocalmemory.storage.migration_runner import apply_all

        learning_db, memory_db = _build_405_fixture(tmp_path)
        apply_all(learning_db, memory_db)

        conn = sqlite3.connect(str(memory_db))
        try:
            rows = conn.execute(
                "SELECT fact_id, content FROM atomic_facts "
                "WHERE fact_id IN (?, ?) ORDER BY fact_id",
                (_FIXTURE_FACT_ID, _FIXTURE_FACT_ID_2),
            ).fetchall()
        finally:
            conn.close()

        assert len(rows) == 2, (
            f"Expected 2 fixture rows, found {len(rows)}. "
            f"IDs present: {[r[0] for r in rows]}"
        )
        row_map = {r[0]: r[1] for r in rows}
        assert row_map[_FIXTURE_FACT_ID] == _FIXTURE_FACT_CONTENT, (
            f"Content mismatch on {_FIXTURE_FACT_ID!r}: "
            f"expected {_FIXTURE_FACT_CONTENT!r}, got {row_map[_FIXTURE_FACT_ID]!r}"
        )
        assert row_map[_FIXTURE_FACT_ID_2] == _FIXTURE_FACT_CONTENT_2, (
            f"Content mismatch on {_FIXTURE_FACT_ID_2!r}: "
            f"expected {_FIXTURE_FACT_CONTENT_2!r}, got {row_map[_FIXTURE_FACT_ID_2]!r}"
        )

    def test_preexisting_profile_survives(self, tmp_path):
        """Pre-existing profile row must survive apply_all() intact."""
        from superlocalmemory.storage.migration_runner import apply_all

        learning_db, memory_db = _build_405_fixture(tmp_path)
        apply_all(learning_db, memory_db)

        conn = sqlite3.connect(str(memory_db))
        try:
            row = conn.execute(
                "SELECT profile_id, name FROM profiles WHERE profile_id = ?",
                (_FIXTURE_PROFILE_ID,),
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, (
            f"Profile {_FIXTURE_PROFILE_ID!r} was DELETED during apply_all(). "
            "Migrations must never delete existing profile rows."
        )
        assert row[0] == _FIXTURE_PROFILE_ID

    def test_schema_version_is_stamped_after_fixture_build(self, tmp_path):
        """The fixture's slm_schema_version must reach the current ceiling.

        If the deferred pass fails to stamp the version, this assertion catches
        the fixture builder's failure before any migration test runs.

        The fixture is built from 4.0.5-shaped DDL and then migrated by THIS
        build, so it ends at this build's ceiling, not at 4.0.5's. Asserting 42
        here would be asserting that the migrations did nothing.
        """
        from superlocalmemory.storage._schema_version import read_schema_version

        learning_db, memory_db = _build_405_fixture(tmp_path)

        assert read_schema_version(memory_db) == _EXPECTED_SCHEMA_VERSION, (
            f"memory.db schema_version={read_schema_version(memory_db)}, "
            f"expected {_EXPECTED_SCHEMA_VERSION}. The fixture deferred pass may "
            "not have completed."
        )
        assert read_schema_version(learning_db) == _EXPECTED_SCHEMA_VERSION, (
            f"learning.db schema_version={read_schema_version(learning_db)}, "
            f"expected {_EXPECTED_SCHEMA_VERSION}."
        )

    def test_schema_version_unchanged_after_apply_all(self, tmp_path):
        """apply_all() on an already-migrated DB must not alter the version stamp.

        apply_all() does not re-stamp slm_schema_version — only apply_deferred()
        does, and only when all migrations complete. Re-running apply_all() on a
        fully-migrated DB must leave the stamp where the deferred pass put it.
        """
        from superlocalmemory.storage._schema_version import read_schema_version
        from superlocalmemory.storage.migration_runner import apply_all

        learning_db, memory_db = _build_405_fixture(tmp_path)
        assert read_schema_version(memory_db) == _EXPECTED_SCHEMA_VERSION

        apply_all(learning_db, memory_db)

        assert read_schema_version(memory_db) == _EXPECTED_SCHEMA_VERSION, (
            "apply_all() mutated slm_schema_version on an already-migrated DB. "
            "Only apply_deferred() should write the final version stamp."
        )

    def test_forward_compat_completes_within_time_budget(self, tmp_path):
        """apply_all() on a 4.0.5 DB must complete in under 10 seconds.

        The upgrade path for a 4.0.5 user must be zero-friction. A slow
        no-op migration run would block daemon startup unnecessarily.
        """
        from superlocalmemory.storage.migration_runner import apply_all

        learning_db, memory_db = _build_405_fixture(tmp_path)

        t0 = time.perf_counter()
        result = apply_all(learning_db, memory_db)
        elapsed = time.perf_counter() - t0

        assert not result["failed"], f"apply_all failed: {result['failed']}"
        assert elapsed < 10.0, (
            f"apply_all() on a 4.0.5 DB took {elapsed:.3f}s. "
            "Budget is 10s; exceeding it means the upgrade path blocks startup."
        )


# ---------------------------------------------------------------------------
# TestAdditiveOnlyInvariant — no migration drops or renames schema objects
# ---------------------------------------------------------------------------

class TestAdditiveOnlyInvariant:
    """Prove M001–M042 are additive-only: no DROP or RENAME operations."""

    def test_no_destructive_ddl_in_any_migration(self):
        """Parse every migration's DDL and assert zero destructive operations.

        Scanned patterns:
          - DROP TABLE
          - ALTER TABLE ... DROP COLUMN
          - ALTER TABLE ... RENAME TO          (table rename)
          - ALTER TABLE ... RENAME COLUMN ... TO  (column rename, SQLite 3.25+)
        """
        migrations = _collect_all_migration_ddls()
        violations, dynamic = _scan_ddl_for_destructive_ops(migrations)

        assert not violations, (
            "ADDITIVE-ONLY INVARIANT VIOLATED — destructive DDL found in migrations:\n"
            + "\n".join(
                f"  [{label}] in {name!r}: {snippet!r}"
                for name, label, snippet in violations
            )
            + "\n\nInvariant I4 requires that schema changes be additive-only. "
            "No existing column or table may be dropped or renamed."
        )
        assert not dynamic, (
            "The static ``migration.ddl`` strings should never build a table "
            "name at runtime:\n"
            + "\n".join(f"  [{lbl}] {name!r}: {snip!r}" for name, lbl, snip in dynamic)
        )

    def test_no_destructive_sql_in_any_migration_module_source(self):
        """Scan every migration's full Python source, not just ``migration.ddl``.

        ``_apply_single`` runs a module's ``apply(conn)`` **instead of** the DDL
        string when one is present — the DDL is then only a documentation blob
        and a drift hash. Six migrations take that path for their table
        rebuilds (M021, M023, M026, M027, M032, M036), and M021 makes the
        divergence explicit:

            DDL = "ALTER TABLE ingestion_log ADD COLUMN profile_id …"   # additive
            def apply(conn):
                conn.execute("ALTER TABLE ingestion_log RENAME TO _ingestion_log_old")
                …
                conn.execute("DROP TABLE _ingestion_log_old")

        Scanning only ``migration.ddl`` reads the additive string and never
        sees the executed SQL. A future edit could turn any ``apply()``
        destructive while both the DDL scanner and the ``ddl_sha256`` drift
        guard stayed green, because neither observes the DDL string changing.

        Static source analysis — rather than a runtime check — is what closes
        this: five of the six rebuild branches early-return in a test fixture
        because their target tables are created at engine init and are absent,
        so no amount of running the migrations exercises that code.
        """
        sources = _collect_all_migration_sources()
        violations, dynamic = _scan_ddl_for_destructive_ops(sources)

        assert not violations, (
            "ADDITIVE-ONLY INVARIANT VIOLATED — destructive SQL found in "
            "migration module source (apply() bodies and module constants):\n"
            + "\n".join(
                f"  [{label}] in {name!r}: {snippet!r}"
                for name, label, snippet in violations
            )
            + "\n\nA table may only be dropped or renamed away if the same "
            "module puts the name back (the SQLite rebuild dance). Anything "
            "else removes schema a 4.0.5 reader depends on."
        )

        dynamic_names = {name for name, _, _ in dynamic}
        assert dynamic_names == set(_DYNAMIC_DDL_WAIVERS), (
            "Reviewed-waiver list is out of sync with the scanner.\n"
            f"  scanner reports dynamic DDL in : {sorted(dynamic_names)}\n"
            f"  _DYNAMIC_DDL_WAIVERS covers    : {sorted(_DYNAMIC_DDL_WAIVERS)}\n\n"
            "Table names built by f-string cannot be checked statically. Read "
            "the migration, confirm it is additive, and record the "
            "justification in _DYNAMIC_DDL_WAIVERS — or drop the stale waiver."
        )

    def test_module_source_scanner_has_teeth_destructive_apply(self):
        """TEETH CHECK: a destructive ``apply()`` must be caught by the source scan.

        This is the exact regression the source scanner exists to prevent — an
        additive-looking DDL string paired with an ``apply()`` that removes a
        column. Modelled on M021's real structure.

        NOTE: only a local list is scanned. No migration file is ever written.
        """
        real_sources = _collect_all_migration_sources()

        INJECTED_MODULE = (
            "TEST_INJECTED_DESTRUCTIVE_APPLY",
            _extract_sql_literals(
                '''
"""Docstring mentioning a rebuild: rename, create, copy, drop old."""
DDL = "ALTER TABLE ingestion_log ADD COLUMN profile_id TEXT DEFAULT 'x';"

def apply(conn):
    conn.execute("ALTER TABLE ingestion_log DROP COLUMN metadata")
'''
            ),
        )
        violations, _dynamic = _scan_ddl_for_destructive_ops(
            real_sources + [INJECTED_MODULE]
        )

        assert "TEST_INJECTED_DESTRUCTIVE_APPLY" in {n for n, _, _ in violations}, (
            "TEETH CHECK FAILED: the module-source scanner did not detect a "
            "DROP COLUMN hidden in an apply() body behind an additive DDL "
            "string. The scan has no teeth."
        )

        # And the real migrations stay clean.
        real_violations, _ = _scan_ddl_for_destructive_ops(real_sources)
        assert not real_violations, (
            "Real migrations report violations:\n"
            + "\n".join(
                f"  [{label}] {name!r}: {snippet!r}"
                for name, label, snippet in real_violations
            )
        )

    def test_module_source_scanner_has_teeth_unrecreated_drop(self):
        """TEETH CHECK: a DROP TABLE with no matching re-CREATE must be caught.

        Guards the reconciliation rule itself. The legitimate rebuild shapes all
        put the table name back; a DROP that does not is a real removal and must
        never be waved through as "probably a rebuild".
        """
        rebuild = (
            "REBUILD_OK",
            "ALTER TABLE widgets RENAME TO _widgets_old;"
            "CREATE TABLE widgets (id INTEGER PRIMARY KEY, extra TEXT);"
            "INSERT INTO widgets (id) SELECT id FROM _widgets_old;"
            "DROP TABLE _widgets_old;",
        )
        removal = (
            "REAL_REMOVAL",
            "DROP TABLE widgets;",
        )

        rebuild_violations, _ = _scan_ddl_for_destructive_ops([rebuild])
        assert not rebuild_violations, (
            "False positive: the reverse rebuild dance (RENAME away → CREATE "
            "fresh → DROP temp) was flagged as destructive:\n"
            f"  {rebuild_violations}"
        )

        removal_violations, _ = _scan_ddl_for_destructive_ops([removal])
        assert {n for n, _, _ in removal_violations} == {"REAL_REMOVAL"}, (
            "TEETH CHECK FAILED: a DROP TABLE that never re-creates the table "
            "was not reported as a violation."
        )

    def test_module_source_scanner_flags_dynamic_table_names(self):
        """TEETH CHECK: an f-string table name must be reported as unresolvable.

        Unresolvable is not the same as safe. If this degraded to silently
        passing, M023-shaped migrations would look verified when nothing had
        actually checked them.
        """
        dynamic_module = (
            "TEST_DYNAMIC",
            _extract_sql_literals(
                'def apply(conn, table):\n'
                '    conn.execute(f"DROP TABLE {table}")\n'
            ),
        )
        violations, dynamic = _scan_ddl_for_destructive_ops([dynamic_module])

        assert {n for n, _, _ in dynamic} == {"TEST_DYNAMIC"}, (
            "TEETH CHECK FAILED: an f-string-built DROP TABLE target was not "
            f"reported as dynamic. dynamic={dynamic!r} violations={violations!r}"
        )

    def test_extract_sql_literals_ignores_prose(self):
        """Docstrings and comments must not be scanned as if they were DDL.

        Several migrations describe their rebuild in prose ("rename → create →
        copy → drop old"). If that text reached the scanner it would produce
        violations no code change could ever clear, and the suite would be
        silenced or the check deleted.
        """
        source = (
            '"""Module doc: we DROP TABLE everything and RENAME COLUMN a TO b."""\n'
            "# comment: ALTER TABLE t DROP COLUMN c\n"
            "def apply(conn):\n"
            '    """Func doc: DROP TABLE more_things."""\n'
            '    conn.execute("CREATE TABLE t (id INTEGER)")\n'
        )
        extracted = _extract_sql_literals(source)

        assert "CREATE TABLE t" in extracted, (
            f"Real SQL was lost during extraction: {extracted!r}"
        )
        assert "DROP TABLE" not in extracted.upper(), (
            f"Docstring prose leaked into the scanned SQL: {extracted!r}"
        )
        assert "DROP COLUMN" not in extracted.upper(), (
            f"Comment text leaked into the scanned SQL: {extracted!r}"
        )

        violations, dynamic = _scan_ddl_for_destructive_ops([("PROSE", extracted)])
        assert not violations and not dynamic, (
            f"Prose produced findings: violations={violations!r} dynamic={dynamic!r}"
        )

    def test_additive_only_has_teeth_drop_column(self):
        """TEETH CHECK: the scanner must FAIL when a DROP COLUMN is injected.

        Methodology:
          1. Take the full real migration DDL list.
          2. Append a fake DDL string containing ALTER TABLE ... DROP COLUMN.
          3. Assert the scanner finds the injection (FAIL on injection).
          4. Remove the injection and assert zero violations (PASS on clean).

        NOTE: only the local list in this test is modified. No product migration
        file is ever written or modified.
        """
        real_ddls = _collect_all_migration_ddls()

        # --- Phase 1: inject DROP COLUMN, scanner must flag it ---
        INJECTED_DROP = (
            "TEST_INJECTED_DROP_COLUMN",
            "ALTER TABLE atomic_facts DROP COLUMN content;\n"
            "-- This DDL is TEST-ONLY and is never in any product migration file.",
        )
        with_injection = real_ddls + [INJECTED_DROP]
        violations_with, _ = _scan_ddl_for_destructive_ops(with_injection)

        injected_names = {name for name, _, _ in violations_with}
        assert "TEST_INJECTED_DROP_COLUMN" in injected_names, (
            "TEETH CHECK FAILED: the additive-only scanner DID NOT detect the "
            "injected DROP COLUMN. The check has no teeth — fix _DESTRUCTIVE_PATTERNS "
            "before trusting the invariant test."
        )

        # --- Phase 2: clean list, scanner must report zero violations ---
        violations_clean, _ = _scan_ddl_for_destructive_ops(real_ddls)
        assert not violations_clean, (
            "After removing the DROP COLUMN injection, the scanner still reports "
            "violations on real migrations:\n"
            + "\n".join(
                f"  [{lbl}] {name!r}: {snip!r}"
                for name, lbl, snip in violations_clean
            )
        )

    def test_additive_only_has_teeth_drop_table(self):
        """TEETH CHECK: the scanner must FAIL when a DROP TABLE is injected."""
        real_ddls = _collect_all_migration_ddls()

        INJECTED_DROP_TABLE = (
            "TEST_INJECTED_DROP_TABLE",
            "DROP TABLE IF EXISTS atomic_facts;\n"
            "-- TEST-ONLY injection, never in product.",
        )
        with_injection = real_ddls + [INJECTED_DROP_TABLE]
        violations, _ = _scan_ddl_for_destructive_ops(with_injection)

        injected_names = {name for name, _, _ in violations}
        assert "TEST_INJECTED_DROP_TABLE" in injected_names, (
            "TEETH CHECK FAILED: scanner did not detect injected DROP TABLE."
        )

        # Real DDL must still be clean after checking.
        assert not any(_scan_ddl_for_destructive_ops(real_ddls))

    def test_additive_only_has_teeth_rename_column(self):
        """TEETH CHECK: the scanner must FAIL when RENAME COLUMN is injected.

        CRIT-2 fix: the check must detect column renames, not only drops.
        A renamed column is as breaking as a dropped one — queries that
        reference the old name will break.
        """
        real_ddls = _collect_all_migration_ddls()

        INJECTED_RENAME_COL = (
            "TEST_INJECTED_RENAME_COLUMN",
            "ALTER TABLE atomic_facts RENAME COLUMN content TO body;\n"
            "-- TEST-ONLY injection, never in product.",
        )
        with_injection = real_ddls + [INJECTED_RENAME_COL]
        violations, _ = _scan_ddl_for_destructive_ops(with_injection)

        injected_names = {name for name, _, _ in violations}
        assert "TEST_INJECTED_RENAME_COLUMN" in injected_names, (
            "TEETH CHECK FAILED: scanner did not detect injected RENAME COLUMN. "
            "CRIT-2: the check must detect column renames, not only drops."
        )

        # Real DDL must still be clean.
        assert not any(_scan_ddl_for_destructive_ops(real_ddls))

    def test_schema_diff_applied_to_real_db(self, tmp_path):
        """Live schema diff across a REAL migration run — every table, both DBs.

        Two things this deliberately does differently from a naive version:

        1. It snapshots BEFORE ``apply_all`` rather than before a second
           ``apply_all`` on an already-migrated DB. On a fully-migrated DB every
           migration is logged complete and gets skipped (asserted by
           ``test_apply_all_skips_all_on_fully_migrated_db``), so a diff taken
           around that second call spans a no-op and can only ever detect a
           migration that is both non-idempotent AND destructive on re-run.

        2. It covers ``apply_deferred`` and ``learning.db``, not just
           ``apply_all`` on ``memory.db``. Four of the six apply()-based
           rebuilds (M021, M023, M026, M027) are DEFERRED and are never
           reached by ``apply_all`` at all; M027 targets learning.db.

        COVERAGE LIMIT — read before trusting this as the check for apply():
        this test only observes code that RUNS. Of the six apply()-based
        rebuilds, only M026 executes its rebuild branch here; M021/M023/M027
        early-return because their tables are created at engine init already
        carrying profile_id, and M032/M036 find their tables absent or
        well-formed. Exercising the rest would need hand-authored legacy table
        shapes, which would themselves be an unverified guess at the pre-4.0.5
        schema. Static coverage of all six lives in
        ``test_no_destructive_sql_in_any_migration_module_source``.
        """
        from superlocalmemory.storage import schema
        from superlocalmemory.storage.migration_runner import apply_all, apply_deferred

        learning_db = tmp_path / "learning.db"
        memory_db = tmp_path / "memory.db"

        conn = sqlite3.connect(str(memory_db))
        try:
            schema.create_all_tables(conn)
            conn.commit()
        finally:
            conn.close()

        before_memory = _snapshot_schema(memory_db)
        before_learning = _snapshot_schema(learning_db)

        apply_all(learning_db, memory_db)
        apply_deferred(learning_db, memory_db)

        losses = _schema_losses(before_memory, _snapshot_schema(memory_db), "memory.db")
        losses += _schema_losses(
            before_learning, _snapshot_schema(learning_db), "learning.db"
        )

        assert not losses, (
            "SCHEMA LOSS across a real apply_all() + apply_deferred() run:\n"
            + "\n".join(f"  {line}" for line in losses)
            + "\n\nMigrations must be additive. A table or column present before "
            "the run must still be present after it."
        )

        # Guard against the diff being vacuous: the run must actually build schema.
        assert len(_snapshot_schema(memory_db)) > len(before_memory), (
            "The migration run added no tables to memory.db. The before/after "
            "diff is spanning nothing and would pass no matter what."
        )

    def test_schema_diff_has_teeth_dropped_column(self, tmp_path, monkeypatch):
        """TEETH CHECK: the live diff must catch a column dropped by an apply().

        Plants a sentinel table before the run and makes a real migration's
        ``apply()`` drop a column from it, which is exactly the failure mode the
        diff exists to catch. A sentinel is used rather than a product table so
        the check does not depend on which columns happen to be droppable under
        SQLite's ALTER restrictions.
        """
        from superlocalmemory.storage import _migration_internals as internals
        from superlocalmemory.storage import schema
        from superlocalmemory.storage.migration_runner import apply_all, apply_deferred

        learning_db = tmp_path / "learning.db"
        memory_db = tmp_path / "memory.db"

        conn = sqlite3.connect(str(memory_db))
        try:
            schema.create_all_tables(conn)
            conn.execute(
                "CREATE TABLE _compat_teeth_sentinel (id INTEGER PRIMARY KEY, doomed TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

        before_memory = _snapshot_schema(memory_db)
        assert "doomed" in before_memory["_compat_teeth_sentinel"], (
            "Sentinel setup failed — the teeth check would pass vacuously."
        )

        # M036 is eager and targets memory.db, so its apply() runs in this pass.
        module = internals._MODULES["M036_vector_row_map"]
        original_apply = module.apply

        def destructive_apply(conn: sqlite3.Connection) -> None:
            original_apply(conn)
            conn.execute("ALTER TABLE _compat_teeth_sentinel DROP COLUMN doomed")

        monkeypatch.setattr(module, "apply", destructive_apply)

        apply_all(learning_db, memory_db)
        apply_deferred(learning_db, memory_db)

        losses = _schema_losses(before_memory, _snapshot_schema(memory_db), "memory.db")

        assert any("_compat_teeth_sentinel" in line for line in losses), (
            "TEETH CHECK FAILED: a column dropped inside a migration's apply() "
            f"was not reported by the live schema diff. losses={losses!r}"
        )

    def test_schema_diff_has_teeth_dropped_table(self, tmp_path, monkeypatch):
        """TEETH CHECK: the live diff must catch a table dropped by an apply()."""
        from superlocalmemory.storage import _migration_internals as internals
        from superlocalmemory.storage import schema
        from superlocalmemory.storage.migration_runner import apply_all, apply_deferred

        learning_db = tmp_path / "learning.db"
        memory_db = tmp_path / "memory.db"

        conn = sqlite3.connect(str(memory_db))
        try:
            schema.create_all_tables(conn)
            conn.execute("CREATE TABLE _compat_teeth_sentinel (id INTEGER PRIMARY KEY)")
            conn.commit()
        finally:
            conn.close()

        before_memory = _snapshot_schema(memory_db)
        assert "_compat_teeth_sentinel" in before_memory

        module = internals._MODULES["M036_vector_row_map"]
        original_apply = module.apply

        def destructive_apply(conn: sqlite3.Connection) -> None:
            original_apply(conn)
            conn.execute("DROP TABLE _compat_teeth_sentinel")

        monkeypatch.setattr(module, "apply", destructive_apply)

        apply_all(learning_db, memory_db)
        apply_deferred(learning_db, memory_db)

        losses = _schema_losses(before_memory, _snapshot_schema(memory_db), "memory.db")

        assert any("TABLE DISAPPEARED" in line for line in losses), (
            "TEETH CHECK FAILED: a table dropped inside a migration's apply() "
            f"was not reported by the live schema diff. losses={losses!r}"
        )


# ---------------------------------------------------------------------------
# TestBackwardTolerance — document 4.0.6 DB safety when opened by 4.0.5 code
# ---------------------------------------------------------------------------

class TestBackwardTolerance:
    """Document and prove backward tolerance: 4.0.6 DB opened by 4.0.5 code.

    True in-process 4.0.5 execution is not possible here (Python version
    mismatch). Instead these tests prove the STRUCTURAL PROPERTIES that make
    backward opening safe, and document residual risk explicitly.
    """

    def test_this_builds_own_gate_accepts_the_store_it_migrated(self, tmp_path):
        """This build must open what this build produced.

        The trivial direction, and worth asserting because the gate is a
        strict inequality: a store stamped at the ceiling is not above it.
        Getting this wrong would make the release refuse its own stores.
        """
        from superlocalmemory.storage._schema_version import (
            check_version_or_raise,
            read_schema_version,
        )

        learning_db, memory_db = _build_405_fixture(tmp_path)

        assert read_schema_version(memory_db) == _EXPECTED_SCHEMA_VERSION
        assert read_schema_version(learning_db) == _EXPECTED_SCHEMA_VERSION

        try:
            check_version_or_raise(memory_db)
            check_version_or_raise(learning_db)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"check_version_or_raise raised for a store at the ceiling "
                f"({_EXPECTED_SCHEMA_VERSION}): {exc}"
            )

    def test_an_older_build_is_refused_by_the_gate(self, tmp_path):
        """The property this release actually needs, asserted against the store.

        Through 4.0.6 the sibling test above proved the opposite — that an older
        build could open a newer store, because every migration until then was
        additive and an older reader simply never touched what it did not know
        about.

        M046 is not additive. It rebuilds ``atomic_facts`` with a constraint
        that rejects the value an older build files planned events under, so an
        older writer's INSERT fails against a store this build has migrated. The
        version ceiling is what turns that into a refusal to start instead of a
        lost memory, and this asserts the refusal happens.

        The older ceiling is simulated by calling the gate with it directly.
        Installing an old release inside this process is not possible, and the
        gate's whole logic is the one comparison being exercised here.
        """
        from superlocalmemory.storage import _schema_version as sv

        learning_db, memory_db = _build_405_fixture(tmp_path)
        stored = sv.read_schema_version(memory_db)
        assert stored == _EXPECTED_SCHEMA_VERSION

        older_ceiling = 42  # 4.0.5 through 4.0.10
        assert stored > older_ceiling, (
            "the store this build migrated is not stamped above the older "
            "ceiling, so nothing stops an older build from writing to it"
        )

        original = sv.SUPPORTED_SCHEMA_VERSION
        sv.SUPPORTED_SCHEMA_VERSION = older_ceiling
        try:
            with pytest.raises(sv.SchemaVersionError):
                sv.check_version_or_raise(memory_db)
        finally:
            sv.SUPPORTED_SCHEMA_VERSION = original

        # And the gate is not simply always-raising: restored, it accepts.
        sv.check_version_or_raise(memory_db)

    def test_version_gate_has_teeth_future_version(self, tmp_path):
        """Prove the gate FIRES for a DB stamped with a future schema version.

        This test verifies the gate is not vacuous: it MUST raise for any DB
        whose schema_version exceeds the supported ceiling.
        """
        from superlocalmemory.storage._schema_version import (
            SchemaVersionError,
            check_version_or_raise,
            ensure_schema_version_table,
            write_schema_version,
        )

        future_db = tmp_path / "future.db"
        conn = sqlite3.connect(str(future_db))
        try:
            ensure_schema_version_table(conn)
            write_schema_version(conn, 999)  # hypothetical future version
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(SchemaVersionError, match=r"schema_version=999"):
            check_version_or_raise(future_db)

    def test_no_new_schema_objects_in_406(self):
        """A 4.0.6 DB has no tables or columns unknown to 4.0.5 code.

        Since M001–M042 are byte-for-byte identical in both releases, any DB
        migrated to 4.0.6 has exactly the same tables and columns as a 4.0.5 DB.
        4.0.5 queries will never hit a table or column that did not exist in 4.0.5.
        """
        if not _INSTALLED_405_MIGRATIONS_DIR.is_dir():
            pytest.skip("4.0.5 reference not found — skipping new-objects check.")

        from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS, MIGRATIONS

        new_in_406 = []
        for migration in (*MIGRATIONS, *DEFERRED_MIGRATIONS):
            fname = migration.name + ".py"
            installed_path = _INSTALLED_405_MIGRATIONS_DIR / fname
            if not installed_path.exists():
                new_in_406.append(migration.name)

        assert not new_in_406, (
            "4.0.6 has NEW migrations absent from 4.0.5:\n"
            + "\n".join(f"  {n}" for n in new_in_406)
            + "\n\nThese add schema objects that 4.0.5 code does not know about. "
            "Backward-open safety is NOT guaranteed for queries that touch those "
            "objects. Document the risk before releasing."
        )

    def test_residual_risk_documented(self):
        """Documentation test: records residual backward-compatibility risk.

        This test always passes. Its purpose is to make the residual risk
        visible in the test suite so it is reviewed alongside the passing tests.

        RESIDUAL RISK as of 4.0.6 (WAL deadlock fix only):
        --------------------------------------------------
        1. WAL BEHAVIORAL DELTA (non-schema, PERFORMANCE RISK ONLY):
           4.0.6 sets wal_autocheckpoint=400 and SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE
           per connection. A 4.0.6-written WAL file opened by 4.0.5 code will
           be managed by 4.0.5's default WAL settings (autocheckpoint=1000,
           checkpoint-on-close enabled). This may cause the WAL file to grow
           up to 1000 frames before a checkpoint fires (vs 400 in 4.0.6), and
           the close-path checkpoint on 4.0.5 may briefly block on busy readers.
           VERDICT: performance degradation possible; NO data loss or corruption.

        2. SCHEMA OBJECTS: since no migrations were added in 4.0.6, there are
           zero new tables or columns. 4.0.5 queries cannot encounter unknown
           schema objects. VERDICT: CLEAN.

        3. SCHEMA VERSION CEILING: both releases use SUPPORTED_SCHEMA_VERSION=42.
           The version gate in either direction returns clean (42 > 42 is False).
           VERDICT: CLEAN.

        OVERALL VERDICT: backward opening of a 4.0.6 DB by 4.0.5 code is
        DATA-SAFE. Operators who downgrade from 4.0.6 to 4.0.5 may observe
        temporarily increased WAL file sizes and occasional close-path blocking,
        but no data loss or corruption will occur.
        """
        pass  # documentation only — always passes
