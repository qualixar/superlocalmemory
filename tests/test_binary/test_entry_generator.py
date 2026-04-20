# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-06 §9.1 / H4-H7, H14

"""Tests for ``scripts/build_entry.py``.

Covers LLD-06 hard rules H4 (AST-generated), H5 (stdlib-only imports),
H6 (no write-mode open), H7 (no network), H14 (envelope shape on hit),
plus the ``source_sha256`` banner and parity-with-live-module test.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import sqlite3
import string
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import build_entry  # noqa: E402 — injected via tests/test_binary/conftest.py


REPO_ROOT = Path(__file__).resolve().parents[2]
TOPIC_SRC = REPO_ROOT / "src" / "superlocalmemory" / "core" / "topic_signature.py"
CACHE_SRC = REPO_ROOT / "src" / "superlocalmemory" / "core" / "context_cache.py"


# ---------------------------------------------------------------------------
# H4 — AST generator + determinism + SHA banner
# ---------------------------------------------------------------------------


def test_generator_emits_deterministic_output():
    """Two runs on identical input -> byte-identical emitted text."""
    r1 = build_entry.generate(TOPIC_SRC, CACHE_SRC)
    r2 = build_entry.generate(TOPIC_SRC, CACHE_SRC)
    assert r1.text == r2.text
    assert r1.source_sha256 == r2.source_sha256
    # Matches direct SHA of concatenated source bytes.
    expected = hashlib.sha256(
        TOPIC_SRC.read_bytes() + CACHE_SRC.read_bytes()
    ).hexdigest()
    assert r1.source_sha256 == expected


def test_generator_records_source_sha256():
    res = build_entry.generate(TOPIC_SRC, CACHE_SRC)
    assert f"# source_sha256: {res.source_sha256}" in res.text
    assert build_entry.BANNER_AUTOGEN in res.text


def test_emit_entry_writes_file(tmp_path):
    dest = tmp_path / "out" / "hook_entry.py"
    res = build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest)
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == res.text


def test_emit_entry_is_byte_identical_twice(tmp_path):
    dest_a = tmp_path / "a.py"
    dest_b = tmp_path / "b.py"
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest_a)
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest_b)
    assert dest_a.read_bytes() == dest_b.read_bytes()


# ---------------------------------------------------------------------------
# H5 — stdlib-only imports
# ---------------------------------------------------------------------------


def _imports_in(text: str) -> set[str]:
    tree = ast.parse(text)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def test_generator_emits_stdlib_only_imports():
    text = build_entry.generate(TOPIC_SRC, CACHE_SRC).text
    imports = _imports_in(text)
    # Every import must be in the allow-list.
    non_stdlib = imports - build_entry.ALLOWED_STDLIB_IMPORTS
    assert non_stdlib == set(), (
        f"emitted file imports non-stdlib modules: {non_stdlib}"
    )


def test_static_check_rejects_nonstdlib_import():
    bad = textwrap.dedent(
        """
        import json
        import torch
        def main() -> int:
            return 0
        """
    )
    with pytest.raises(build_entry.GeneratorError):
        build_entry.static_check(bad)


def test_static_check_rejects_superlocalmemory_import():
    bad = textwrap.dedent(
        """
        import json
        from superlocalmemory.core.engine import MemoryEngine
        """
    )
    with pytest.raises(build_entry.GeneratorError):
        build_entry.static_check(bad)


# ---------------------------------------------------------------------------
# H6 — no write-mode open()
# ---------------------------------------------------------------------------


def test_generator_rejects_write_modes():
    for mode in ("w", "wb", "a", "a+", "w+", "x", "x+"):
        bad = textwrap.dedent(
            f"""
            import os
            def f():
                with open('/tmp/x', '{mode}') as fh:
                    fh.write('nope')
            """
        )
        with pytest.raises(build_entry.GeneratorError):
            build_entry.static_check(bad)


def test_static_check_allows_read_mode():
    ok = textwrap.dedent(
        """
        import os
        def f():
            with open('/etc/hostname', 'r') as fh:
                return fh.read()
        """
    )
    # Should not raise.
    build_entry.static_check(ok)


def test_static_check_allows_default_mode():
    ok = textwrap.dedent(
        """
        import os
        def f():
            with open('/etc/hostname') as fh:
                return fh.read()
        """
    )
    build_entry.static_check(ok)


def test_generated_entry_has_no_write_mode_open():
    text = build_entry.generate(TOPIC_SRC, CACHE_SRC).text
    # Static check passes => H6 enforced.
    build_entry.static_check(text)


# ---------------------------------------------------------------------------
# H7 — no network imports
# ---------------------------------------------------------------------------


def test_static_check_rejects_network_imports():
    for mod in ("urllib", "http", "socket", "requests", "httpx"):
        bad = f"import {mod}\n"
        with pytest.raises(build_entry.GeneratorError):
            build_entry.static_check(bad)


def test_static_check_rejects_network_from_import():
    bad = "from urllib.request import urlopen\n"
    with pytest.raises(build_entry.GeneratorError):
        build_entry.static_check(bad)


def test_generated_entry_has_no_network_imports():
    text = build_entry.generate(TOPIC_SRC, CACHE_SRC).text
    imports = _imports_in(text)
    forbidden = {"urllib", "http", "socket", "requests",
                 "httpx", "ftplib", "smtplib"}
    assert imports & forbidden == set()


# ---------------------------------------------------------------------------
# H4 parity — extracted compute_topic_signature matches live module
# ---------------------------------------------------------------------------


def _random_text(rng: random.Random, length: int) -> str:
    alphabet = (string.ascii_letters + string.digits +
                " /\"'._-:" + "\u00e9\u00f1\u4e2d")
    return "".join(rng.choice(alphabet) for _ in range(length))


def _load_emitted_module(tmp_path: Path):
    """Write emitted entry and import it as a fresh module."""
    dest = tmp_path / "emitted_entry.py"
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest)
    # Import it as a unique module by inserting its dir on sys.path.
    sys.path.insert(0, str(tmp_path))
    try:
        # Fresh import each call.
        if "emitted_entry" in sys.modules:
            del sys.modules["emitted_entry"]
        import emitted_entry  # type: ignore
        return emitted_entry
    finally:
        sys.path.pop(0)


def test_topic_signature_parity_100_random_inputs(tmp_path):
    """Extracted _compute_topic_signature must match the live module
    byte-for-byte on 100 random inputs (LLD-06 H4 spirit + §9.1)."""
    from superlocalmemory.core.topic_signature import (
        compute_topic_signature as live_sig,
    )
    emitted = _load_emitted_module(tmp_path)

    rng = random.Random(0xC0FFEE)
    # Include corner cases + randoms.
    cases: list[str] = [
        "",
        "hello world",
        "the quick brown fox",
        "class FooBar(BaseClass): pass",
        "https://qualixar.com/path?x=1",
        '"quoted string with spaces"',
        "/usr/bin/python3",
        "caf\u00e9 resum\u00e9",
    ]
    for _ in range(92):
        cases.append(_random_text(rng, rng.randint(0, 200)))

    for text in cases:
        expected = live_sig(text)
        got = emitted._compute_topic_signature(text)
        assert got == expected, (
            f"parity mismatch for {text!r}: live={expected} emitted={got}"
        )


# ---------------------------------------------------------------------------
# H14 — envelope on cache hit matches LLD-01 §4.3
# ---------------------------------------------------------------------------


def _write_install_token(home: Path) -> str:
    import hmac as _hmac
    token = "deadbeef" * 4  # 32 chars
    (home / ".install_token").write_text(token, encoding="utf-8")
    binding = _hmac.new(
        token.encode("utf-8"),
        b"active_brain_cache",
        hashlib.sha256,
    ).hexdigest()[:32]
    return binding


def _seed_cache(home: Path, session_id: str, sig: str, content: str):
    import time as _time
    db = home / "active_brain_cache.db"
    binding = _write_install_token(home)
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE slm_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE context_entries (
            session_id TEXT NOT NULL,
            topic_sig TEXT NOT NULL,
            content TEXT NOT NULL,
            fact_ids TEXT NOT NULL DEFAULT '[]',
            provenance TEXT NOT NULL DEFAULT 'tool_observation',
            computed_at INTEGER NOT NULL,
            byte_size INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (session_id, topic_sig)
        );
        """
    )
    now = int(_time.time())
    conn.execute(
        "INSERT INTO slm_meta VALUES (?, ?, ?)",
        ("install_token_hmac", binding, now),
    )
    conn.execute(
        "INSERT INTO context_entries "
        "(session_id, topic_sig, content, fact_ids, provenance, "
        " computed_at, byte_size) VALUES (?, ?, ?, '[]', "
        " 'tool_observation', ?, ?)",
        (session_id, sig, content, now, len(content)),
    )
    conn.commit()
    conn.close()
    return db


def _run_emitted(
    emitted_dir: Path,
    stdin_bytes: bytes,
    env_overrides: dict,
) -> tuple[str, int]:
    env = os.environ.copy()
    env.update(env_overrides)
    env["PYTHONPATH"] = str(emitted_dir) + os.pathsep + env.get(
        "PYTHONPATH", "",
    )
    proc = subprocess.run(
        [sys.executable, "-c",
         "import emitted_entry as e; import sys; sys.exit(e.main())"],
        input=stdin_bytes,
        env=env,
        capture_output=True,
        timeout=30,
    )
    return proc.stdout.decode("utf-8", errors="replace"), proc.returncode


def test_binary_entry_envelope_on_hit(tmp_path):
    """H14: on a cache hit, main() outputs the LLD-01 §4.3 envelope."""
    home = tmp_path / ".slm"
    home.mkdir()
    dest = tmp_path / "emitted_entry.py"
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest)

    from superlocalmemory.core.topic_signature import compute_topic_signature
    prompt = "build the login button in React"
    sig = compute_topic_signature(prompt)
    db = _seed_cache(home, "sess-123", sig, "cached context for tests")

    stdout, rc = _run_emitted(
        tmp_path,
        stdin_bytes=json.dumps(
            {"session_id": "sess-123", "prompt": prompt}
        ).encode("utf-8"),
        env_overrides={"SLM_CACHE_DB": str(db)},
    )
    assert rc == 0
    doc = json.loads(stdout)
    assert "hookSpecificOutput" in doc
    assert doc["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert (
        doc["hookSpecificOutput"]["additionalContext"]
        == "cached context for tests"
    )


def test_binary_entry_fail_open_on_empty_stdin(tmp_path):
    dest = tmp_path / "emitted_entry.py"
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest)
    stdout, rc = _run_emitted(tmp_path, stdin_bytes=b"",
                              env_overrides={})
    assert rc == 0
    assert stdout == "{}"


def test_binary_entry_fail_open_on_missing_db(tmp_path):
    dest = tmp_path / "emitted_entry.py"
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest)
    stdout, rc = _run_emitted(
        tmp_path,
        stdin_bytes=json.dumps(
            {"session_id": "s", "prompt": "hello"}
        ).encode("utf-8"),
        env_overrides={"SLM_CACHE_DB": str(tmp_path / "nope.db")},
    )
    assert rc == 0
    assert stdout == "{}"


def test_binary_entry_fail_open_on_corrupt_json(tmp_path):
    dest = tmp_path / "emitted_entry.py"
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest)
    stdout, rc = _run_emitted(
        tmp_path, stdin_bytes=b"{not-json",
        env_overrides={},
    )
    assert rc == 0
    assert stdout == "{}"


def test_binary_entry_fail_open_on_missing_session_id(tmp_path):
    dest = tmp_path / "emitted_entry.py"
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest)
    stdout, rc = _run_emitted(
        tmp_path,
        stdin_bytes=json.dumps({"prompt": "x"}).encode("utf-8"),
        env_overrides={},
    )
    assert rc == 0
    assert stdout == "{}"


def test_binary_entry_fail_open_on_miss(tmp_path):
    """Cache miss -> '{}', exit 0."""
    home = tmp_path / ".slm"
    home.mkdir()
    dest = tmp_path / "emitted_entry.py"
    build_entry.emit_entry(TOPIC_SRC, CACHE_SRC, dest)

    # Seed a DB but with a DIFFERENT topic signature so this prompt misses.
    db = _seed_cache(home, "sess-x", "deadbeefdeadbeef", "other")

    stdout, rc = _run_emitted(
        tmp_path,
        stdin_bytes=json.dumps(
            {"session_id": "sess-x", "prompt": "unrelated prompt"}
        ).encode("utf-8"),
        env_overrides={"SLM_CACHE_DB": str(db)},
    )
    assert rc == 0
    assert stdout == "{}"


# ---------------------------------------------------------------------------
# Spec file rule guards (H1, H2, H3)
# ---------------------------------------------------------------------------


def _spec_text() -> str:
    return (REPO_ROOT / "scripts" / "slm-hook.spec").read_text(
        encoding="utf-8"
    )


def test_spec_uses_onedir_not_onefile():
    text = _spec_text()
    import re as _re
    assert _re.search(r"onefile\s*=\s*True", text) is None
    # Must have COLLECT step (onedir hallmark).
    assert "COLLECT(" in text


def test_spec_disables_console_on_windows():
    text = _spec_text()
    import re as _re
    assert _re.search(r"console\s*=\s*False", text) is not None
    assert _re.search(r"console\s*=\s*True", text) is None


def test_spec_disables_upx():
    text = _spec_text()
    import re as _re
    assert _re.search(r"upx\s*=\s*False", text) is not None
    assert _re.search(r"upx\s*=\s*True", text) is None


# ---------------------------------------------------------------------------
# H15 — DDL must not appear in legacy_migration.py
# ---------------------------------------------------------------------------


def test_extract_function_raises_on_missing():
    """_extract_function raises LookupError for unknown function names."""
    with pytest.raises(LookupError):
        build_entry._extract_function(
            "def foo(): pass\n", "does_not_exist",
        )


def test_static_check_rejects_syntax_error():
    with pytest.raises(build_entry.GeneratorError,
                       match="syntax error"):
        build_entry.static_check("def broken(:\n")


def test_static_check_detects_keyword_mode_write():
    bad = textwrap.dedent(
        """
        def f():
            open('/tmp/x', mode='w')
        """
    )
    with pytest.raises(build_entry.GeneratorError):
        build_entry.static_check(bad)


def test_static_check_allows_keyword_mode_read():
    ok = textwrap.dedent(
        """
        def f():
            open('/tmp/x', mode='r')
        """
    )
    build_entry.static_check(ok)


def test_default_source_paths_resolves_from_repo_root(tmp_path):
    t, c = build_entry.default_source_paths(tmp_path)
    assert t.name == "topic_signature.py"
    assert c.name == "context_cache.py"
    assert "core" in t.parts
    assert "core" in c.parts


def test_legacy_migration_has_zero_ddl():
    """LLD-06 H15: no ALTER/CREATE/DROP TABLE or CREATE INDEX."""
    import re as _re
    target = (
        REPO_ROOT / "src" / "superlocalmemory"
        / "learning" / "legacy_migration.py"
    )
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    # Strip docstrings/comments that might mention these as prose: we
    # enforce the rule on code lines only. Cheapest safe approach: the
    # regex must not fire on words inside triple-quoted strings. Parse
    # the file and concatenate only non-docstring source by AST-walk.
    tree = ast.parse(text)
    code_lines: list[str] = []
    for line in text.splitlines():
        code_lines.append(line)

    # Compile-time regex (case-insensitive).
    pattern = _re.compile(
        r"(?i)(ALTER\s+TABLE|CREATE\s+TABLE|DROP\s+TABLE|CREATE\s+INDEX)"
    )
    # Skip docstring spans.
    doc_spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                doc_spans.append(
                    (body[0].lineno, body[0].end_lineno or body[0].lineno)
                )
    for idx, line in enumerate(code_lines, start=1):
        in_doc = any(s <= idx <= e for (s, e) in doc_spans)
        if in_doc:
            continue
        assert pattern.search(line) is None, (
            f"DDL found in legacy_migration.py:{idx}: {line}"
        )
