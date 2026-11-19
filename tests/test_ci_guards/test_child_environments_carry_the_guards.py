# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-02 (19 Nov 2026): a test child environment built by hand must carry the four guards.

The pytest plugin sets ``PYTHON_KEYRING_BACKEND`` and three more variables in ``os.environ``, which
only a child that *inherits* the environment gets. A builder that constructs its own environment
(``HOME`` plus a few names) loses them: a daemon started with it asked the macOS keychain with a
``HOME`` that has no login keychain, and the person at the computer saw a dialog.

This test is static (it parses the files, it starts nothing). It fails when a function in ``tests/``
builds an environment with its own ``HOME`` (a dict key or an item assignment) without taking the
guards from ``tests._portable`` (``child_env_base`` / ``GUARD_ENV``) or naming the keyring variable.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from tests import _portable

TESTS = Path(__file__).resolve().parents[1]

GUARDS = {
    "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
    "SLM_NO_AUTO_DOWNLOAD": "1",
    "SLM_NO_HOOK_UPGRADE": "1",
    "SLM_SKIP_LAYA": "1",
}
MARKERS = ("child_env_base", "GUARD_ENV", "PYTHON_KEYRING_BACKEND")

#: Functions that read like builders but never reach a process (a fake psutil object, a pure parser).
NOT_A_PROCESS = {
    "tests/test_mcp/test_default_root_presence.py",
    "tests/test_mcp/test_legacy_process_presence.py",
    "tests/test_isolation_guard.py::test_the_account_store_is_protected_even_if_home_was_redirected",
}


_INHERITS = re.compile(
    r"\*\*\s*(?:__import__\(.os.\)\.)?os\.environ|dict\(\s*os\.environ|os\.environ\.copy\(\)")


def _sets_home(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value == "HOME" for k in sub.keys if k):
            return True
        if isinstance(sub, ast.Assign):
            for target in sub.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "HOME"):
                    return True
        if isinstance(sub, ast.Call):
            for kw in sub.keywords:
                if kw.arg == "HOME":
                    return True
    return False


def _hand_built_without_guards() -> list[str]:
    failures: list[str] = []
    for path in sorted(TESTS.rglob("*.py")):
        rel = path.relative_to(TESTS.parent).as_posix()
        if rel == "tests/test_ci_guards/test_child_environments_carry_the_guards.py":
            continue
        if rel in NOT_A_PROCESS:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if f"{rel}::{fn.name}" in NOT_A_PROCESS or not _sets_home(fn):
                continue
            text = ast.get_source_segment(source, fn) or ""
            if _INHERITS.search(text):
                continue  # copies os.environ, so it inherits the guards the plugin set
            if any(marker in text for marker in MARKERS):
                continue
            failures.append(f"{rel}:{fn.lineno} {fn.name}")
    return failures


def test_the_shared_base_carries_all_four_guards(tmp_path):
    env = _portable.child_env_base(tmp_path)
    for name, value in GUARDS.items():
        assert env[name] == value, name
    assert env["HOME"] == str(tmp_path)


def test_the_shared_guard_constant_is_exactly_the_four_variables():
    assert dict(_portable.GUARD_ENV) == GUARDS


def test_every_hand_built_child_environment_takes_the_guards():
    failures = _hand_built_without_guards()
    assert not failures, (
        "These functions build a child environment with its own HOME and do not carry the keyring "
        "and download guards (use tests._portable.child_env_base or GUARD_ENV):\n  "
        + "\n  ".join(failures))


def test_the_scan_catches_a_builder_that_forgets_them(tmp_path, monkeypatch):
    """The scan itself can fail: a builder with HOME and no guard is reported."""
    (tmp_path / "test_bad.py").write_text(
        "def build(home):\n    return {'HOME': str(home), 'PATH': '/usr/bin'}\n", encoding="utf-8")
    (tmp_path / "test_good.py").write_text(
        "from tests._portable import child_env_base\n"
        "def build(home):\n    return {**child_env_base(home), 'PATH': '/usr/bin'}\n",
        encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "TESTS", tmp_path)
    found = _hand_built_without_guards()
    assert any("test_bad.py" in f for f in found)
    assert not any("test_good.py" in f for f in found)
