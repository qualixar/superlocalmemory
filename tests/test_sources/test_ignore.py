"""The gitignore subset and the fixed skip rules."""

from __future__ import annotations

import pytest

from superlocalmemory.sources.ignore import GitIgnore, IgnoreRules, DEFAULT_TYPES

CASES = [
    # (patterns, path, is_dir, ignored)
    ("*.log", "a.log", False, True),
    ("*.log", "deep/er/a.log", False, True),
    ("*.log", "a.txt", False, False),
    ("# comment\n*.log", "# comment", False, False),
    ("build/", "build", True, True),
    ("build/", "build", False, False),
    ("build/", "src/build", True, True),
    ("build/", "build/out.txt", False, True),
    ("/build", "build", True, True),
    ("/build", "src/build", True, False),
    ("doc/frotz", "doc/frotz", False, True),
    ("doc/frotz", "a/doc/frotz", False, False),
    ("*.log\n!keep.log", "keep.log", False, False),
    ("*.log\n!keep.log", "drop.log", False, True),
    ("build/\n!build/keep.txt", "build/keep.txt", False, True),  # parent excluded
    ("**/foo", "foo", False, True),
    ("**/foo", "a/b/foo", False, True),
    ("a/**", "a/b/c.txt", False, True),
    ("a/**/z", "a/z", False, True),
    ("a/**/z", "a/b/c/z", False, True),
    ("a/**/z", "b/z", False, False),
    ("f?o", "foo", False, True),
    ("f?o", "fo", False, False),
    ("[ab].txt", "a.txt", False, True),
    ("[ab].txt", "c.txt", False, False),
    ("[!ab].txt", "c.txt", False, True),
    ("\\#file", "#file", False, True),
    ("\\!file", "!file", False, True),
    ("*.tmp \n", "x.tmp", False, True),
    ("foo*bar", "foo/bar", False, False),
    ("", "anything", False, False),
]


@pytest.mark.parametrize("patterns,path,is_dir,expected", CASES)
def test_gitignore_cases(patterns, path, is_dir, expected):
    assert GitIgnore.parse(patterns).ignored(path, is_dir) is expected


def _rules(tmp_path, **files):
    for name, body in files.items():
        (tmp_path / ("." + name)).write_text(body)
    rules = IgnoreRules(tmp_path)
    rules.enter_dir("")
    return rules


def reason(rules, rel, *, is_dir=False, size=10):
    return rules.skip_reason(rel, is_dir=is_dir, size=size)


def test_fixed_rules_in_order(tmp_path):
    r = _rules(tmp_path)
    assert reason(r, ".git", is_dir=True) == "dotfile"
    assert reason(r, "notes/.env.local") == "dotfile"
    assert reason(r, "node_modules", is_dir=True) == "excluded_dir"
    assert reason(r, "a/venv", is_dir=True) == "excluded_dir"
    assert reason(r, "a/site-packages", is_dir=True) == "excluded_dir"
    for name in ("server.pem", "id_rsa", "vault.kdbx", "x.p12", "x.pfx", "x.key",
                 "app.db", "app.sqlite"):
        assert reason(r, name) == "secret_name", name
    assert reason(r, "note.md") is None


def test_type_allow_list_and_temp_names(tmp_path):
    r = _rules(tmp_path)
    assert reason(r, "a.exe") == "type_not_allowed"
    assert reason(r, "~$draft.txt") == "temp_file"
    for name in ("a.tmp", "a.swp", "a.crdownload", "a.part"):
        assert reason(r, name) is not None, name
    assert reason(r, "a.PNG") is None
    assert ".canvas" in DEFAULT_TYPES


def test_size_caps(tmp_path):
    r = _rules(tmp_path)
    assert reason(r, "a.md", size=5 * 1024 * 1024) is None
    assert reason(r, "a.md", size=5 * 1024 * 1024 + 1) == "too_large"
    assert reason(r, "a.pdf", size=50 * 1024 * 1024) is None
    assert reason(r, "a.pdf", size=50 * 1024 * 1024 + 1) == "too_large"
    assert reason(r, "a.png", size=25 * 1024 * 1024 + 1) == "too_large"


def test_include_types_narrows(tmp_path):
    rules = IgnoreRules(tmp_path, include_types=(".md",))
    rules.enter_dir("")
    assert rules.skip_reason("a.txt", is_dir=False, size=1) == "type_not_allowed"
    assert rules.skip_reason("a.md", is_dir=False, size=1) is None


def test_gitignore_and_slmignore(tmp_path):
    r = _rules(tmp_path, gitignore="drafts/\n*.md\n")
    r2 = reason(r, "a.md")
    assert r2 == "ignored"
    assert reason(r, "drafts", is_dir=True) == "ignored"


def test_slmignore_overrides_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("*.md\n")
    (tmp_path / ".slmignore").write_text("!keep.md\nsecret.txt\n")
    r = IgnoreRules(tmp_path)
    r.enter_dir("")
    assert reason(r, "keep.md") is None
    assert reason(r, "other.md") == "ignored"
    assert reason(r, "secret.txt") == "ignored"


def test_nested_ignore_file_applies_only_below_its_folder(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".gitignore").write_text("*.md\n")
    r = IgnoreRules(tmp_path)
    r.enter_dir("")
    r.enter_dir("sub")
    assert reason(r, "sub/a.md") == "ignored"
    assert reason(r, "a.md") is None
    assert reason(r, "other/a.md") is None


def test_fixed_rules_cannot_be_negated(tmp_path):
    (tmp_path / ".slmignore").write_text("!id_rsa\n!.env\n")
    r = IgnoreRules(tmp_path)
    r.enter_dir("")
    assert reason(r, "id_rsa") == "secret_name"
    assert reason(r, ".env") == "dotfile"


def test_unreadable_ignore_file_is_not_fatal(tmp_path):
    (tmp_path / ".gitignore").write_bytes(b"\xff\xfe\x00bad")
    r = IgnoreRules(tmp_path)
    r.enter_dir("")
    assert reason(r, "a.md") is None
