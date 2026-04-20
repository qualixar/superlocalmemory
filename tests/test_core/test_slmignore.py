# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 SB-5

"""Tests for core.slmignore — path-level SLM opt-out."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core import slmignore


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    slmignore.clear_cache()


def _write_slmignore(dir_: Path, *lines: str) -> None:
    (dir_ / ".slmignore").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_no_ignore_file_means_not_ignored(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text("x")
    assert slmignore.path_is_ignored(p) is False


def test_segment_match_anywhere(tmp_path: Path) -> None:
    _write_slmignore(tmp_path, "node_modules")
    nested = tmp_path / "pkg" / "node_modules" / "react" / "index.js"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    assert slmignore.path_is_ignored(nested) is True


def test_hash_comment_line_ignored(tmp_path: Path) -> None:
    _write_slmignore(tmp_path, "# private workspace", "secrets")
    target = tmp_path / "secrets" / "keys.env"
    target.parent.mkdir()
    target.write_text("x")
    assert slmignore.path_is_ignored(target) is True
    # Comment text itself should not match a directory named "#"
    assert slmignore.path_is_ignored(tmp_path / "other.txt") is False


def test_empty_lines_ignored(tmp_path: Path) -> None:
    _write_slmignore(tmp_path, "", "   ", "secrets")
    target = tmp_path / "secrets" / "a"
    target.parent.mkdir()
    target.write_text("x")
    assert slmignore.path_is_ignored(target) is True


def test_absolute_prefix_match(tmp_path: Path) -> None:
    inside = tmp_path / "work" / "deep"
    inside.mkdir(parents=True)
    _write_slmignore(tmp_path, f"{tmp_path / 'work'}")
    leaf = inside / "file.py"
    leaf.write_text("x")
    assert slmignore.path_is_ignored(leaf) is True
    assert slmignore.path_is_ignored(tmp_path / "other.txt") is False


def test_unrelated_path_not_ignored(tmp_path: Path) -> None:
    _write_slmignore(tmp_path, "node_modules")
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("x")
    assert slmignore.path_is_ignored(target) is False


def test_ignore_file_applies_to_descendants_only(tmp_path: Path) -> None:
    sub = tmp_path / "work"
    sub.mkdir()
    _write_slmignore(sub, "secrets")
    # A sibling directory unaffected — the .slmignore lives at work/
    # and only ancestors of the target are consulted, so the sibling
    # escapes the rule even though it contains a "secrets" segment.
    sibling = tmp_path / "other" / "secrets" / "file.txt"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("x")
    assert slmignore.path_is_ignored(sibling) is False

    # Descendant of work/ does get ignored.
    inside = sub / "secrets" / "key.env"
    inside.parent.mkdir()
    inside.write_text("x")
    assert slmignore.path_is_ignored(inside) is True


def test_descendant_of_ignored_dir_is_ignored(tmp_path: Path) -> None:
    _write_slmignore(tmp_path, "build")
    nested = tmp_path / "src" / "build" / "out" / "chunk.js"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    assert slmignore.path_is_ignored(nested) is True


def test_non_existent_target_path_is_valid(tmp_path: Path) -> None:
    _write_slmignore(tmp_path, "secrets")
    target = tmp_path / "secrets" / "does_not_exist_yet" / "file.json"
    assert slmignore.path_is_ignored(target) is True


def test_cache_honours_mtime(tmp_path: Path) -> None:
    _write_slmignore(tmp_path, "node_modules")
    target = tmp_path / "secrets" / "k.env"
    target.parent.mkdir()
    target.write_text("x")
    assert slmignore.path_is_ignored(target) is False
    # Rewrite the ignore file with a different pattern. mtime changes.
    import time
    time.sleep(0.01)
    _write_slmignore(tmp_path, "secrets")
    assert slmignore.path_is_ignored(target) is True
