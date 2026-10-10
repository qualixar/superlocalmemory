"""The walk looks at files with lstat, so a link swapped in after the listing is never followed."""

from __future__ import annotations

import os

import superlocalmemory.sources.walk as walk
from superlocalmemory.sources.ignore import IgnoreRules


class _Stale:
    """A directory item whose cached answers are out of date: it says it is a plain file or folder."""

    def __init__(self, inner, log):
        self._i, self._log = inner, log
        self.name, self.path = inner.name, inner.path

    def is_symlink(self):
        return False

    def is_dir(self, follow_symlinks=True):
        return self._i.is_dir(follow_symlinks=False)

    def is_file(self, follow_symlinks=True):
        return not self._i.is_dir(follow_symlinks=False)

    def stat(self, follow_symlinks=True):
        self._log.append(("stat", self.name))
        return self._i.stat(follow_symlinks=follow_symlinks)


def _patched_scandir(monkeypatch, log):
    real = os.scandir

    class _Ctx:
        def __init__(self, path):
            self._c = real(path)

        def __enter__(self):
            return [_Stale(e, log) for e in self._c.__enter__()]

        def __exit__(self, *a):
            return self._c.__exit__(*a)

    monkeypatch.setattr(walk.os, "scandir", _Ctx)


def test_a_link_swapped_in_after_the_listing_is_skipped(tmp_path, monkeypatch):
    root = tmp_path / "r"
    root.mkdir()
    (root / "real.md").write_text("real")
    os.symlink(root / "real.md", root / "swapped.md")
    _patched_scandir(monkeypatch, [])
    out = walk.walk_tree(root, IgnoreRules(root, (".md",)))
    assert [e.relpath for e in out.entries] == ["real.md"]
    assert out.skipped.get("symlink_file") == 1


def test_directories_are_not_stat_again(tmp_path, monkeypatch):
    root = tmp_path / "r"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.md").write_text("a")
    log: list = []
    _patched_scandir(monkeypatch, log)
    out = walk.walk_tree(root, IgnoreRules(root, (".md",)))
    assert [e.relpath for e in out.entries] == ["sub/a.md"]
    assert log == []
