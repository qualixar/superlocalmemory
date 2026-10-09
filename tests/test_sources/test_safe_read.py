"""Files are opened without following links or blocking, and only when they are the file the walk saw."""

from __future__ import annotations

import os
import threading

import pytest

from superlocalmemory.sources import preview as preview_mod
from superlocalmemory.sources.ignore import IgnoreRules
from superlocalmemory.sources.reconcile import _digest
from superlocalmemory.sources.walk import Entry

pytestmark = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")


def within(seconds, fn, *args):
    """Run ``fn``; (done, result_or_exception). A call still blocked after ``seconds`` is not done."""
    box = {}

    def run():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    return (not t.is_alive()), box


def release(path):
    """Unblock a reader stuck on a pipe (only reached when the code under test blocked)."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
    except OSError:
        pass


def test_a_pipe_in_place_of_a_file_never_blocks_the_hash(tmp_path):
    fifo = tmp_path / "n.md"
    os.mkfifo(fifo)
    done, box = within(2, _digest, fifo, None, None)
    release(fifo)
    assert done and isinstance(box.get("error"), OSError)


def test_a_link_in_place_of_a_file_is_not_followed(tmp_path):
    target = tmp_path / "real.md"
    target.write_text("secret place")
    link = tmp_path / "n.md"
    link.symlink_to(target)
    with pytest.raises(OSError):
        _digest(link, None, None)


def test_a_file_that_is_not_the_one_the_walk_saw_is_refused(tmp_path):
    path = tmp_path / "n.md"
    path.write_text("x")
    st = os.stat(path)
    assert _digest(path, None, f"{st.st_dev}:{st.st_ino}")[0]
    with pytest.raises(OSError):
        _digest(path, None, f"{st.st_dev}:{st.st_ino + 1}")


def test_a_pipe_named_like_an_ignore_file_never_blocks_the_walk(tmp_path):
    os.mkfifo(tmp_path / ".gitignore")
    rules = IgnoreRules(tmp_path)
    done, box = within(2, rules.enter_dir, "")
    release(tmp_path / ".gitignore")
    assert done and "error" not in box


def test_the_preview_never_blocks_on_a_pipe(tmp_path):
    os.mkfifo(tmp_path / "n.md")
    done, box = within(2, preview_mod._quarantined, tmp_path, [Entry("n.md", 0, 0, "1:1")])
    release(tmp_path / "n.md")
    assert done and box.get("value") == 0


def test_a_scan_skips_a_swapped_in_link_and_does_not_read_it(env):
    path = env.write("n.md", "a note")
    sid = env.add_and_confirm()
    path.unlink()
    path.symlink_to(env.write("elsewhere.txt", "other words"))
    stats = env.scan(sid)
    assert stats.errors == 0 and stats.skipped == {"symlink_file": 1} and "n.md" not in env.files(sid)
    assert env.runtime.contents() == ["other words"]  # only the real file was read
