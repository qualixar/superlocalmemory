"""Issue #297 (question 4): a purge on a store with backups finishes, and says what backups hold.

A snapshot that still holds an erased fact (``pre-migration-snapshots/`` from a scale promotion,
``backups/``) cannot be rewritten; the erasure records it as an obligation, replayed on any restore,
and its receipt stays outstanding. A purge used to read that as "erasure incomplete, try again",
which no retry can change, so the folder stayed listed for good and every scan erased again.

Now: the live store is clean, so the purge completes and reports how many snapshots still hold the
memories. Anything else left behind still refuses.
"""

from __future__ import annotations

import pytest

from superlocalmemory import sources
from superlocalmemory.sources.api import SourceRefused


def _eraser(env, counts: dict):
    def erase(profile_id, fact_ids, subject_id):
        env.erased.append((profile_id, tuple(fact_ids), subject_id))
        return dict(counts)

    env.host.eraser = erase


_HELD_BY_A_SNAPSHOT = {"facts": 1, "erasure_complete": 0, "backup_obligations_pending": 1}


def test_purge_completes_and_reports_snapshots_that_still_hold_the_memories(env):
    env.write("a.md", "x")
    env.write("b.md", "y")
    sid = env.add_and_confirm()
    env.scan(sid)
    _eraser(env, _HELD_BY_A_SNAPSHOT)
    report = sources.remove_source(sid, purge=True)
    assert env.files(sid) == {} and sources.list_sources("default") == []
    assert report == {"backup_copies_holding": 1}


def test_purge_with_nothing_held_reports_nothing_held(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert sources.remove_source(sid, purge=True) == {"backup_copies_holding": 0}


@pytest.mark.parametrize("left", [
    {"erasure_complete": 0, "residue_rows": 2},
    {"erasure_complete": 0, "backup_obligations_pending": 1, "residue_rows": 1},
    {"erasure_complete": 0, "backup_obligations_pending": 1, "receipt_persist_failed": 1},
    {"erasure_complete": 0, "backup_scan_failed": 1},
])
def test_anything_else_left_still_refuses(env, left):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    _eraser(env, left)
    with pytest.raises(SourceRefused) as refused:
        sources.remove_source(sid, purge=True)
    assert refused.value.code == "erasure_incomplete"
    assert "a.md" in env.files(sid)


def test_the_grace_period_purge_does_not_erase_again_every_scan(env):
    path = env.write("a.md", "gone soon")
    env.write("keep.canvas", "{}")  # the folder is never empty
    sid = env.add_and_confirm()
    env.scan(sid)
    path.unlink()
    env.scan(sid)
    _eraser(env, _HELD_BY_A_SNAPSHOT)
    env.host.purge_after_s = 0
    env.scan(sid)
    attempts = len(env.erased)
    assert attempts == 1 and "a.md" not in env.files(sid)
    env.scan(sid)
    assert len(env.erased) == attempts


def test_the_cli_says_backups_still_hold_them(monkeypatch, capsys):
    from argparse import Namespace

    from superlocalmemory.cli import sources_cmd

    monkeypatch.setattr(sources_cmd, "daemon_request", lambda *a, **k: {
        "removed": True, "source_id": "s1", "purged": True, "backup_copies_holding": 2})
    sources_cmd.cmd_sources(Namespace(sources_command="remove", source_id="a" * 32, purge=True,
                                      yes=True, json=False))
    out = capsys.readouterr().out
    assert "2 backup copies still hold" in out and "pre-migration-snapshots/" in out
