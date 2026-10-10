"""Credentials quarantine a file; the remote rule pauses a source."""

from __future__ import annotations

import hashlib
import json

from superlocalmemory import sources
from superlocalmemory.sources.store import SourceStore


def fake_key() -> str:
    return "AKIA" + "ABCDEFGHIJKLMNOP"


def test_file_with_a_key_shape_is_quarantined_not_ingested(env):
    env.write("keys.md", f"my aws key {fake_key()} is here")
    env.write("ok.md", "nothing secret")
    sid = env.add_and_confirm()
    stats = env.scan(sid)
    assert env.runtime.contents() == ["nothing secret"]
    row = env.files(sid)["keys.md"]
    assert row["state"] == "quarantined" and "AWS" in row["reason"]
    assert fake_key() not in json.dumps(row)
    assert stats.quarantined == 1


def test_sha256_hex_is_not_a_credential(env):
    digest = hashlib.sha256(b"x").hexdigest()
    env.write("sums.md", f"checksum {digest}")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.files(sid)["sums.md"]["state"] == "indexed"
    assert len(env.runtime.saved) == 1


def test_secret_beyond_the_first_256k_is_not_screened(env):
    env.write("big.md", "a " * 140_000 + fake_key())
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.files(sid)["big.md"]["state"] == "indexed"


def test_quarantine_can_be_released_and_is_then_redacted(env):
    env.write("keys.md", f"key {fake_key()} end")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert sources.release_file(sid, "keys.md") is True
    env.scan(sid)
    assert env.files(sid)["keys.md"]["state"] == "indexed"
    assert len(env.runtime.saved) == 1


def test_changed_file_that_gains_a_key_is_quarantined_and_old_hidden(env):
    import os
    path = env.write("a.md", "clean")
    sid = env.add_and_confirm()
    env.scan(sid)
    path.write_text(f"now {fake_key()}")
    os.utime(path, ns=(2_000_000_000_000_000_000,) * 2)
    env.scan(sid)
    assert env.files(sid)["a.md"]["state"] == "quarantined"
    assert env.runtime.archived == ["f1"] and len(env.runtime.saved) == 1


def test_remote_access_pauses_the_source_and_ingests_nothing(env):
    env.write("a.md", "first")
    sid = env.add_and_confirm()
    env.scan(sid)
    env.write("b.md", "second")
    env.remote = True
    stats = env.scan(sid)
    assert stats.paused and len(env.runtime.saved) == 1 and env.runtime.archived == []
    media = env.store()
    try:
        src = SourceStore(media).get_source(sid)
    finally:
        media.close()
    assert src["state"] == "paused"
    assert json.loads(src["last_scan_stats_json"])["paused_reason"] == "remote_access_on"
    assert env.files(sid)["a.md"]["state"] == "indexed"


def test_paused_source_resumes_when_remote_access_is_gone(env):
    env.write("a.md", "first")
    sid = env.add_and_confirm()
    env.remote = True
    env.scan(sid)
    env.remote = False
    env.scan(sid)
    assert len(env.runtime.saved) == 1
    media = env.store()
    try:
        assert SourceStore(media).get_source(sid)["state"] == "active"
    finally:
        media.close()


def test_a_raising_remote_check_counts_as_on(env):
    env.write("a.md", "first")
    sid = env.add_and_confirm()

    def boom():
        raise RuntimeError("cannot tell")

    env.host.remote_check = boom
    stats = env.scan(sid)
    assert stats.paused and env.runtime.saved == []


def test_release_covers_only_the_content_that_was_reviewed(env):
    import os

    path = env.write("k.md", "token ghp_" + "a" * 36)
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.files(sid)["k.md"]["state"] == "quarantined"
    assert sources.release_file(sid, "k.md")
    path.write_text("new content with AKIAABCDEFGHIJKLMNOP and ghp_" + "b" * 36)
    os.utime(path, ns=(2_000_000_000_000_000_000,) * 2)
    env.scan(sid)
    assert env.files(sid)["k.md"]["state"] == "quarantined" and env.runtime.saved == []


def test_release_queues_a_scan_and_still_lets_the_reviewed_content_in(env):
    env.write("k.md", "token ghp_" + "a" * 36)
    sid = env.add_and_confirm()
    env.scan(sid)
    media = env.store()
    from superlocalmemory.sources.store import SourceStore

    SourceStore(media).cancel_scans(sid)
    media.close()
    assert sources.release_file(sid, "k.md")
    media = env.store()
    queued = media.list_jobs("default", ["queued"])
    media.close()
    assert len(queued) == 1
    env.scan(sid)
    assert env.files(sid)["k.md"]["state"] == "indexed"
