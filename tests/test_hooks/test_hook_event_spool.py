# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The hook event spool: where a hook leaves an event the daemon did not ack.

Append is the hook's only write when the daemon is away. It must be fast,
never tear a line across concurrent hook processes, respect a size cap, and
count every event it refuses so ``slm doctor`` can report it.
"""

from __future__ import annotations

import json
import subprocess
import time
import sys
import textwrap
from pathlib import Path

import pytest

from superlocalmemory.hooks import event_spool


def _ev(i: int, **extra) -> dict:
    return {"kind": "tool_event", "event_id": f"{i:032x}", "profile_id": "default",
            "session_id": "s", "tool_name": "Read", **extra}


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def test_append_writes_one_json_line_per_event(tmp_path: Path) -> None:
    result = event_spool.append_events([_ev(1), _ev(2)], profile="default", root=tmp_path)
    assert (result.written, result.dropped) == (2, 0)
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert [e["event_id"] for e in _lines(files[0])] == [f"{1:032x}", f"{2:032x}"]


def test_spool_file_is_private(tmp_path: Path) -> None:
    if sys.platform == "win32":  # pragma: no cover - POSIX modes only
        pytest.skip("POSIX permissions")
    event_spool.append_events([_ev(1)], profile="default", root=tmp_path)
    (path,) = tmp_path.glob("*.jsonl")
    assert path.stat().st_mode & 0o077 == 0


def test_unsafe_profile_names_never_escape_the_spool_folder(tmp_path: Path) -> None:
    root = tmp_path / "spool"
    event_spool.append_events([_ev(1)], profile="../../etc/evil", root=root)
    created = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert created and all(root in p.parents for p in created)


def test_cap_refuses_new_events_and_counts_them(tmp_path: Path) -> None:
    first = event_spool.append_events([_ev(i) for i in range(5)], profile="default",
                                      root=tmp_path, max_bytes=400)
    assert first.written >= 1 and first.dropped >= 1
    second = event_spool.append_events([_ev(9)], profile="default", root=tmp_path,
                                       max_bytes=400)
    assert (second.written, second.dropped) == (0, 1)
    stats = event_spool.read_stats(tmp_path)
    assert stats.dropped == first.dropped + 1
    assert stats.last_drop_at


def test_an_oversized_event_is_counted_not_written(tmp_path: Path) -> None:
    huge = _ev(1, output_summary="x" * (event_spool.MAX_LINE_BYTES + 10))
    result = event_spool.append_events([huge], profile="default", root=tmp_path)
    assert (result.written, result.dropped) == (0, 1)
    assert event_spool.read_stats(tmp_path).dropped == 1


def test_claim_moves_active_file_aside_in_order(tmp_path: Path) -> None:
    event_spool.append_events([_ev(1)], profile="default", root=tmp_path)
    first = event_spool.claim_ready(tmp_path)
    event_spool.append_events([_ev(2)], profile="default", root=tmp_path)
    second = event_spool.claim_ready(tmp_path)
    assert len(first) == 1 and first[0].suffix == ".draining"
    # Unfinished draining files come back first, then the newly claimed one.
    assert second[0] == first[0] and len(second) == 2
    assert [_lines(p)[0]["event_id"] for p in second] == [f"{1:032x}", f"{2:032x}"]
    assert not list(tmp_path.glob("*.jsonl"))


def test_pending_counts_events_waiting(tmp_path: Path) -> None:
    event_spool.append_events([_ev(1), _ev(2)], profile="default", root=tmp_path)
    event_spool.claim_ready(tmp_path)
    event_spool.append_events([_ev(3)], profile="work", root=tmp_path)
    stats = event_spool.read_stats(tmp_path)
    assert stats.pending_events == 3 and stats.dropped == 0


def test_stats_on_a_missing_folder_are_empty(tmp_path: Path) -> None:
    stats = event_spool.read_stats(tmp_path / "nope")
    assert (stats.pending_events, stats.dropped, stats.invalid) == (0, 0, 0)
    assert not (tmp_path / "nope").exists()


def test_invalid_count_is_recorded(tmp_path: Path) -> None:
    event_spool.record_counts(tmp_path, "default", invalid=3)
    assert event_spool.read_stats(tmp_path).invalid == 3


_WRITER = textwrap.dedent("""
    import sys
    from pathlib import Path
    from superlocalmemory.hooks import event_spool
    root, me = Path(sys.argv[1]), int(sys.argv[2])
    for i in range(200):
        ev = {"kind": "tool_event", "event_id": f"{me:08x}{i:024x}", "profile_id": "default",
              "session_id": "s", "tool_name": "Read", "output_summary": "y" * 900}
        r = event_spool.append_events([ev], profile="default", root=root)
        assert r.written == 1, r
""")


def test_concurrent_hook_processes_never_tear_a_line(tmp_path: Path) -> None:
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(tmp_path), str(n)])
             for n in range(6)]
    # Claims racing the writers is the daemon's normal life, at the daemon's
    # fastest pace (hook_event_writer floors its interval at 50 ms). A claim
    # loop with no pause starved the writers past their 1 s lock wait on a
    # loaded test host, which no daemon does.
    claimed: dict[Path, None] = {}
    while any(p.poll() is None for p in procs):
        claimed.update(dict.fromkeys(event_spool.claim_ready(tmp_path)))
        time.sleep(0.05)
    assert all(p.wait(timeout=120) == 0 for p in procs)
    claimed.update(dict.fromkeys(event_spool.claim_ready(tmp_path)))
    ids = [e["event_id"] for p in claimed for e in _lines(p)]
    assert len(ids) == 1200 and len(set(ids)) == 1200


def test_a_dotted_profile_never_shares_files_with_another(tmp_path: Path) -> None:
    event_spool.append_events([_ev(1)], profile="work", root=tmp_path)
    event_spool.append_events([_ev(2)], profile="work.v2", root=tmp_path)
    assert event_spool.profile_stem("work.v2") != "work"
    assert "." not in event_spool.profile_stem("work.v2")
    assert event_spool.erase_profile(tmp_path, "work") == 1
    assert event_spool.read_stats(tmp_path).pending_events == 1
