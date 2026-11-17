# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-46: logged adjacent swaps in agents' recalls, to measure position attention."""

from __future__ import annotations

import sqlite3
from collections import Counter
from types import SimpleNamespace

import pytest

from superlocalmemory.core.feature_flags import FLAGS, FeatureFlags, FlagState
from superlocalmemory.outcomes import receipts, swaps
from superlocalmemory.storage.migrations import M075_injection_receipts as M075


@pytest.fixture
def on(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_SWAPS", "on")
    monkeypatch.delenv("SLM_4230_OFF", raising=False)
    return tmp_path


def test_swaps_are_off_by_default_like_withholding() -> None:
    assert FLAGS["outcomes.swaps"].default is FlagState.OFF
    assert not FLAGS["outcomes.swaps"].shadowable


def test_nothing_is_drawn_while_swaps_are_off(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SLM_FEATURE_OUTCOMES_SWAPS", raising=False)
    assert swaps.decide(tmp_path, "d1", 10) is None
    assert swaps.decide(None, "d1", 10) is None


def test_the_kill_switch_and_receipts_off_stop_swaps(on, monkeypatch) -> None:
    monkeypatch.setenv("SLM_4230_OFF", "1")
    assert swaps.decide(on, "d1", 10) is None
    monkeypatch.delenv("SLM_4230_OFF")
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_RECEIPTS", "off")
    assert swaps.decide(on, "d1", 10) is None


def test_the_share_and_the_pair_follow_the_design(on) -> None:
    n = 6000
    got = [swaps.decide(on, f"d{i}", 10) for i in range(n)]
    swapped = [s for s in got if s.at is not None]
    assert all(s.share == swaps.SWAP_SHARE for s in got)
    assert abs(len(swapped) / n - swaps.SWAP_SHARE) < 4 * (0.05 * 0.95 / n) ** 0.5
    pairs = Counter(s.at for s in swapped)
    assert set(pairs) == {1, 2, 3, 4}                      # within the top five only
    assert all(s.draw < swaps.SWAP_SHARE for s in swapped)
    assert all(s.draw >= swaps.SWAP_SHARE for s in got if s.at is None)


def test_the_draw_is_keyed_and_reproducible(on) -> None:
    assert swaps.decide(on, "same", 10) == swaps.decide(on, "same", 10)


def test_short_answers_have_fewer_pairs_and_one_result_none(on, monkeypatch) -> None:
    monkeypatch.setattr(swaps, "SWAP_SHARE", 1.0)
    assert {swaps.decide(on, f"d{i}", 3).at for i in range(200)} == {1, 2}
    one = swaps.decide(on, "d", 1)
    assert one is not None and one.at is None             # logged as a control
    assert swaps.pairs(0) == swaps.pairs(1) == 0 and swaps.pairs(40) == 4


def test_apply_returns_a_new_list_and_swaps_only_the_pair() -> None:
    results = ["a", "b", "c", "d"]
    out = swaps.apply(results, swaps.Swap("d", 0.01, 2, 0.05))
    assert out == ["a", "c", "b", "d"] and results == ["a", "b", "c", "d"]
    assert swaps.apply(results, None) == results
    assert swaps.apply(results, swaps.Swap("d", 0.5, None, 0.05)) == results


def test_m075_repairs_a_table_without_the_swap_columns() -> None:
    import re

    older = re.sub(r"\n    swap_[a-z]+ .*", "", M075.DDL)
    assert "swap_" not in older
    conn = sqlite3.connect(":memory:")
    for statement in older.split(";"):
        if statement.strip():
            conn.execute(statement)
    assert not M075.verify(conn)
    M075.repair(conn)
    assert M075.verify(conn)
    with pytest.raises(sqlite3.IntegrityError):  # a swap needs its draw
        conn.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, path, design,"
                     " holdout_state, epsilon, eligible_count, candidates, delivered_at,"
                     " swap_at) VALUES ('x', 'p', 'mcp_recall', 'd', 'off', 0, 0, 0, 'now', 2)")


def test_a_shared_answer_is_swapped_per_caller_and_logged(on, monkeypatch) -> None:
    from superlocalmemory.server import recall_core

    monkeypatch.setattr(swaps, "SWAP_SHARE", 1.0)
    monkeypatch.setattr(receipts, "engine_paths", lambda e: (on, on / "memory.db"))
    sent = []
    monkeypatch.setattr(receipts, "submit_pull", sent.append)
    shared = {"ok": True, "query_id": "rq", "results": [
        {"fact_id": f"f{i}", "score": 1 - i / 10} for i in range(5)]}
    call = SimpleNamespace(origin="", via="mcp", profile_id="default", session_id="",
                           query="q", agent_id="claude")
    out, swap = recall_core._swapped(object(), call, shared)
    recall_core._receipt(object(), call, out, swap)
    assert [r["fact_id"] for r in shared["results"]] == ["f0", "f1", "f2", "f3", "f4"]
    shown = [r["fact_id"] for r in out["results"]]
    k = swap.at
    assert shown[k - 1] == f"f{k}" and shown[k] == f"f{k - 1}"
    (pull,) = sent
    assert [f for f, _ in pull.results] == shown and pull.swap is swap


def test_a_person_reading_is_never_swapped(on, monkeypatch) -> None:
    from superlocalmemory.core.answer_check_history import ORIGIN_DASHBOARD
    from superlocalmemory.server import recall_core

    monkeypatch.setattr(swaps, "SWAP_SHARE", 1.0)
    monkeypatch.setattr(receipts, "engine_paths", lambda e: (on, on / "memory.db"))
    env = {"ok": True, "results": [{"fact_id": "a"}, {"fact_id": "b"}]}
    agent = SimpleNamespace(origin="", via="mcp", profile_id="d", session_id="",
                            query="q", agent_id="")
    assert recall_core._swapped(object(), agent, env)[1] is not None  # an agent would be
    call = SimpleNamespace(origin=ORIGIN_DASHBOARD, via="", profile_id="d", session_id="",
                           query="q", agent_id="")
    assert recall_core._swapped(object(), call, env) == (env, None)


from tests.test_modes.test_mode_a_never_fails import mode_a_engine  # noqa: E402,F401


def test_a_real_hook_recall_is_swapped_and_its_receipt_says_so(
        mode_a_engine, monkeypatch) -> None:  # noqa: F811
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_SWAPS", "on")
    # Force the draw, not the share: the schema caps the logged share at 0.10.
    monkeypatch.setattr(swaps.randomiser, "draw",
                        lambda key, did, label: 0.0 if label == "swap" else 0.3)
    for text in ("The invoice run happens on the last working day of the month.",
                 "Invoices are approved by the finance lead before the run.",
                 "The invoice archive lives in the shared finance folder."):
        mode_a_engine.store(text)
    adapter = EngineRecallAdapter(mode_a_engine)
    plain = [r["fact_id"] for r in adapter._recall("invoice run", 5, "conv-s")["results"]]
    assert len(plain) >= 2
    out = adapter.recall("invoice run", 5, "conv-s")
    shown = [r["fact_id"] for r in out["results"]]
    assert sorted(shown) == sorted(plain) and shown != plain
    assert receipts.PULL_WRITER.drain()
    conn = sqlite3.connect(str(mode_a_engine.db.db_path))
    (did, at, share) = conn.execute("SELECT delivery_id, swap_at, swap_share FROM"
                                    " injection_deliveries").fetchone()
    stored = [r[0] for r in conn.execute("SELECT fact_id FROM injection_receipts WHERE"
                                         " delivery_id = ? ORDER BY position", (did,))]
    conn.close()
    assert stored == shown and share == swaps.SWAP_SHARE
    assert shown[at - 1] == plain[at] and shown[at] == plain[at - 1]
    assert FeatureFlags(mode_a_engine._config.base_dir).enabled("outcomes.swaps")
