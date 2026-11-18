# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-7d (M-16): the need gate decides which prompts get memories, and the
skip threshold comes from conformal risk control over logged turns."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from superlocalmemory.core import need_gate as ng
from superlocalmemory.outcomes import receipts
from tests.test_outcomes.test_injection_receipts import _fact

KEY = b"k" * 32


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


# -- rules and score ------------------------------------------------------------------------------

@pytest.mark.parametrize(("prompt", "expected"), [
    ("", ng.RULE_EMPTY), ("  ", ng.RULE_EMPTY), ("yes", ng.RULE_ACK), ("ok go ahead", ng.RULE_ACK),
    ("Thanks!", ng.RULE_ACK), ("good morning", ng.RULE_GREETING), ("hi claude", ng.RULE_GREETING),
    ("2+2", ng.RULE_ARITHMETIC), ("what is 17 * 23?", ng.RULE_ARITHMETIC),
    ("translate 'where is the station' into Hindi", ng.RULE_TRANSLATION),
    ("/compact", ng.RULE_COMMAND), ("/code-review ultra", ng.RULE_COMMAND),
])
def test_rules_skip_the_obvious_turns(prompt, expected) -> None:
    assert ng.rule(prompt) == expected


@pytest.mark.parametrize("prompt", [
    "yes but what did we decide about the cache key last time?",
    "fix the failing test in src/app/main.py", "2+2 is what the test asserts, why does it fail",
    "translate", "good morning, what is left on the release checklist?",
    "/usr/bin/env is missing on the runner",
])
def test_rules_never_skip_a_real_request(prompt) -> None:
    assert ng.rule(prompt) is None


def test_the_score_ranks_memory_cues_above_plain_text() -> None:
    cue = ng.score("what did we decide last time about MemoryEngine retention?")
    plain = ng.score("write a haiku")
    assert 0.0 <= plain < cue <= 1.0
    assert ng.score("x " * 5000) <= 1.0


# -- conformal risk control -----------------------------------------------------------------------

def _rows(spec):
    return [ng.Labelled(None, s, needed, w) for s, needed, w, n in spec for _ in range(n)]


def test_with_too_few_labels_the_score_skips_nothing() -> None:
    assert ng.calibrate([]) is None
    # (0 + 1) / (n + 1) <= 0.10 needs n >= 9 clean turns before anything is skipped
    assert ng.calibrate(_rows([(0.2, False, 1.0, 8)])) is None
    assert ng.calibrate(_rows([(0.2, False, 1.0, 9)])) == math.nextafter(0.2, math.inf)


def test_the_threshold_stops_before_needed_turns_cost_more_than_alpha() -> None:
    rows = _rows([(0.2, False, 1.0, 100), (0.6, True, 1.0, 100), (0.9, True, 1.0, 100)])
    lam = ng.calibrate(rows)
    assert lam == math.nextafter(0.2, math.inf)
    assert ng.bound(rows, lam) <= ng.NEED_ALPHA < ng.bound(rows, math.nextafter(0.6, math.inf))
    skip = ng.decide("write a haiku", threshold=lam, key=None)
    assert ng.score("write a haiku") < lam and not skip.recall


def test_the_bound_is_exactly_the_weighted_conformal_formula() -> None:
    rows = _rows([(0.2, True, 20.0, 1), (0.2, False, 1.0, 100), (0.7, True, 1.0, 10)])
    # skipping score 0.2: loss 20, total 130, w_max 20
    assert ng.bound(rows, 0.3) == pytest.approx((20 + 20) / (130 + 20))
    assert ng.bound(rows, None) == pytest.approx(20 / 150)


def test_one_explored_miss_counts_as_much_as_it_stands_for() -> None:
    clean = _rows([(0.2, False, 1.0, 200)])
    assert ng.calibrate(clean) is not None
    missed = clean + [ng.Labelled(None, 0.2, True, 1 / ng.EXPLORE)]
    assert ng.calibrate(missed) is None


def test_rules_that_miss_needed_turns_stop_the_score_from_skipping() -> None:
    rows = [ng.Labelled(ng.RULE_ACK, 0.0, True, 1.0) for _ in range(30)] + \
        _rows([(0.2, False, 1.0, 200)])
    assert ng.bound(rows, None) > ng.NEED_ALPHA
    assert ng.calibrate(rows) is None


# -- one turn's decision --------------------------------------------------------------------------

def test_a_let_through_turn_is_recalled_with_certainty() -> None:
    d = ng.decide("what did we decide about the release branch?", threshold=0.1, key=KEY)
    assert d.recall and not d.explored and d.propensity == 1.0 and d.rule is None


def test_skipped_turns_are_recalled_to_learn_at_the_logged_rate() -> None:
    picks = [ng.decide("yes", threshold=None, key=KEY) for _ in range(4000)]
    explored = [d for d in picks if d.explored]
    assert all(d.rule == ng.RULE_ACK for d in picks)
    assert all(d.recall and d.propensity == ng.EXPLORE for d in explored)
    assert all(not d.recall and d.propensity == 1 - ng.EXPLORE
               for d in picks if not d.explored)
    share = len(explored) / len(picks)
    assert abs(share - ng.EXPLORE) < 4 * math.sqrt(ng.EXPLORE * (1 - ng.EXPLORE) / len(picks))


def test_without_a_key_nothing_is_explored_and_nothing_is_labelled() -> None:
    d = ng.decide("yes", threshold=None, key=None)
    assert not d.recall and not d.explored and d.propensity == 0.0


# -- the log and its labels on a real store -------------------------------------------------------

def _turn(prompt, at, *, recall=True, explored=False, p=1.0, rule=None, s=0.5):
    return ng.Decision(f"turn_{at.timestamp()}_{prompt}", recall, rule, s, None, explored, p, at)


def _pull(store, tmp_path, at, *, path="hook_recall", query_id=None, facts=()):
    assert receipts.record_pull(receipts.Pull(
        tmp_path, store.db_path, "default", "conv-1", path, "q",
        tuple((f, 0.5) for f in facts), "hook", at=at, recall_query_id=query_id))


def _use(store, fact, at, query_id):
    conn = sqlite3.connect(str(store.db_path))
    did = conn.execute("SELECT delivery_id FROM injection_deliveries WHERE recall_query_id = ?",
                       (query_id,)).fetchone()[0]
    conn.execute("INSERT INTO injection_usage VALUES (?,?,?,?,?,?)",
                 (did, fact, "default", "marker_edit", None, at.isoformat()))
    conn.commit()
    conn.close()


def test_labels_come_from_use_and_from_explicit_pulls(store, tmp_path) -> None:
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    t0 = now - timedelta(hours=30)
    a = _fact(store, "the release branch is cut on Thursdays")
    used = _turn("used", t0)
    pulled = _turn("pulled", t0 + timedelta(minutes=10), s=0.3)
    idle = _turn("idle", t0 + timedelta(minutes=20), s=0.7)
    fresh = _turn("fresh", now - timedelta(hours=2))
    skipped = _turn("skipped", t0 + timedelta(minutes=25), recall=False, p=0.95, rule="ack")
    explored = _turn("explored", t0 + timedelta(minutes=26), explored=True, p=ng.EXPLORE,
                     rule="ack")
    for d, qid in ((used, "q-used"), (pulled, None), (idle, None), (fresh, None),
                   (skipped, None), (explored, None)):
        assert ng.log(store.db_path, d, profile_id="default", session_id="conv-1",
                      prompt_digest=None, recall_query_id=qid)
    _pull(store, tmp_path, t0 + timedelta(seconds=5), query_id="q-used", facts=(a,))
    _use(store, a, t0 + timedelta(minutes=2), "q-used")
    _pull(store, tmp_path, t0 + timedelta(minutes=12), path="mcp_recall", facts=(a,))
    # a hook's own delivery after a turn is not the agent asking: idle stays unneeded
    _pull(store, tmp_path, t0 + timedelta(minutes=21), path="hook_recall", facts=(a,))
    conn = sqlite3.connect(str(store.db_path))
    got = ng.labelled(conn, "default", now=now)
    conn.close()
    assert len(got) == 4  # fresh: window open; skipped: no label
    by_score = sorted((r.score, r.needed, r.weight, r.rule) for r in got)
    assert by_score == [(0.3, True, 1.0, None), (0.5, False, 1 / ng.EXPLORE, "ack"),
                        (0.5, True, 1.0, None), (0.7, False, 1.0, None)]


def test_a_use_after_the_outcome_window_is_not_a_label(store, tmp_path) -> None:
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    t0 = now - timedelta(days=3)
    a = _fact(store, "the gate code is 41")
    ng.log(store.db_path, _turn("late", t0), profile_id="default", session_id="conv-1",
           prompt_digest=None, recall_query_id="q-late")
    _pull(store, tmp_path, t0, query_id="q-late", facts=(a,))
    _use(store, a, t0 + timedelta(hours=ng.OUTCOME_HOURS + 1), "q-late")
    conn = sqlite3.connect(str(store.db_path))
    assert [r.needed for r in ng.labelled(conn, "default", now=now)] == [False]
    conn.close()


def test_a_store_without_the_log_has_no_sample_and_says_so() -> None:
    conn = sqlite3.connect(":memory:")
    assert ng.labelled(conn, "default") == []
    assert ng.report(conn, "default") == {"turns": None}
    assert "no per-prompt turns" in next(iter(ng.describe({"turns": None})))


def test_the_report_counts_turns_and_states_the_caveat(store) -> None:
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    for i in range(3):
        ng.log(store.db_path, _turn(f"t{i}", now - timedelta(hours=i), recall=i != 1,
                                    p=1.0 if i != 1 else 0.95),
               profile_id="default", session_id="conv-1", prompt_digest=None)
    conn = sqlite3.connect(str(store.db_path))
    section = ng.report(conn, "default", now=now)
    conn.close()
    assert section["turns"] == 3 and section["skipped"] == 1 and section["threshold"] is None
    text = "\n".join(ng.describe(section))
    assert "skips nothing yet" in text and "detection misses some uses" in text


# -- never shown twice in one conversation --------------------------------------------------------

def test_what_a_conversation_already_got_is_known(store, tmp_path) -> None:
    a, b = _fact(store, "fact a about the cache"), _fact(store, "fact b about the cache")
    _pull(store, tmp_path, datetime.now(UTC), path="mcp_recall", facts=(a,))
    assert ng.shown_in_session(store.db_path, "default", "conv-1") == {a}
    assert ng.shown_in_session(store.db_path, "default", "conv-2") == frozenset()
    assert ng.shown_in_session(store.db_path, "default", "") == frozenset()
    assert b not in ng.shown_in_session(store.db_path, "default", "conv-1")


def test_the_hook_adapter_drops_what_was_already_shown(store, tmp_path) -> None:
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    a, b = _fact(store, "fact a about the cache"), _fact(store, "fact b about the cache")
    _pull(store, tmp_path, datetime.now(UTC), path="mcp_recall",
          facts=(a,))
    engine = SimpleNamespace(_config=SimpleNamespace(base_dir=tmp_path),
                             db=SimpleNamespace(db_path=store.db_path), profile_id="default")
    adapter = EngineRecallAdapter(engine)
    out = {"results": [{"fact_id": a}, {"fact_id": b}], "result_count": 2}
    kept = adapter._not_yet_shown(out, "conv-1")
    assert [r["fact_id"] for r in kept["results"]] == [b] and kept["already_shown"] == 1
    assert adapter._not_yet_shown(out, "conv-2") is out
    assert adapter._not_yet_shown(out, "") is out
    gate, profile = adapter.need_gate()
    assert isinstance(gate, ng.Gate) and profile == "default"
    assert adapter.need_gate()[0] is gate  # one gate per store: its cache survives


# -- the daemon path ------------------------------------------------------------------------------

class _Gate:
    def __init__(self, recall: bool) -> None:
        self.decision = ng.Decision("turn_x", recall, None if recall else "ack", 0.4, None,
                                    False, 1.0 if recall else 0.95, datetime.now(UTC))
        self.logged: list = []

    def decide(self, prompt, profile_id):
        return self.decision

    def record(self, decision, **kw):
        self.logged.append((decision, kw))


class _Queue:
    def __init__(self) -> None:
        self.asked: list = []

    def enqueue(self, **kw):
        self.asked.append(kw)
        return "r1"

    def poll_result(self, request_id, timeout_s):
        assert timeout_s <= 4.0
        return {"ok": True, "query_id": "qid-1", "results": []}


def _daemon(gate):
    from superlocalmemory.hooks.hook_daemon import HookDaemon

    adapter = SimpleNamespace(need_gate=lambda: (gate, "default"))
    d = HookDaemon(recall_adapter=adapter)
    d._queue = _Queue()
    return d


def test_a_skipped_prompt_is_logged_and_never_recalled(monkeypatch) -> None:
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._switched_off", lambda: False)
    gate = _Gate(recall=False)
    d = _daemon(gate)
    assert d._process_request({"prompt": "yes", "session_id": "conv-1"}) == {}
    assert d._queue.asked == []
    assert gate.logged[0][1]["recall_query_id"] is None


def test_a_recalled_prompt_is_logged_with_its_answer_id(monkeypatch) -> None:
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._switched_off", lambda: False)
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._detect_mode", lambda: "B")
    gate = _Gate(recall=True)
    d = _daemon(gate)
    d._process_request({"prompt": "what did we decide about caching?", "session_id": "conv-1"})
    assert len(d._queue.asked) == 1 and d._queue.asked[0]["stall_timeout_s"] <= 4.0
    (decision, kw), = gate.logged
    assert decision.recall and kw["recall_query_id"] == "qid-1" and kw["session_id"] == "conv-1"


def test_a_broken_gate_falls_back_to_the_acknowledgement_rule(monkeypatch) -> None:
    from superlocalmemory.hooks.hook_daemon import HookDaemon

    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._switched_off", lambda: False)
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._detect_mode", lambda: "A")

    def broken():
        raise RuntimeError("no store")

    d = HookDaemon(recall_adapter=SimpleNamespace(need_gate=broken))
    d._queue = _Queue()
    assert d._process_request({"prompt": "ok", "session_id": "s"}) == {}
    assert d._queue.asked == []
    d._process_request({"prompt": "what did we decide about caching?", "session_id": "s"})
    assert len(d._queue.asked) == 1


def test_the_report_carries_the_need_gate(store) -> None:
    from superlocalmemory.outcomes import memory_report as mr

    conn = sqlite3.connect(str(store.db_path))
    report = mr.build(conn, "default")
    conn.close()
    assert report["need_gate"]["turns"] == 0
    assert any("Need gate" in line for line in mr.describe(report))


# -- the flag (D-47) ------------------------------------------------------------------------------

def test_the_gate_is_off_by_default_and_on_the_kill_switch(store, tmp_path, monkeypatch) -> None:
    from superlocalmemory.core.feature_flags import FlagState, write_flag

    gate = ng.Gate(tmp_path, store.db_path)
    assert ng.flag_state(tmp_path) is None and gate.decide("what did we decide?", "default") is None
    write_flag(tmp_path, "delivery.need_gate", FlagState.ON)
    assert gate.decide("what did we decide?", "default").recall
    monkeypatch.setenv("SLM_4230_OFF", "1")
    assert gate.decide("what did we decide?", "default") is None


def test_in_shadow_the_threshold_is_logged_and_never_applied(store, tmp_path,
                                                            monkeypatch) -> None:
    from superlocalmemory.core.feature_flags import FlagState, write_flag

    gate = ng.Gate(tmp_path, store.db_path)
    monkeypatch.setattr(gate, "threshold", lambda profile_id: 0.99)
    write_flag(tmp_path, "delivery.need_gate", FlagState.SHADOW)
    d = gate.decide("write a haiku", "default")
    assert d.recall and d.threshold == 0.99 and d.acting is False
    write_flag(tmp_path, "delivery.need_gate", FlagState.ON)
    d = gate.decide("write a haiku", "default")
    assert d.threshold == 0.99 and d.acting is True and (not d.recall or d.explored)
    assert ng.log(store.db_path, d, profile_id="default", session_id="conv-1", prompt_digest=None)
    conn = sqlite3.connect(str(store.db_path))
    assert conn.execute("SELECT acting, threshold FROM need_turns").fetchone() == (1, 0.99)
    conn.close()


def test_an_off_gate_answers_nothing_and_logs_nothing(monkeypatch) -> None:
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._switched_off", lambda: False)
    gate = _Gate(recall=True)
    gate.decide = lambda prompt, profile_id: None
    d = _daemon(gate)
    assert d._process_request({"prompt": "what did we decide?", "session_id": "s"}) == {}
    assert d._queue.asked == [] and gate.logged == []
