# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track D.1 (Stage 6 lock-in test)

"""Cross-platform learning-transfer integration test.

Stage-6 Track D.1 per IMPLEMENTATION-MANIFEST-v3.4.22-FINAL.md §D.1 /
MASTER-PLAN §4.8 / gap-audit item #8.

Goal
----
Prove that when three different host platforms (Claude Code, Cursor,
Antigravity) drive recall+outcome writes against the *same* SLM profile,
the learning loop treats them as one training set — not three silos.
The underlying code (``EngagementRewardModel`` from Track A.1 plus the
three outcome-population hooks from Track A.2) is already correct per
Stage-5 audit; this file **locks in** that contract by asserting:

    (a) All 3 platforms land their ``action_outcomes`` rows under the
        same ``profile_id`` (SEC-C-05 cross-profile guard held).
    (b) ``LearningDatabase.fetch_training_examples`` sees all 3
        platforms' rows as one combined training set.
    (c) The per-platform ``tool_events`` tag is preserved so a future
        trainer can de-skew by source-platform if desired.
    (d) A retrain invoked after all 3 platforms have written sees the
        combined training set, not platform-siloed shards.
    (e) Each platform's session produces a distinct ``session_id`` —
        cross-platform writes coexist, they do not collide.
    (f) No leakage: a *different* profile's action_outcomes row is
        never surfaced by the shared profile's training-example fetch.

Test approach
-------------
- Isolated ``tmp_path`` DBs (``memory.db`` + ``learning.db``).
- Hand-bootstrapped schema mirrors production (post-M001/M006/M007)
  to the extent the ``fetch_training_examples`` join needs.
- Each platform simulates its outcome write via
  ``EngagementRewardModel.record_recall`` +
  ``EngagementRewardModel.register_signal`` +
  ``EngagementRewardModel.finalize_outcome`` — identical to what the
  real hooks do.
- No subprocess, no real daemon, no real external hooks — per manifest.

Contract references
-------------------
- LLD-00 §1.1 — action_outcomes.profile_id is NOT NULL (SEC-C-05).
- LLD-00 §1.2 — pending_outcomes on memory.db.
- LLD-00 §2  — finalize_outcome(outcome_id=...) kwarg-only.
- LLD-07 §3.1 / M001 — learning_signals.query_id column required so
  the fetch_training_examples JOIN works.
- LLD-07 §3.6 / M006 — action_outcomes.reward + settled + settled_at +
  recall_query_id columns required for the M006-enabled JOIN path.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Iterator

import pytest

from superlocalmemory.learning.database import LearningDatabase
from superlocalmemory.learning.reward import EngagementRewardModel


# ---------------------------------------------------------------------------
# Platform simulation constants
# ---------------------------------------------------------------------------

#: Three host platforms we simulate. Tag strings match what the real
#: adapters in ``src/superlocalmemory/hooks/`` self-identify as
#: (``claude_code``, ``cursor``, ``antigravity_workspace``) — stable
#: vocabulary so downstream trainers can group/de-skew by host.
_PLATFORMS: tuple[str, ...] = ("claude_code", "cursor", "antigravity_workspace")

#: Shared profile across every platform — this is the contract under test.
_SHARED_PROFILE: str = "varun_balanced"

#: A neighbouring profile used by leakage assertions.
_OTHER_PROFILE: str = "varun_other"


# ---------------------------------------------------------------------------
# Bootstrap helpers — production-shape schema in an isolated tmp_path.
# ---------------------------------------------------------------------------


def _bootstrap_memory_db(path: Path) -> None:
    """Create the memory.db tables the reward model + training join touch.

    Mirrors post-M006 action_outcomes (LLD-00 §1.1), post-M007
    pending_outcomes (LLD-00 §1.2), and tool_events from
    ``storage/schema_v347.py``. WAL is enabled to match production.
    """
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE action_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL DEFAULT 'default',
                query            TEXT NOT NULL DEFAULT '',
                fact_ids_json    TEXT NOT NULL DEFAULT '[]',
                outcome          TEXT NOT NULL DEFAULT '',
                context_json     TEXT NOT NULL DEFAULT '{}',
                timestamp        TEXT NOT NULL DEFAULT (datetime('now')),
                reward           REAL,
                settled          INTEGER NOT NULL DEFAULT 0,
                settled_at       TEXT,
                recall_query_id  TEXT
            );
            CREATE INDEX idx_outcomes_profile
                ON action_outcomes(profile_id);
            CREATE INDEX idx_outcomes_recall_query
                ON action_outcomes(recall_query_id);

            CREATE TABLE pending_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL,
                session_id       TEXT NOT NULL,
                recall_query_id  TEXT NOT NULL,
                fact_ids_json    TEXT NOT NULL,
                query_text_hash  TEXT NOT NULL,
                created_at_ms    INTEGER NOT NULL,
                expires_at_ms    INTEGER NOT NULL,
                signals_json     TEXT NOT NULL DEFAULT '{}',
                status           TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE tool_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                profile_id TEXT DEFAULT 'default',
                project_path TEXT DEFAULT '',
                tool_name TEXT NOT NULL,
                event_type TEXT NOT NULL DEFAULT 'invoke',
                input_summary TEXT DEFAULT '',
                output_summary TEXT DEFAULT '',
                duration_ms INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX idx_tool_events_session
                ON tool_events(session_id);
            """
        )


def _bootstrap_learning_db(lrn: LearningDatabase) -> None:
    """Apply the M001 column additions that ``fetch_training_examples``
    joins against. ``LearningDatabase._init_schema`` creates the base
    schema; we extend it here so the production-shape JOIN
    (``learning_features.signal_id`` + ``learning_signals.query_id``)
    resolves.

    We also seed the ``migration_log`` table and mark ``M006`` as
    ``complete`` so ``fetch_training_examples`` takes the M006-enabled
    JOIN path (the WITH_OUTCOMES SQL). That's the path where the
    cross-platform evidence flows through — the position-proxy fallback
    would mask the contract we're locking in.
    """
    conn = sqlite3.connect(lrn.path, timeout=5)
    try:
        # M001 — rich signal columns + feature signal_id + is_synthetic.
        conn.executescript(
            """
            ALTER TABLE learning_signals ADD COLUMN query_id        TEXT DEFAULT '';
            ALTER TABLE learning_signals ADD COLUMN query_text_hash TEXT DEFAULT '';
            ALTER TABLE learning_signals ADD COLUMN position        INTEGER DEFAULT 0;
            ALTER TABLE learning_signals ADD COLUMN channel_scores  TEXT DEFAULT '{}';
            ALTER TABLE learning_signals ADD COLUMN cross_encoder   REAL;
            ALTER TABLE learning_features ADD COLUMN signal_id      INTEGER DEFAULT 0;
            ALTER TABLE learning_features ADD COLUMN is_synthetic   INTEGER NOT NULL DEFAULT 0;
            """
        )
        # Minimal migration_log so fetch_training_examples's M006 gate
        # sees "applied" and uses the outcomes JOIN path.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS migration_log (
                name       TEXT PRIMARY KEY,
                status     TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT OR REPLACE INTO migration_log (name, status)
                VALUES ('M006_action_outcomes_reward', 'complete');
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_db_path(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"
    _bootstrap_memory_db(path)
    return path


@pytest.fixture()
def learning_db(tmp_path: Path) -> Iterator[LearningDatabase]:
    lrn = LearningDatabase(tmp_path / "learning.db")
    _bootstrap_learning_db(lrn)
    yield lrn


@pytest.fixture()
def reward_model(memory_db_path: Path) -> Iterator[EngagementRewardModel]:
    model = EngagementRewardModel(memory_db_path)
    try:
        yield model
    finally:
        model.close()


# ---------------------------------------------------------------------------
# Platform simulator — one function, three invocations.
# ---------------------------------------------------------------------------


def _simulate_platform_outcome(
    *,
    platform: str,
    profile_id: str,
    reward_model: EngagementRewardModel,
    learning_db: LearningDatabase,
    memory_db_path: Path,
    fact_ids: list[str],
    query_text: str,
    cited: bool,
) -> dict[str, str]:
    """Simulate one platform's full end-to-end outcome-write path.

    Mirrors what the real A.2 hooks do:
      1. ``record_recall`` -> pending_outcomes row (hot path).
      2. ``tool_events`` INSERT with the platform tag in ``metadata``
         — this is what the real post_tool_outcome_hook writes.
      3. ``register_signal`` -> ``cite`` or ``dwell_ms`` signal.
      4. ``finalize_outcome`` -> action_outcomes row with
         ``profile_id`` populated + settled + reward + recall_query_id.
      5. ``learning_db.store_signal`` + ``store_features`` with
         ``query_id = recall_query_id`` so the downstream
         ``fetch_training_examples`` JOIN catches this row.

    Returns the identifiers the test asserts against.
    """
    session_id = f"{platform}-sess-{uuid.uuid4().hex[:8]}"
    recall_query_id = f"rq-{platform}-{uuid.uuid4().hex[:8]}"

    outcome_id = reward_model.record_recall(
        profile_id=profile_id,
        session_id=session_id,
        recall_query_id=recall_query_id,
        fact_ids=fact_ids,
        query_text=query_text,
    )

    # Write a tool_events row with the platform tag in metadata — this
    # is the de-skew channel the trainer can group by.
    import json as _json
    with sqlite3.connect(memory_db_path, timeout=5) as conn:
        conn.execute(
            "INSERT INTO tool_events "
            "(session_id, profile_id, tool_name, event_type, metadata, created_at) "
            "VALUES (?, ?, ?, 'invoke', ?, datetime('now'))",
            (
                session_id,
                profile_id,
                "recall",
                _json.dumps({"platform": platform, "outcome_id": outcome_id}),
            ),
        )
        conn.commit()

    if cited:
        reward_model.register_signal(
            outcome_id=outcome_id, signal_name="cite", signal_value=True
        )
    else:
        reward_model.register_signal(
            outcome_id=outcome_id, signal_name="dwell_ms", signal_value=3000
        )

    reward = reward_model.finalize_outcome(outcome_id=outcome_id)

    # Feed the learning_db so fetch_training_examples can join. Match
    # the production shape: one "candidate" signal per fact, matching
    # query_id, with a feature row bound by signal_id.
    for position, fact_id in enumerate(fact_ids):
        signal_row_id = learning_db.store_signal(
            profile_id=profile_id,
            query=query_text,
            fact_id=fact_id,
            signal_type="candidate",
            value=1.0,
        )
        # Back-fill the M001 columns directly — the public API predates
        # M001 and doesn't surface them. This mirrors the real
        # signal_worker.py's writer path.
        with sqlite3.connect(learning_db.path, timeout=5) as lconn:
            lconn.execute(
                "UPDATE learning_signals "
                "   SET query_id = ?, position = ? "
                " WHERE id = ?",
                (recall_query_id, position, signal_row_id),
            )
            lconn.commit()
        learning_db.store_features(
            profile_id=profile_id,
            query_id=recall_query_id,
            fact_id=fact_id,
            features={"platform_tag": platform, "position": float(position)},
            label=reward,
        )
        # Bind the feature row to the signal row (M001 signal_id column).
        with sqlite3.connect(learning_db.path, timeout=5) as lconn:
            lconn.execute(
                "UPDATE learning_features "
                "   SET signal_id = ? "
                " WHERE profile_id = ? AND query_id = ? AND fact_id = ?",
                (signal_row_id, profile_id, recall_query_id, fact_id),
            )
            lconn.commit()

    return {
        "platform": platform,
        "session_id": session_id,
        "recall_query_id": recall_query_id,
        "outcome_id": outcome_id,
        "reward": reward,
    }


# ---------------------------------------------------------------------------
# Assertion (a) — all 3 platforms land under the same profile_id.
# ---------------------------------------------------------------------------


def test_three_platforms_write_to_same_profile_id(
    memory_db_path: Path,
    reward_model: EngagementRewardModel,
    learning_db: LearningDatabase,
) -> None:
    """SEC-C-05 cross-profile guard must hold across every platform.

    Each platform writes an outcome under ``_SHARED_PROFILE`` via the
    canonical ``EngagementRewardModel.finalize_outcome`` path. The
    resulting ``action_outcomes`` table must have EXACTLY 3 rows, all
    carrying ``profile_id == _SHARED_PROFILE``, and zero rows with a
    NULL / empty / mismatched profile_id.
    """
    results = [
        _simulate_platform_outcome(
            platform=p,
            profile_id=_SHARED_PROFILE,
            reward_model=reward_model,
            learning_db=learning_db,
            memory_db_path=memory_db_path,
            fact_ids=[f"fact-{p}-1", f"fact-{p}-2"],
            query_text=f"query from {p}",
            cited=True,
        )
        for p in _PLATFORMS
    ]
    assert len(results) == 3

    with sqlite3.connect(memory_db_path) as conn:
        rows = conn.execute(
            "SELECT profile_id, outcome_id FROM action_outcomes"
        ).fetchall()
        assert len(rows) == 3, "exactly one action_outcomes row per platform"
        profile_ids = {r[0] for r in rows}
        assert profile_ids == {_SHARED_PROFILE}, (
            f"SEC-C-05 breach: expected only {_SHARED_PROFILE!r}, "
            f"got {profile_ids!r}"
        )
        # Cross-check: outcome_ids are unique (no collisions).
        outcome_ids = {r[1] for r in rows}
        assert len(outcome_ids) == 3

        # No NULL / empty profile_id slipped through.
        null_count = conn.execute(
            "SELECT COUNT(*) FROM action_outcomes "
            "WHERE profile_id IS NULL OR profile_id = ''"
        ).fetchone()[0]
        assert null_count == 0


# ---------------------------------------------------------------------------
# Assertion (b) — training-set is one combined view of all platforms.
# ---------------------------------------------------------------------------


def test_learning_database_fetches_all_platform_outcomes(
    memory_db_path: Path,
    reward_model: EngagementRewardModel,
    learning_db: LearningDatabase,
) -> None:
    """``fetch_training_examples`` must surface every platform's rows.

    Each platform contributes 2 facts. ``fetch_training_examples`` for
    ``_SHARED_PROFILE`` must therefore return 6 rows (2 facts × 3
    platforms), each carrying an ``outcome_reward`` joined from
    ``action_outcomes``.

    The learning DB and memory DB are separate files in production;
    the JOIN is done by the trainer through ``recall_query_id`` (also
    known as ``query_id`` on the learning side). This test proves
    that join works across platform-origin data.
    """
    for p in _PLATFORMS:
        _simulate_platform_outcome(
            platform=p,
            profile_id=_SHARED_PROFILE,
            reward_model=reward_model,
            learning_db=learning_db,
            memory_db_path=memory_db_path,
            fact_ids=[f"fact-{p}-1", f"fact-{p}-2"],
            query_text=f"query from {p}",
            cited=True,
        )

    # Attach memory.db to the learning connection so the JOIN resolves.
    # In production the trainer copies/joins across files too; the
    # fetch_training_examples method is DB-local, so we mirror that by
    # copying action_outcomes rows into the learning.db for the JOIN.
    _mirror_action_outcomes_into_learning_db(memory_db_path, learning_db)

    rows = learning_db.fetch_training_examples(
        profile_id=_SHARED_PROFILE, limit=100, min_outcome_age_sec=0
    )
    assert len(rows) == 6, (
        f"expected 6 training rows (3 platforms × 2 facts), got {len(rows)}"
    )

    # Every row must carry a non-null outcome_reward from action_outcomes
    # — that's the gate that proves the cross-platform JOIN fired.
    rewards = [r["outcome_reward"] for r in rows]
    assert all(
        r is not None for r in rewards
    ), f"outcome_reward NULL on at least one row — JOIN failed: {rewards!r}"

    # All three platform tags should appear in the feature dicts.
    platform_tags = {r["features"].get("platform_tag") for r in rows}
    assert platform_tags == set(_PLATFORMS), (
        f"expected every platform represented; got {platform_tags!r}"
    )


# ---------------------------------------------------------------------------
# Assertion (c) — platform tag preserved in training features.
# ---------------------------------------------------------------------------


def test_platform_tag_preserved_in_training_features(
    memory_db_path: Path,
    reward_model: EngagementRewardModel,
    learning_db: LearningDatabase,
) -> None:
    """The per-platform tag must survive into the training features.

    Trainers that want to de-skew by host-platform need the tag to be
    preserved. This test writes one outcome per platform (each with
    one fact for simplicity) and verifies the resulting training rows
    each carry the correct ``platform_tag``.

    It also checks the tag is preserved on the ``tool_events`` side —
    the channel a future analytics pass would use if the feature dict
    ever stopped carrying it.
    """
    recorded: list[dict[str, str]] = []
    for p in _PLATFORMS:
        r = _simulate_platform_outcome(
            platform=p,
            profile_id=_SHARED_PROFILE,
            reward_model=reward_model,
            learning_db=learning_db,
            memory_db_path=memory_db_path,
            fact_ids=[f"f-{p}"],
            query_text=f"what did {p} do",
            cited=(p != "cursor"),  # mix labels to prove weights survive
        )
        recorded.append(r)

    _mirror_action_outcomes_into_learning_db(memory_db_path, learning_db)

    rows = learning_db.fetch_training_examples(
        profile_id=_SHARED_PROFILE, limit=100, min_outcome_age_sec=0
    )
    assert len(rows) == 3

    # Feature-side preservation: platform tag must match the source.
    tag_by_query_id = {r["query_id"]: r["features"]["platform_tag"] for r in rows}
    for rec in recorded:
        assert tag_by_query_id[rec["recall_query_id"]] == rec["platform"], (
            f"platform tag lost for query_id={rec['recall_query_id']}"
        )

    # tool_events-side preservation: the metadata blob carries the tag.
    import json as _json
    with sqlite3.connect(memory_db_path) as conn:
        events = conn.execute(
            "SELECT session_id, metadata FROM tool_events"
        ).fetchall()
    assert len(events) == 3
    tagged = {_json.loads(m)["platform"] for (_sid, m) in events}
    assert tagged == set(_PLATFORMS)


# ---------------------------------------------------------------------------
# Assertion (d) — retrain sees the combined training set.
# ---------------------------------------------------------------------------


def test_retrain_sees_combined_training_set(
    memory_db_path: Path,
    reward_model: EngagementRewardModel,
    learning_db: LearningDatabase,
) -> None:
    """A retrain invoked after all 3 platforms have written must see
    the combined training set (not platform-siloed shards).

    We model "retrain" as the trainer's canonical data fetch:
    ``fetch_training_examples`` + group-by ``query_id`` to form
    lambdarank groups. A combined fetch must produce 3 query-groups
    (one per platform) + 6 documents (2 per group) + 6 labels — i.e.
    the union, not the max-of-platform.
    """
    for p in _PLATFORMS:
        _simulate_platform_outcome(
            platform=p,
            profile_id=_SHARED_PROFILE,
            reward_model=reward_model,
            learning_db=learning_db,
            memory_db_path=memory_db_path,
            fact_ids=[f"{p}-a", f"{p}-b"],
            query_text=f"{p} retrain probe",
            cited=True,
        )
    _mirror_action_outcomes_into_learning_db(memory_db_path, learning_db)

    rows = learning_db.fetch_training_examples(
        profile_id=_SHARED_PROFILE, limit=100, min_outcome_age_sec=0
    )

    # Lambdarank groups = distinct query_ids (one per platform's recall).
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["query_id"], []).append(r)

    assert len(groups) == 3, (
        f"expected 3 retrain groups (one per platform), got {len(groups)}"
    )
    for qid, members in groups.items():
        assert len(members) == 2, (
            f"group {qid} expected 2 docs, got {len(members)}"
        )

    # Combined document count.
    assert sum(len(v) for v in groups.values()) == 6

    # Labels are all non-None — the retrain has real gradient on every
    # row, not the 0.5 fallback.
    labels = [r["outcome_reward"] for r in rows]
    assert all(isinstance(v, float) for v in labels)
    assert all(0.0 <= v <= 1.0 for v in labels)


# ---------------------------------------------------------------------------
# Assertion (e) — distinct session_ids across platforms.
# ---------------------------------------------------------------------------


def test_cross_platform_rows_have_distinct_session_ids(
    memory_db_path: Path,
    reward_model: EngagementRewardModel,
    learning_db: LearningDatabase,
) -> None:
    """Each platform's writes must carry a distinct ``session_id``.

    Same-profile does not mean same-session. Sessions are the unit of
    consolidation, rehash, and Stop-hook ownership; collapsing them
    would merge hooks' pending-outcome registries and corrupt the
    rehash/Stop windows. Test that the three platforms' writes land in
    three distinct ``session_id`` namespaces.
    """
    for p in _PLATFORMS:
        _simulate_platform_outcome(
            platform=p,
            profile_id=_SHARED_PROFILE,
            reward_model=reward_model,
            learning_db=learning_db,
            memory_db_path=memory_db_path,
            fact_ids=[f"sess-fact-{p}"],
            query_text=f"{p} session probe",
            cited=False,
        )

    with sqlite3.connect(memory_db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM pending_outcomes"
        ).fetchall()
        assert len(rows) == 3, (
            f"expected 3 distinct session_ids, got {len(rows)}"
        )

        # Every session_id is also represented in tool_events — the
        # two tables are stitched by session_id in the real system.
        te_rows = conn.execute(
            "SELECT DISTINCT session_id FROM tool_events"
        ).fetchall()
        assert {r[0] for r in te_rows} == {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Assertion (f) — no leakage: other profile's writes stay siloed.
# ---------------------------------------------------------------------------


def test_cross_platform_shared_profile_no_leakage(
    memory_db_path: Path,
    reward_model: EngagementRewardModel,
    learning_db: LearningDatabase,
) -> None:
    """Cross-platform sharing must not cross PROFILE boundaries.

    Writes done under ``_SHARED_PROFILE`` from all 3 platforms must
    remain invisible to ``_OTHER_PROFILE``'s training fetch. This is
    the SEC-C-05 complement of (a): we asserted same-profile
    aggregates; here we assert different-profile isolation holds even
    when the same reward model + learning DB are in play.
    """
    # Three platforms write under the shared profile.
    for p in _PLATFORMS:
        _simulate_platform_outcome(
            platform=p,
            profile_id=_SHARED_PROFILE,
            reward_model=reward_model,
            learning_db=learning_db,
            memory_db_path=memory_db_path,
            fact_ids=[f"shared-{p}"],
            query_text=f"{p} probe shared",
            cited=True,
        )
    # The OTHER profile writes a single outcome from one platform too.
    _simulate_platform_outcome(
        platform="claude_code",
        profile_id=_OTHER_PROFILE,
        reward_model=reward_model,
        learning_db=learning_db,
        memory_db_path=memory_db_path,
        fact_ids=["other-fact"],
        query_text="other profile probe",
        cited=True,
    )

    _mirror_action_outcomes_into_learning_db(memory_db_path, learning_db)

    shared_rows = learning_db.fetch_training_examples(
        profile_id=_SHARED_PROFILE, limit=100, min_outcome_age_sec=0
    )
    other_rows = learning_db.fetch_training_examples(
        profile_id=_OTHER_PROFILE, limit=100, min_outcome_age_sec=0
    )

    # Shared profile sees ONLY its 3 rows, not 4 (not the other's).
    assert len(shared_rows) == 3
    shared_fact_ids = {r["fact_id"] for r in shared_rows}
    assert "other-fact" not in shared_fact_ids

    # Other profile sees ONLY its 1 row.
    assert len(other_rows) == 1
    assert other_rows[0]["fact_id"] == "other-fact"

    # SEC-C-05 check at the raw action_outcomes level too.
    with sqlite3.connect(memory_db_path) as conn:
        shared_count = conn.execute(
            "SELECT COUNT(*) FROM action_outcomes WHERE profile_id = ?",
            (_SHARED_PROFILE,),
        ).fetchone()[0]
        other_count = conn.execute(
            "SELECT COUNT(*) FROM action_outcomes WHERE profile_id = ?",
            (_OTHER_PROFILE,),
        ).fetchone()[0]
    assert shared_count == 3
    assert other_count == 1


# ---------------------------------------------------------------------------
# Helper — mirror memory.db's action_outcomes into learning.db.
#
# In production, memory.db and learning.db are separate files and the
# trainer bridges them by attaching or by out-of-DB lookups. The
# ``fetch_training_examples`` SQL does a LEFT JOIN on a same-DB
# ``action_outcomes`` table, so for test realism we mirror the settled
# rows over before fetching. The mirrored copy carries the same
# profile_id + recall_query_id + reward + settled_at so the JOIN is
# identical to production. This is test plumbing only — no production
# code path copies these rows.
# ---------------------------------------------------------------------------


def _mirror_action_outcomes_into_learning_db(
    memory_db_path: Path, learning_db: LearningDatabase
) -> None:
    with sqlite3.connect(memory_db_path) as mconn:
        rows = mconn.execute(
            "SELECT outcome_id, profile_id, reward, settled, settled_at, "
            "       recall_query_id "
            "FROM action_outcomes WHERE settled = 1"
        ).fetchall()

    lconn = sqlite3.connect(learning_db.path, timeout=5)
    try:
        lconn.executescript(
            """
            CREATE TABLE IF NOT EXISTS action_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL DEFAULT 'default',
                reward           REAL,
                settled          INTEGER NOT NULL DEFAULT 0,
                settled_at       TEXT,
                recall_query_id  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ao_recall_q
                ON action_outcomes(recall_query_id);
            """
        )
        lconn.executemany(
            "INSERT OR REPLACE INTO action_outcomes "
            "(outcome_id, profile_id, reward, settled, settled_at, recall_query_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        lconn.commit()
    finally:
        lconn.close()
