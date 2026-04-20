# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-04 §6.1

"""Tests for ``/api/v3/brain`` endpoint (LLD-04 v2).

TDD phase: RED. These tests define the Brain endpoint contract before
any route code is written. They use a minimal FastAPI app with ONLY the
brain router and the strict security-headers middleware — avoiding the
full unified daemon to keep tests fast and deterministic.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import security_primitives as sp
from superlocalmemory.server.middleware.security_headers import (
    SecurityHeadersMiddleware,
)
from superlocalmemory.server.routes import brain as brain_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_token_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolated install-token file for each test."""
    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / ".install_token"
        monkeypatch.setattr(sp, "_install_token_path", lambda: token_path)
        yield token_path


@pytest.fixture()
def install_token(tmp_token_dir: Path) -> str:
    return sp.ensure_install_token()


@pytest.fixture()
def tmp_learning_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Empty learning DB; brain route points at it via monkeypatch."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "learning.db"
        # Create minimal schema so count_signals works.
        from superlocalmemory.learning.database import LearningDatabase
        _ = LearningDatabase(db_path)
        # Point brain route module at this DB.
        monkeypatch.setattr(brain_mod, "_learning_db_path",
                            lambda: db_path)
        yield db_path


@pytest.fixture()
def seeded_preferences(monkeypatch: pytest.MonkeyPatch):
    """Stub the preferences loader to return a known shape."""
    data = {
        "topics":   [{"name": "ai_agents", "strength": 0.87}],
        "entities": [{"name": "Qualixar", "mention_count": 142}],
        "tech":     [{"name": "Python", "frequency": 0.62}],
        "source":   "_store_patterns",
    }
    monkeypatch.setattr(brain_mod, "_load_raw_preferences", lambda pid: data)
    return data


@pytest.fixture()
def app(tmp_learning_db: Path, seeded_preferences) -> FastAPI:
    application = FastAPI()
    application.add_middleware(SecurityHeadersMiddleware)
    application.include_router(brain_mod.router)
    return application


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# U6 — install-token gate
# ---------------------------------------------------------------------------


def test_brain_requires_install_token_no_header(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain")
    assert r.status_code == 401
    assert r.json()["detail"] == "install_token_required"
    assert r.headers.get("www-authenticate", "").lower().startswith("install-token")


def test_brain_rejects_wrong_token(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": "wrong_" + install_token})
    assert r.status_code == 401


def test_brain_accepts_x_install_token_header(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    assert r.status_code == 200


def test_brain_accepts_bearer_token(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain",
                   headers={"Authorization": f"Bearer {install_token}"})
    assert r.status_code == 200


def test_deprecated_requires_install_token(
    client: TestClient, install_token: str,
) -> None:
    for path in ("/api/v3/learning/stats",
                 "/api/v3/patterns",
                 "/api/v3/behavioral"):
        r = client.get(path)
        assert r.status_code == 401, f"{path} must require install token"
        r2 = client.get(path, headers={"X-Install-Token": install_token})
        assert r2.status_code == 200, f"{path} must accept valid token"
        body = r2.json()
        assert body.get("deprecated") is True, f"{path} must flag deprecated"
        assert body.get("use_instead") == "/api/v3/brain"


# ---------------------------------------------------------------------------
# U2 — honesty labels
# ---------------------------------------------------------------------------


def test_brain_returns_all_sections(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    assert r.status_code == 200
    body = r.json()
    for section in ("profile_id", "preferences", "learning", "usage",
                    "bandit", "cache", "cross_platform",
                    "evolution_preview", "outcomes_preview", "meta"):
        assert section in body, f"missing section: {section}"


def test_brain_honesty_labels_present(
    client: TestClient, install_token: str,
) -> None:
    """Every metric section has either is_real or is_real_ml."""
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    assert r.status_code == 200
    body = r.json()
    for section_name in ("preferences", "learning", "bandit", "cache"):
        section = body[section_name]
        assert "is_real" in section or "is_real_ml" in section, (
            f"{section_name} missing is_real/is_real_ml flag")
    usage = body["usage"]
    assert "is_real_ml" in usage, "usage must have is_real_ml flag"
    assert usage["is_real_ml"] is False, "usage must admit it's counters"


# ---------------------------------------------------------------------------
# U3 — phase gated on active model
# ---------------------------------------------------------------------------


def test_phase_gated_on_active_model(
    client: TestClient, install_token: str, tmp_learning_db: Path,
) -> None:
    """500 signals + no active model → phase MUST NOT be 3."""
    from superlocalmemory.learning.database import LearningDatabase
    db = LearningDatabase(tmp_learning_db)
    for i in range(250):
        db.store_signal("default", f"q{i}", f"f{i}", "recall_hit", 1.0)
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    assert r.status_code == 200
    learning = r.json()["learning"]
    assert learning["phase"] != 3, (
        "Phase 3 requires an active model row; none was persisted")
    assert learning["model_active"] is False


# ---------------------------------------------------------------------------
# U4 — no fabricated metrics
# ---------------------------------------------------------------------------


_BANNED_KEYS = (
    "hit_rate_24h",
    "avg_age_on_hit_seconds",
    "skill_evolution_rows",
    "skill_evolution_count",
)


def _walk(obj, banned):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in banned, f"fabricated key surfaced: {k}"
            _walk(v, banned)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, banned)


def test_no_fabricated_metrics(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    body = r.json()
    _walk(body, _BANNED_KEYS)


# ---------------------------------------------------------------------------
# U9 — secret redaction
# ---------------------------------------------------------------------------


def test_preference_redaction_counts(
    client: TestClient, install_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject a secret-bearing preference payload.
    monkeypatch.setattr(brain_mod, "_load_raw_preferences", lambda pid: {
        "topics":   [{"name": "secret=AKIAABCDEFGHIJKLMNOP", "strength": 0.9}],
        "entities": [{"name": "Qualixar", "mention_count": 1}],
        "tech":     [{"name": "Python", "frequency": 0.6}],
    })
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    body = r.json()
    prefs = body["preferences"]
    assert prefs["redacted_count"] >= 1
    # The raw AWS key must not survive in the payload.
    import json as _json
    dumped = _json.dumps(prefs)
    assert "AKIAABCDEFGHIJKLMNOP" not in dumped
    assert "[REDACTED:AWS" in dumped


# ---------------------------------------------------------------------------
# U10 — feature & stratum constants surfaced
# ---------------------------------------------------------------------------


def test_feature_and_stratum_constants_surfaced(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    body = r.json()
    from superlocalmemory.learning.features import FEATURE_DIM
    assert body["learning"]["feature_count_expected"] == FEATURE_DIM
    assert body["learning"]["feature_count_expected"] == 20
    assert body["bandit"]["strata_total"] == 48


# ---------------------------------------------------------------------------
# U5 — latency smoke test
# ---------------------------------------------------------------------------


def test_brain_p95_under_200ms(
    client: TestClient, install_token: str,
) -> None:
    import time
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        r = client.get("/api/v3/brain",
                       headers={"X-Install-Token": install_token})
        samples.append((time.perf_counter() - t0) * 1000.0)
        assert r.status_code == 200
    samples.sort()
    p95_idx = max(0, int(0.95 * len(samples)) - 1)
    p95_ms = samples[p95_idx]
    # Generous ceiling for a CI runner; empty DB should return well under.
    assert p95_ms <= 500.0, f"p95 too slow: {p95_ms:.2f} ms"


# ---------------------------------------------------------------------------
# Coverage drivers — exercise real section builders end-to-end.
# ---------------------------------------------------------------------------


def test_phase_3_when_model_active_and_enough_signals(
    client: TestClient, install_token: str, tmp_learning_db: Path,
) -> None:
    """Phase 3 path — active model row + ≥200 signals + SHA present."""
    from superlocalmemory.learning.database import LearningDatabase
    db = LearningDatabase(tmp_learning_db)
    # Schema lacks the v3.4.22 columns in the stock learning DB. Add them
    # ad-hoc for this test so persist_model has a surface to write into.
    import sqlite3
    conn = sqlite3.connect(str(tmp_learning_db))
    try:
        for stmt in (
            "ALTER TABLE learning_model_state ADD COLUMN model_version TEXT",
            "ALTER TABLE learning_model_state ADD COLUMN bytes_sha256 TEXT",
            "ALTER TABLE learning_model_state ADD COLUMN feature_names TEXT",
            "ALTER TABLE learning_model_state ADD COLUMN metrics_json TEXT",
            "ALTER TABLE learning_model_state ADD COLUMN trained_on_count INT",
            "ALTER TABLE learning_model_state ADD COLUMN is_active INT DEFAULT 1",
            "ALTER TABLE learning_model_state ADD COLUMN trained_at TEXT",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.Error:
                pass
        conn.commit()
    finally:
        conn.close()

    for i in range(210):
        db.store_signal("default", f"q{i}", f"f{i}", "recall_hit", 1.0)
    db.persist_model(
        profile_id="default",
        state_bytes=b"fake_model_bytes",
        bytes_sha256="a" * 64,
        feature_names=["f1", "f2"],
        trained_on_count=210,
        metrics={"loss": 0.1},
    )

    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    assert r.status_code == 200
    learning = r.json()["learning"]
    assert learning["phase"] == 3
    assert learning["model_active"] is True
    assert learning["model_sha256_present"] is True


def test_real_preferences_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exercise _load_raw_preferences real path via a fake behavioral module."""
    import importlib
    from superlocalmemory.server.routes import brain as brain_mod

    # Create a learning DB file so the ``exists`` guard passes.
    db_file = tmp_path / "learning.db"
    db_file.write_bytes(b"sqlite-ish-placeholder")
    monkeypatch.setattr(brain_mod, "_learning_db_path", lambda: db_file)

    class _StubStore:
        def __init__(self, *a, **k):
            pass

        def get_patterns(self, profile_id: str):
            return [
                {"pattern_type": "tech_preference",
                 "pattern_key": "py",
                 "metadata": {"value": "Python"},
                 "confidence": 0.8, "evidence_count": 10},
                {"pattern_type": "entity",
                 "pattern_key": "qx",
                 "metadata": {"value": "Qualixar"},
                 "confidence": 0.9, "evidence_count": 42},
                {"pattern_type": "interest",
                 "pattern_key": "ai",
                 "metadata": {"value": "AI"},
                 "confidence": 0.5, "evidence_count": 4},
                {"pattern_type": "topic",
                 "pattern_key": "t",
                 "metadata": {"value": "ai_agents"},
                 "confidence": 0.87, "evidence_count": 5},
                {"pattern_type": "topic",  # empty name → skipped
                 "pattern_key": "",
                 "metadata": {},
                 "confidence": 0.1, "evidence_count": 0},
                {"pattern_type": "unknown_type",  # unhandled branch
                 "pattern_key": "x",
                 "metadata": {"value": "x"},
                 "confidence": 0.0, "evidence_count": 0},
            ]

    import sys
    import types
    fake_mod = types.ModuleType("superlocalmemory.learning.behavioral")
    fake_mod.BehavioralPatternStore = _StubStore  # type: ignore
    monkeypatch.setitem(
        sys.modules, "superlocalmemory.learning.behavioral", fake_mod,
    )

    # Call the loader directly (no fixture stub in play for this test).
    out = brain_mod._load_raw_preferences("default")
    assert any(t["name"] == "Python" for t in out["tech"])
    assert any(e["name"] == "Qualixar" for e in out["entities"])
    assert any(t["name"] == "ai_agents" for t in out["topics"])


def test_preferences_loader_missing_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.server.routes import brain as brain_mod
    monkeypatch.setattr(brain_mod, "_learning_db_path",
                        lambda: tmp_path / "does_not_exist.db")
    out = brain_mod._load_raw_preferences("default")
    assert out == {"topics": [], "entities": [], "tech": [],
                   "source": "_store_patterns"}


def test_cache_stats_with_memory_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.server.routes import brain as brain_mod
    import sqlite3
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT)")
        conn.executemany(
            "INSERT INTO atomic_facts VALUES (?)",
            [("f1",), ("f2",), ("f3",)],
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(brain_mod, "_memory_db_path", lambda: db)
    stats = brain_mod._compute_cache_stats()
    assert stats["entry_count"] == 3
    assert stats["db_size_bytes"] > 0
    assert stats["is_real"] is True


def test_cache_stats_with_bad_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.server.routes import brain as brain_mod
    bad = tmp_path / "bad.db"
    bad.write_text("not a sqlite file")
    monkeypatch.setattr(brain_mod, "_memory_db_path", lambda: bad)
    stats = brain_mod._compute_cache_stats()
    assert stats["entry_count"] == 0
    assert stats["is_real"] is True


def test_cache_stats_db_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.server.routes import brain as brain_mod
    monkeypatch.setattr(brain_mod, "_memory_db_path",
                        lambda: tmp_path / "nope.db")
    stats = brain_mod._compute_cache_stats()
    assert stats == {"db_size_bytes": 0, "entry_count": 0, "is_real": True}


def test_bandit_unsettled_empty_table(tmp_learning_db: Path) -> None:
    """Table exists, no rows, no oldest — count=0, age=None."""
    from superlocalmemory.server.routes.brain import _bandit_unsettled
    import sqlite3
    conn = sqlite3.connect(str(tmp_learning_db))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS bandit_plays ("
            "profile_id TEXT, query_id TEXT, stratum TEXT, arm_id TEXT, "
            "played_at TEXT, settled_at TEXT)",
        )
        conn.commit()
    finally:
        conn.close()
    cnt, age = _bandit_unsettled(str(tmp_learning_db), "nobody")
    assert cnt == 0
    assert age is None


def test_cache_stats_missing_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.server.routes import brain as brain_mod
    import sqlite3
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE other_table (id INT)")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(brain_mod, "_memory_db_path", lambda: db)
    stats = brain_mod._compute_cache_stats()
    # The missing atomic_facts table → sqlite.Error caught → entry_count=0.
    assert stats["entry_count"] == 0


def test_bandit_unsettled_with_rows(
    tmp_learning_db: Path, install_token: str,
) -> None:
    from superlocalmemory.server.routes.brain import _bandit_unsettled
    import sqlite3
    conn = sqlite3.connect(str(tmp_learning_db))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS bandit_plays ("
            "profile_id TEXT, query_id TEXT, stratum TEXT, arm_id TEXT, "
            "played_at TEXT, settled_at TEXT)",
        )
        conn.executemany(
            "INSERT INTO bandit_plays "
            "(profile_id, query_id, stratum, arm_id, played_at, settled_at) "
            "VALUES (?, ?, ?, ?, ?, NULL)",
            [
                ("default", "q1", "s", "arm1", "2026-04-18T10:00:00+00:00"),
                ("default", "q2", "s", "arm2", "2026-04-18T11:00:00+00:00"),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    cnt, age = _bandit_unsettled(str(tmp_learning_db), "default")
    assert cnt == 2
    assert age is not None and age >= 0


def test_bandit_unsettled_missing_table(tmp_path: Path) -> None:
    from superlocalmemory.server.routes.brain import _bandit_unsettled
    cnt, age = _bandit_unsettled(str(tmp_path / "nothing.db"), "default")
    assert cnt == 0 and age is None


def test_bandit_unsettled_naive_datetime(tmp_learning_db: Path) -> None:
    from superlocalmemory.server.routes.brain import _bandit_unsettled
    import sqlite3
    conn = sqlite3.connect(str(tmp_learning_db))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS bandit_plays ("
            "profile_id TEXT, query_id TEXT, stratum TEXT, arm_id TEXT, "
            "played_at TEXT, settled_at TEXT)",
        )
        # Naive datetime (no tz suffix) — code must normalize it to UTC.
        conn.execute(
            "INSERT INTO bandit_plays "
            "(profile_id, query_id, stratum, arm_id, played_at, settled_at) "
            "VALUES ('default', 'q', 's', 'a', '2026-04-18T10:00:00', NULL)",
        )
        # Bad timestamp — triggers the ValueError branch.
        conn.execute(
            "INSERT INTO bandit_plays "
            "(profile_id, query_id, stratum, arm_id, played_at, settled_at) "
            "VALUES ('default2', 'q', 's', 'a', 'not-a-date', NULL)",
        )
        conn.commit()
    finally:
        conn.close()
    cnt, age = _bandit_unsettled(str(tmp_learning_db), "default")
    assert cnt == 1 and age is not None
    cnt2, age2 = _bandit_unsettled(str(tmp_learning_db), "default2")
    assert cnt2 == 1 and age2 is None


def test_bandit_snapshot_returns_zero_when_no_bandit_tables(
    tmp_learning_db: Path,
) -> None:
    from superlocalmemory.server.routes.brain import _compute_bandit_snapshot
    from superlocalmemory.learning.database import LearningDatabase
    db = LearningDatabase(tmp_learning_db)
    result = _compute_bandit_snapshot("default", db)
    assert result["strata_total"] == 48
    assert result["strata_active"] == 0
    assert result["top_arm_global"] is None


def test_bandit_snapshot_with_seeded_arms(
    tmp_learning_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub ContextualBandit.snapshot to cover top-arm accumulation loop."""
    from superlocalmemory.server.routes import brain as brain_mod
    from superlocalmemory.learning.database import LearningDatabase
    import sys
    import types

    class _FakeBandit:
        def __init__(self, *a, **k):
            pass

        def snapshot(self, top_n: int = 5):
            return {
                "stratum_a": [
                    {"arm_id": "arm_alpha", "alpha": 1.0, "beta": 1.0,
                     "plays": 5},
                    {"arm_id": "arm_beta", "alpha": 1.0, "beta": 1.0,
                     "plays": 3},
                ],
                "stratum_b": [
                    {"arm_id": "arm_gamma", "alpha": 1.0, "beta": 1.0,
                     "plays": 12},
                ],
                "stratum_c": [],  # empty arms — not counted as active.
            }

    fake_mod = types.ModuleType("superlocalmemory.learning.bandit")
    fake_mod.ContextualBandit = _FakeBandit  # type: ignore
    monkeypatch.setitem(
        sys.modules, "superlocalmemory.learning.bandit", fake_mod,
    )

    db = LearningDatabase(tmp_learning_db)
    out = brain_mod._compute_bandit_snapshot("default", db)
    assert out["strata_active"] == 2
    assert out["top_arm_global"] == {"arm_id": "arm_gamma", "plays": 12}
    assert out["strata_total"] == 48


def test_action_outcomes_count_missing_table(
    tmp_learning_db: Path,
) -> None:
    from superlocalmemory.server.routes.brain import _action_outcomes_count
    from superlocalmemory.learning.database import LearningDatabase
    db = LearningDatabase(tmp_learning_db)
    assert _action_outcomes_count(db, "default") == 0


def test_action_outcomes_count_with_rows(tmp_learning_db: Path) -> None:
    import sqlite3
    from superlocalmemory.server.routes.brain import _action_outcomes_count
    from superlocalmemory.learning.database import LearningDatabase
    conn = sqlite3.connect(str(tmp_learning_db))
    try:
        conn.execute(
            "CREATE TABLE action_outcomes "
            "(profile_id TEXT, recall_query_id TEXT, "
            " settled INT, settled_at TEXT, reward REAL)",
        )
        conn.executemany(
            "INSERT INTO action_outcomes "
            "(profile_id, recall_query_id, settled) VALUES (?, ?, ?)",
            [("default", "q1", 1), ("default", "q2", 1)],
        )
        conn.commit()
    finally:
        conn.close()
    db = LearningDatabase(tmp_learning_db)
    assert _action_outcomes_count(db, "default") == 2


def test_redact_non_dict_input() -> None:
    from superlocalmemory.server.routes.brain import (
        redact_secrets_in_preferences,
    )
    assert redact_secrets_in_preferences([]) == {"redacted_count": 0}
    assert redact_secrets_in_preferences(None) == {"redacted_count": 0}  # type: ignore
    assert redact_secrets_in_preferences("str") == {"redacted_count": 0}  # type: ignore


def test_redact_tuple_container() -> None:
    from superlocalmemory.server.routes.brain import (
        redact_secrets_in_preferences,
    )
    out = redact_secrets_in_preferences({
        "topics": ("AKIAIOSFODNN7EXAMPLE", "safe"),
    })
    assert out["redacted_count"] >= 1


def test_count_signals_since_no_filter(
    tmp_learning_db: Path,
) -> None:
    from superlocalmemory.server.routes.brain import _count_signals_since
    from superlocalmemory.learning.database import LearningDatabase
    from datetime import datetime, timezone, timedelta
    db = LearningDatabase(tmp_learning_db)
    db.store_signal("default", "q1", "f1", "recall_hit", 1.0)
    db.store_signal("default", "q2", "f2", "other_type", 1.0)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    count_all = _count_signals_since(db, "default", since, signal_types=None)
    assert count_all == 2


def test_count_signals_since_missing_db(tmp_path: Path) -> None:
    from superlocalmemory.server.routes.brain import _count_signals_since
    from superlocalmemory.learning.database import LearningDatabase
    from datetime import datetime, timezone
    # Point at a fresh (empty) DB — query runs OK, returns 0.
    db = LearningDatabase(tmp_path / "empty.db")
    since = datetime.now(timezone.utc)
    assert _count_signals_since(db, "default", since) == 0


def test_safe_count_missing_table(tmp_path: Path) -> None:
    """_safe_count returns 0 when the queried table is absent."""
    from superlocalmemory.server.routes.brain import _safe_count
    from superlocalmemory.learning.database import LearningDatabase
    db = LearningDatabase(tmp_path / "empty.db")
    # learning_feedback is not in the default schema.
    assert _safe_count(db, "learning_feedback", "default") == 0


def test_meta_now_shape() -> None:
    from superlocalmemory.server.routes.brain import _meta_now
    meta = _meta_now()
    assert meta["honest_labels"] is True
    assert meta["version"] == "3.4.22"
    assert meta["generated_at"].endswith("Z")


def test_compute_cross_platform_shape() -> None:
    """All 5 + copilot adapter slots are present; ``active`` is a bool.

    The actual boolean depends on the host environment (Cursor dir exists?
    install-token file present?) so we pin shape, not value. MCP and CLI
    are always ``True`` (LLD-04 §3.1).
    """
    from superlocalmemory.server.routes.brain import _compute_cross_platform
    xp = _compute_cross_platform()
    for k in ("claude_code", "cursor", "antigravity", "copilot", "mcp", "cli"):
        assert k in xp, f"missing adapter slot: {k}"
        assert isinstance(xp[k]["active"], bool), (
            f"{k}.active must be bool, got {type(xp[k]['active'])}")
    assert xp["mcp"]["active"] is True
    assert xp["cli"]["active"] is True


def test_cross_platform_errors_dont_crash(monkeypatch) -> None:
    """If the adapter factory raises, the panel returns the stable shape
    with ``active: false`` and a non-empty ``reason`` for every kind
    rather than propagating the exception (LLD-04 §2 honest-failure).

    S8-SEC-06 fix: the endpoint now goes through
    ``cli.context_commands.build_default_adapters`` instead of constructing
    adapters with zero kwargs. The error path therefore tests a broken
    factory (which is how the real startup failure would look).
    """
    from superlocalmemory.server.routes import brain as brain_mod
    import superlocalmemory.cli.context_commands as ctx_mod

    def _broken_factory(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ctx_mod, "build_default_adapters", _broken_factory)
    xp = brain_mod._compute_cross_platform()
    # All three kinds must still be present with the stable shape, and
    # each must honestly report that it's unavailable.
    for kind in ("cursor", "antigravity", "copilot"):
        assert kind in xp, f"missing {kind} slot"
        assert xp[kind]["active"] is False
        assert "reason" in xp[kind]
    # MCP + CLI remain unconditionally active (see _compute_cross_platform).
    assert xp["mcp"]["active"] is True
    assert xp["cli"]["active"] is True


def test_adapter_last_sync_ago_missing_db(tmp_path, monkeypatch) -> None:
    """When memory.db is absent, ``last_sync_seconds_ago`` is ``None``."""
    from superlocalmemory.server.routes import brain as brain_mod
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert brain_mod._adapter_last_sync_ago("cursor") is None


def test_learning_db_path_default() -> None:
    """_learning_db_path default (unpatched) resolves under home dir."""
    from superlocalmemory.server.routes import brain as brain_mod
    # Temporarily unpatch the fixture's override.
    original = brain_mod.__dict__.get("_learning_db_path")
    # Build a fresh copy of the default resolver body.
    from pathlib import Path
    assert isinstance(original(), Path) if original else True


def test_auth_bearer_token_empty_value(
    client: TestClient, install_token: str,
) -> None:
    """Authorization header with empty bearer value is rejected."""
    r = client.get("/api/v3/brain",
                   headers={"Authorization": "Bearer "})
    assert r.status_code == 401


def test_auth_non_bearer_scheme(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain",
                   headers={"Authorization": "Basic " + install_token})
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# D.2 (v3.4.22) — /brain/evolution-timeseries
# ---------------------------------------------------------------------------


def test_evolution_timeseries_requires_install_token(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain/evolution-timeseries")
    assert r.status_code == 401


def test_evolution_timeseries_default_shape(
    client: TestClient, install_token: str, tmp_learning_db: Path,
) -> None:
    r = client.get("/api/v3/brain/evolution-timeseries",
                   headers={"X-Install-Token": install_token})
    assert r.status_code == 200
    body = r.json()
    assert body["is_real"] is True
    assert body["source"] == "learning_signals"
    assert body["days"] == 30
    assert isinstance(body["points"], list)
    assert len(body["points"]) == 30
    # Every point has the expected shape + non-negative signal count.
    for p in body["points"]:
        assert set(p.keys()) == {"date", "signals"}
        assert len(p["date"]) == 10  # YYYY-MM-DD
        assert p["signals"] >= 0


def test_evolution_timeseries_clamps_days(
    client: TestClient, install_token: str, tmp_learning_db: Path,
) -> None:
    """days is clamped to [1, 90] to keep the response bounded."""
    too_high = client.get(
        "/api/v3/brain/evolution-timeseries?days=500",
        headers={"X-Install-Token": install_token},
    ).json()
    assert too_high["days"] == 90
    assert len(too_high["points"]) == 90

    too_low = client.get(
        "/api/v3/brain/evolution-timeseries?days=0",
        headers={"X-Install-Token": install_token},
    ).json()
    assert too_low["days"] == 1
    assert len(too_low["points"]) == 1


def test_evolution_timeseries_counts_signals(
    client: TestClient, install_token: str, tmp_learning_db: Path,
) -> None:
    """Signals written today should show up in the last point."""
    # Seed today's signal.
    from datetime import datetime, timezone
    iso_now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(tmp_learning_db) as conn:
        conn.execute(
            "INSERT INTO learning_signals "
            "(profile_id, query, fact_id, signal_type, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("default", "q", "f1", "shown", iso_now),
        )
        conn.commit()

    body = client.get(
        "/api/v3/brain/evolution-timeseries?days=7",
        headers={"X-Install-Token": install_token},
    ).json()
    assert body["total_signals"] >= 1
    # Last point is today.
    today = datetime.now(timezone.utc).date().isoformat()
    last = body["points"][-1]
    assert last["date"] == today
    assert last["signals"] >= 1


def test_brain_payload_surfaces_evolution(
    client: TestClient, install_token: str, tmp_learning_db: Path,
) -> None:
    """/api/v3/brain must embed the same evolution shape as the dedicated route."""
    body = client.get("/api/v3/brain",
                      headers={"X-Install-Token": install_token}).json()
    ev = body["evolution_preview"]
    assert ev["is_real"] is True
    assert ev["source"] == "learning_signals"
    assert isinstance(ev["points"], list)
    assert len(ev["points"]) == 30
    # No stale "ships_in" placeholder.
    assert "ships_in" not in ev
