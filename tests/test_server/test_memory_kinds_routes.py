# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The memory-kinds HTTP surface the dashboard and the CLI share.

Permissions are the product's RBAC, enforced for real (a real ``RbacEngine``
with real users, roles and sessions): reading is READ, changing a fact's kind
is WRITE on a fact the active profile owns, and starting, pausing, cancelling
or undoing a classification run — or changing the settings — is MANAGE. The
machine credential gate has its own suites and is stubbed here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.access.rbac import RbacEngine
from superlocalmemory.core.hooks import HookRegistry
from superlocalmemory.core.memory_kind_backfill import BackfillRunner
from superlocalmemory.core.memory_kind_config import MemoryKindConfig, memory_kind_config_from
from superlocalmemory.encoding.memory_kind_recipe import KindAnswer
from superlocalmemory.server import write_identity
from superlocalmemory.server.routes import memory_kinds as routes
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord, Mode


class _Runtime:
    """The canonical mutation writer, applying the real SET_FACT_KIND handler."""

    ready = True

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.calls: list = []

    def set_fact_kinds(self, profile_id, items, *, idempotency_key=None, promote_as=None):
        from superlocalmemory.core.remember_runtime import _set_fact_kinds
        from superlocalmemory.storage.memory_kinds import parse_kind

        pairs = [[f, parse_kind(k).value] for f, k in items]
        self.calls.append((profile_id, pairs))
        with self.db.raw_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            payload = {"items": pairs}
            if promote_as:
                payload["promote_by"], payload["promote_channel"] = promote_as
            return _set_fact_kinds(self.db, profile_id, payload, connection=conn)


class _Judge:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.ready = True
        self.calls = 0

    def ask_kinds(self, documents, recipe, verify, consent=None):
        self.calls += 1
        return [KindAnswer("semantic", {"semantic": 0.9}, 0.9, None) for _ in documents]


def _store(path: Path, *, with_kinds: bool = True) -> DatabaseManager:
    from superlocalmemory.storage.migrations import M024_rbac_users_roles as m024
    from superlocalmemory.storage.migrations import M052_memory_kinds as m052

    db = DatabaseManager(path)
    db.initialize(real_schema)
    conn = sqlite3.connect(str(path))
    try:
        m024.apply(conn)
        if not with_kinds:
            conn.execute("DROP INDEX IF EXISTS idx_facts_memory_kind")
            for col in ("memory_kind", "memory_kind_source", "memory_kind_confidence",
                        "memory_kind_recipe", "memory_kind_at"):
                conn.execute(f"ALTER TABLE atomic_facts DROP COLUMN {col}")
        conn.commit()
    finally:
        conn.close()
    db = DatabaseManager(path)
    if with_kinds:
        with db.raw_connection() as c:
            m052.apply(c)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')")
    return db


def _fact(db, content, *, profile_id="default", kind=None, source=None, conf=None,
          scope="personal", shared_with=None) -> str:
    mid = db.store_memory(MemoryRecord(profile_id=profile_id, content="s"))
    f = AtomicFact(profile_id=profile_id, memory_id=mid, content=content,
                   fact_type=FactType.SEMANTIC, scope=scope, shared_with=shared_with)
    if db.has_memory_kind_columns():
        f.memory_kind, f.memory_kind_source, f.memory_kind_confidence = kind, source, conf
    return db.store_fact(f)


def _client(tmp_path: Path, monkeypatch, *, with_kinds: bool = True, judge=None,
            cfg: MemoryKindConfig | None = None):
    monkeypatch.setattr(write_identity, "require_write_actor",
                        lambda request, descriptor, actor_kind="dashboard": "test-actor")
    db = _store(tmp_path / "memory.db", with_kinds=with_kinds)
    config = SimpleNamespace(mode=Mode.A, memory_kinds=cfg or MemoryKindConfig(),
                             base_dir=tmp_path, db_path=tmp_path / "memory.db",
                             active_profile="default")
    engine = SimpleNamespace(
        db=db, _db=db, profile_id="default", _profile_id="default", _hooks=HookRegistry(),
        _config=config, _llm=None,
        _retrieval_engine=SimpleNamespace(_sufficiency_judge=judge),
    )
    from superlocalmemory.core.memory_kind_wiring import build_kind_classifier
    engine._kind_classifier = build_kind_classifier(engine)
    app = FastAPI()
    routes.register(app)
    app.state.engine = engine
    app.state.config = config
    app.state.rbac = RbacEngine(str(tmp_path / "memory.db"))
    app.state.canonical_remember_runtime = _Runtime(db)
    app.state.memory_kind_backfill = BackfillRunner(engine_supplier=lambda: app.state.engine)
    return TestClient(app), app, db


def _session(app, username: str, role: str, profile: str = "default") -> dict[str, str]:
    rbac: RbacEngine = app.state.rbac
    user = rbac.create_user(username, "password-1234")
    rbac.set_membership(profile, user["user_id"], role)
    return {"X-SLM-User-Session": rbac.create_session(user["user_id"])}


def test_status_needs_only_read(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _fact(db, "Never push to main.")
    viewer = _session(app, "vera", "viewer")
    r = tc.get("/api/memory-kinds/status", headers=viewer)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_ready"] is True
    assert set(body) >= {"enabled", "backend", "calibration", "counts", "active_run",
                         "recent_runs"}
    app.state.rbac.set_require_login(True)
    assert tc.get("/api/memory-kinds/status").status_code == 401


def test_backfill_needs_manage(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _fact(db, "Never push to main.")
    viewer = _session(app, "vera", "viewer")
    member = _session(app, "mo", "member")
    body = {"mode": "untyped"}
    assert tc.post("/api/memory-kinds/backfill", json=body, headers=viewer).status_code == 403
    assert tc.post("/api/memory-kinds/backfill", json=body, headers=member).status_code == 403
    assert tc.post("/api/memory-kinds/settings", json={"enabled": False},
                   headers=member).status_code == 403
    r = tc.post("/api/memory-kinds/backfill", json=body)          # the machine owner
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["status"] == "queued" and run["profile_id"] == "default"
    again = tc.post("/api/memory-kinds/backfill", json=body)
    assert again.status_code == 409
    action = tc.post(f"/api/memory-kinds/backfill/{run['run_id']}/pause", headers=member)
    assert action.status_code == 403


def test_run_actions_follow_the_lifecycle(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    for t in ("Never push to main.", "We decided to use SQLite."):
        _fact(db, t)
    run = tc.post("/api/memory-kinds/backfill", json={"mode": "untyped"}).json()
    rid = run["run_id"]
    assert tc.post(f"/api/memory-kinds/backfill/{rid}/revert").status_code == 409
    assert tc.post(f"/api/memory-kinds/backfill/{rid}/pause").json()["status"] == "paused"
    assert tc.post(f"/api/memory-kinds/backfill/{rid}/resume").json()["status"] == "queued"
    runner = app.state.memory_kind_backfill
    while runner.run_once().action != "idle":
        pass
    assert tc.post(f"/api/memory-kinds/backfill/{rid}/revert").json()["status"] == "reverting"
    assert tc.post(f"/api/memory-kinds/backfill/{rid}/explode").status_code == 422
    assert tc.post("/api/memory-kinds/backfill/nope/pause").status_code == 404


def test_a_run_of_another_profile_is_not_found(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _fact(db, "work fact", profile_id="work")
    run = app.state.memory_kind_backfill.create_run("work", mode="untyped", requested_by="t")
    r = tc.post(f"/api/memory-kinds/backfill/{run['run_id']}/cancel")
    assert r.status_code == 404


def test_fact_edit_needs_write_and_owner(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    mine = _fact(db, "Alice likes green tea.")
    theirs = _fact(db, "work's shared fact", profile_id="work", scope="shared",
                   shared_with=["default"])
    viewer = _session(app, "vera", "viewer")
    member = _session(app, "mo", "member")
    r = tc.patch(f"/api/memory-kinds/fact/{mine}", json={"kind": "opinion"}, headers=viewer)
    assert r.status_code == 403
    r = tc.patch(f"/api/memory-kinds/fact/{mine}", json={"kind": "preference"}, headers=member)
    assert r.status_code == 200, r.text
    assert r.json()["memory_kind"] == "opinion"
    assert r.json()["memory_kind_state"] == "confirmed"
    row = db.execute("SELECT memory_kind_source, fact_type FROM atomic_facts WHERE fact_id=?",
                     (mine,))[0]
    assert (row["memory_kind_source"], row["fact_type"]) == ("user", "opinion")
    r = tc.patch(f"/api/memory-kinds/fact/{theirs}", json={"kind": "rule"}, headers=member)
    assert r.status_code == 404
    row = db.execute("SELECT memory_kind FROM atomic_facts WHERE fact_id=?", (theirs,))[0]
    assert row["memory_kind"] is None
    assert tc.patch(f"/api/memory-kinds/fact/{mine}", json={"kind": "banana"}).status_code == 422


def test_member_set_kind_admitted_in_company_mode(tmp_path, monkeypatch) -> None:
    """L3-15: set-kind/confirm follow the documented WRITE contract.

    Before the fix, ``_authorize_memory_mutation("update", ...)`` admitted
    these routes through ``OperationKind.CORRECT`` (owner/admin only), so a
    MEMBER who holds RBAC WRITE on the profile — and is explicitly allowed by
    this route's own docstring — was refused with 403 in company mode, even
    though the identical personal-mode call (no login required) succeeded.
    """
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    tc, app, db = _client(tmp_path, monkeypatch)
    mine = _fact(db, "Alice likes green tea.")
    member = _session(app, "mo", "member")
    app.state.rbac.set_require_login(True)
    r = tc.patch(f"/api/memory-kinds/fact/{mine}", json={"kind": "opinion"}, headers=member)
    assert r.status_code == 200, r.text
    assert r.json()["memory_kind"] == "opinion"
    fid2 = _fact(db, "We went with SQLite.")
    r2 = tc.post("/api/memory-kinds/confirm",
                 json={"items": [{"fact_id": fid2, "kind": "decision"}]}, headers=member)
    assert r2.status_code == 200, r2.text
    assert r2.json()["items"][0]["ok"] is True


def test_confirm_accepts_the_suggestion_or_a_given_kind(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    suggested = _fact(db, "We went with SQLite.", kind="decision", source="model:laya", conf=0.6)
    plain = _fact(db, "Untyped words.")
    r = tc.post("/api/memory-kinds/confirm", json={"items": [
        {"fact_id": suggested}, {"fact_id": plain}, {"fact_id": plain, "kind": "rule"}]})
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert items[0]["ok"] is True and items[0]["memory_kind"] == "decision"
    assert items[1]["ok"] is False and "suggestion" in items[1]["error"]
    assert items[2]["ok"] is True and items[2]["memory_kind"] == "rule"
    row = db.execute("SELECT memory_kind_source, fact_type FROM atomic_facts WHERE fact_id=?",
                     (suggested,))[0]
    assert (row["memory_kind_source"], row["fact_type"]) == ("user", "episodic")
    too_many = {"items": [{"fact_id": plain}] * 201}
    assert tc.post("/api/memory-kinds/confirm", json=too_many).status_code == 422


def test_jev_consent_must_be_literal_true(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    for bad in ("true", 1, "yes"):
        r = tc.post("/api/memory-kinds/settings", json={"jev_consent": bad})
        assert r.status_code == 422, bad
    assert tc.post("/api/memory-kinds/settings", json={"backend": "skynet"}).status_code == 422
    r = tc.post("/api/memory-kinds/settings", json={"jev_consent": True, "backend": "jev"})
    assert r.status_code == 200, r.text
    got = tc.get("/api/memory-kinds/settings").json()
    assert got["jev_consent"] is True and got["backend"] == "jev"
    assert got["active_backend"] == "rules"          # the answer check is not Jev
    assert app.state.engine._config.memory_kinds.jev_consent is True
    assert (tmp_path / "memory_kinds.json").is_file()


def test_backfill_needs_device_leaving_confirmation_for_jev(tmp_path, monkeypatch) -> None:
    judge = _Judge("jev")
    cfg = memory_kind_config_from({"backend": "jev", "jev_consent": True})
    tc, app, db = _client(tmp_path, monkeypatch, judge=judge, cfg=cfg)
    for t in ("a", "b", "c"):
        _fact(db, t)
    r = tc.post("/api/memory-kinds/backfill", json={"mode": "untyped"})
    assert r.status_code == 409
    body = r.json()
    assert body["needs_confirmation"] is True and body["leaves_device"] is True
    assert body["facts"] == 3 and body["requests"] == 1 and body["detail"]
    bad = tc.post("/api/memory-kinds/backfill",
                  json={"mode": "untyped", "confirm_data_leaves_device": "true"})
    assert bad.status_code == 422
    ok = tc.post("/api/memory-kinds/backfill",
                 json={"mode": "untyped", "confirm_data_leaves_device": True})
    assert ok.status_code == 200 and ok.json()["backend"] == "jev"
    assert judge.calls == 0


def test_schema_not_ready_reason(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch, with_kinds=False)
    status = tc.get("/api/memory-kinds/status").json()
    assert status["schema_ready"] is False and status["reason"]
    r = tc.post("/api/memory-kinds/backfill", json={"mode": "untyped"})
    assert r.status_code == 409 and r.json()["code"] == "schema_not_ready"
    assert tc.get("/api/memory-kinds/suggestions").json() == {"items": []}


def test_suggestions_are_previews_with_kind_fields(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _fact(db, "x" * 500, kind="decision", source="model:laya", conf=0.6)
    _fact(db, "low", kind="rule", source="model:laya", conf=0.05)
    r = tc.get("/api/memory-kinds/suggestions?limit=10")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1 and len(items[0]["content_preview"]) <= 200
    assert items[0]["memory_kind_state"] == "suggested"
    assert tc.get("/api/memory-kinds/suggestions?limit=101").status_code == 422
    assert tc.get("/api/memory-kinds/suggestions?kind=banana").status_code == 422


def test_kinds_from_the_rules_are_counted_and_offered_for_review(tmp_path, monkeypatch) -> None:
    # A rules run stores kinds without a confidence. They must show up in the
    # counts and in the review list, or a rule can never be confirmed.
    tc, app, db = _client(tmp_path, monkeypatch)
    ruled = _fact(db, "Never push to main on Fridays.", kind="rule", source="rules")
    _fact(db, "A model guess with no confidence.", kind="decision", source="model:laya")
    counts = tc.get("/api/memory-kinds/status").json()["counts"]
    assert counts["kind"]["rule"]["suggested"] == 1
    assert counts["kind"]["decision"]["suggested"] == 0
    assert counts["legacy"] == 1
    items = tc.get("/api/memory-kinds/suggestions?kind=rule").json()["items"]
    assert [i["fact_id"] for i in items] == [ruled]
    assert items[0]["memory_kind_state"] == "suggested"
    assert items[0]["memory_kind_source"] == "rules"


def test_errors_never_leak_a_traceback(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("/Users/alice/private-notes memory text")

    monkeypatch.setattr(app.state.memory_kind_backfill, "status", boom)
    r = tc.get("/api/memory-kinds/status")
    assert r.status_code == 500
    assert "private-notes" not in r.text and "Traceback" not in r.text


def test_start_and_stop_backfill_own_one_thread(tmp_path, monkeypatch) -> None:
    import threading

    tc, app, db = _client(tmp_path, monkeypatch)
    del app.state.memory_kind_backfill
    runner = routes.start_backfill(app)
    assert routes.start_backfill(app) is runner          # idempotent
    assert any(t.name == "slm-kind-backfill" for t in threading.enumerate())
    assert routes.stop_backfill(app) is True
    assert not any(t.name == "slm-kind-backfill" for t in threading.enumerate())
