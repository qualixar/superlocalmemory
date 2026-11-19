# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Poisoning scenario 4 (W4 exit gate): a secret pasted into ``remember`` is stored only as a placeholder.

The whole data directory is read back as bytes (the database, its write-ahead log, the
admission journal, every index inside them, and any other file the save wrote) and the secret
must be nowhere in it. With the flag off the same text is stored as written, so the test
also proves it can see a secret that is there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, env_name
from tests.test_server.test_canonical_remember_route import _client
from tests.test_trust_firewall import secrets as fake

FLAG = env_name("trust.secret_screen")


def _everywhere(root: Path, needle: str) -> list[str]:
    """The files under ``root`` whose bytes hold ``needle`` (a database, a journal, a log...)."""
    raw = needle.encode()
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and raw in p.read_bytes())


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv(KILL_ENV, raising=False)
    monkeypatch.setenv(FLAG, "off")
    return tmp_path


def _remember(engine, text, key, headers=None):
    with _client(engine) as client:
        r = client.post("/remember?wait=true", headers=headers or {},
                        json={"content": text, "idempotency_key": key,
                              "tags": f"deploy,{fake.stripe()}",
                              "metadata": {"note": f"also {fake.slack()}"}})
        assert r.status_code == 200, r.text[:300]
    return r.json()


def _checkpoint(engine) -> None:
    with engine._db.raw_connection() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def test_with_the_screen_off_the_test_can_see_a_secret(home, engine_with_mock_deps) -> None:
    secret = fake.github()
    _remember(engine_with_mock_deps, f"Staging deploy token is {secret} for this sprint.",
              "off-1")
    _checkpoint(engine_with_mock_deps)
    assert _everywhere(home, secret), "the screen is off: the text is stored as written"


def test_a_secret_pasted_into_remember_is_stored_only_as_a_placeholder(
        home, engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    engine = engine_with_mock_deps
    secrets = [fake.github(), fake.aws(), fake.anthropic(), fake.pem()]
    for i, secret in enumerate(secrets):
        got = _remember(engine, f"Deploy notes number {i}. The credential is {secret}. "
                                "Rotate it after the release on Friday.", f"on-{i}")
        assert got["fact_ids"]
    _checkpoint(engine)
    leaks = {s[:12]: _everywhere(home, s) for s in (*secrets, fake.stripe(), fake.slack())}
    assert leaks == {k: [] for k in leaks}, leaks
    stored = " ".join(r["content"] for r in engine._db.execute("SELECT content FROM atomic_facts"))
    stored += " ".join(r["content"] for r in engine._db.execute("SELECT content FROM memories"))
    for placeholder in ("[REDACTED:GITHUB:", "[REDACTED:AWS:", "[REDACTED:ANTHROPIC:",
                        "[REDACTED:PRIVATE_KEY:"):
        assert placeholder in stored
    assert "Rotate it after the release on Friday." in stored  # the sentence around it survives


def test_the_full_text_index_cannot_find_the_secret(home, engine_with_mock_deps,
                                                    monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    engine = engine_with_mock_deps
    secret = fake.github()
    _remember(engine, f"Release bot token {secret} is for the nightly job.", "fts-1")
    hits = engine._db.execute(
        "SELECT COUNT(*) AS n FROM atomic_facts WHERE content LIKE ?", (f"%{secret[4:20]}%",))
    assert hits[0]["n"] == 0


def test_the_python_api_door_screens_too(home, engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    engine = engine_with_mock_deps
    secret = fake.aws()
    engine.store(f"The data pipeline reads from the bucket with key {secret} every night.")
    _checkpoint(engine)
    assert _everywhere(home, secret) == []


def test_the_observe_door_screens_too(home, engine_with_mock_deps, monkeypatch) -> None:
    from superlocalmemory.server.unified_daemon import ObserveBuffer

    monkeypatch.setenv(FLAG, "on")
    engine = engine_with_mock_deps
    secret = fake.github()
    buf = ObserveBuffer(debounce_sec=0.01)
    buf.set_engine(engine)
    buf.enqueue(f"I decided to deploy every Friday using the token {secret} from now on.",
                trusted_actor_id="local-capability:test")
    _checkpoint(engine)
    assert _everywhere(home, secret) == []


def test_the_edit_door_screens_too(home, engine_with_mock_deps, monkeypatch) -> None:
    from superlocalmemory.core.mutations import update_fact_authorized

    monkeypatch.setenv(FLAG, "on")
    engine = engine_with_mock_deps
    secret = fake.github()
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.migrations import M018_ingestion_operations as m018
    from superlocalmemory.storage.migrations import M032_write_coordinator_admission as m032
    from superlocalmemory.storage.migrations import M042_correction_case_ledger as m042

    fid = engine.store("The release job runs every night at two in the morning.")[0]
    with engine._db.raw_connection() as conn:
        for m in (m018, m032, m042):
            m.apply(conn)
    runtime = CanonicalRememberRuntime.for_engine(engine)
    runtime.start()
    try:
        result = update_fact_authorized(
            engine, fid, f"The release job runs every night with the token {secret} attached.",
            trusted_actor_id="local-capability:test", source_agent_id="test-edit",
            canonical_runtime=runtime)
    finally:
        runtime.stop()
    assert result.get("ok"), result
    assert secret not in str(result)
    _checkpoint(engine)
    assert _everywhere(home, secret) == []


def test_a_screen_that_cannot_run_refuses_the_save_instead_of_storing_clear(
        home, engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")

    def boom(_text):
        raise RuntimeError("screen failed")

    monkeypatch.setattr("superlocalmemory.core.security_primitives.redact_secrets",
                        lambda *a, **k: boom(a[0]))
    secret = fake.github()
    with _client(engine_with_mock_deps) as client:
        r = client.post("/remember?wait=true",
                        json={"content": f"token {secret} for the nightly build job",
                              "idempotency_key": "refuse-1"})
    assert r.status_code >= 400
    _checkpoint(engine_with_mock_deps)
    assert _everywhere(home, secret) == []


# -- A-11 (Fable audit): the side fields of ``remember`` are screened too --------------------------

def _post(engine, **fields):
    with _client(engine) as client:
        r = client.post("/remember?wait=true", json={"idempotency_key": fields.pop("key"), **fields})
    return r


def test_a_secret_in_slot_value_is_stored_only_as_a_placeholder(
        home, engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    secret = fake.github()
    r = _post(engine_with_mock_deps, key="sv-1", slot="staging/deploy_token", slot_value=secret,
              content="The deploy token for staging was rotated today.")
    assert r.status_code == 200, r.text[:300]
    _checkpoint(engine_with_mock_deps)
    assert _everywhere(home, secret) == []
    meta = " ".join(r["metadata_json"] for r in engine_with_mock_deps._db.execute(
        "SELECT metadata_json FROM memories"))
    assert "[REDACTED:GITHUB:" in meta  # the claim survives, as a placeholder


def test_a_secret_in_the_slot_spec_never_reaches_the_store(
        home, engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    secret = fake.aws()
    r = _post(engine_with_mock_deps, key="ss-1", slot=f"bucket/{secret}", slot_value="rotated",
              content="The bucket key was rotated on Friday by the ops team.")
    assert r.status_code in (200, 422), r.text[:300]  # stored as a placeholder, or refused
    assert secret not in r.text
    print("SLOT SPEC STATUS", r.status_code)
    _checkpoint(engine_with_mock_deps)
    assert _everywhere(home, secret) == []


def test_a_secret_in_a_cue_pattern_is_stored_only_as_a_placeholder(
        home, engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    secret = fake.github()
    r = _post(engine_with_mock_deps, key="cu-1", cues=[f"error:{secret}"],
              content="When this error shows, restart the staging worker pool.")
    assert r.status_code == 200, r.text[:300]
    assert secret not in r.text
    _checkpoint(engine_with_mock_deps)
    assert _everywhere(home, secret) == []
    from superlocalmemory.core import cue_binding

    with engine_with_mock_deps._db.raw_connection() as conn:
        cue_binding.bind_explicit(conn)  # the maintenance pass that turns the metadata into rows
        conn.commit()
    pats = [r["pattern"] for r in engine_with_mock_deps._db.execute("SELECT pattern FROM memory_cues")]
    assert pats and all("[REDACTED:GITHUB:" in p for p in pats)  # the cue survives, screened
    _checkpoint(engine_with_mock_deps)
    assert _everywhere(home, secret) == []


def test_the_python_api_door_screens_the_slot_too(home, engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    secret = fake.github()
    engine_with_mock_deps.store("The deploy token for staging was rotated today.",
                                slot="staging/deploy_token", slot_value=secret)
    _checkpoint(engine_with_mock_deps)
    assert _everywhere(home, secret) == []


def test_with_the_screen_off_the_side_fields_are_stored_as_written(
        home, engine_with_mock_deps) -> None:
    secret = fake.github()
    r = _post(engine_with_mock_deps, key="off-sv", slot="staging/deploy_token", slot_value=secret,
              cues=[f"error:{secret}"], content="The deploy token for staging was rotated today.")
    assert r.status_code == 200, r.text[:300]
    _checkpoint(engine_with_mock_deps)
    assert _everywhere(home, secret), "the screen is off: the test must see what is stored"
