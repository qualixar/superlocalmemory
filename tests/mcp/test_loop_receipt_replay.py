"""PR-247/6: genuine old PASS receipts cannot certify a changed build."""

import asyncio
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from superlocalmemory.loops.execution_receipts import run_check
from superlocalmemory.mcp.tools_loops import register_loop_tools
from tests.loops.hermetic_fixtures import private_data_root as private_data_root
from tests.mcp.test_loop_independent_receipts import _threaded_receipt_connection
from tests.mcp.test_loop_tools import _Capture, _FakeEngine


def checker(tmp_path):
    state = tmp_path / "state"
    state.write_text("pass")
    command = [
        sys.executable, "-I", "-c",
        "from pathlib import Path; raise SystemExit(0 if Path('state').read_text()=='pass' else 1)",
    ]
    return state, command


def harness(db, profile_id="default"):
    engine = _FakeEngine()
    engine._db = db
    engine.profile_id = profile_id
    engine.recall = lambda *a, **k: SimpleNamespace(
        results=[{"score": .99, "content": "Build FAILED; deployment BLOCKED"}],
        no_confident_match=False,
    )
    server = _Capture()
    register_loop_tools(server, lambda: engine)
    return server.fns["slm_loop_run"]


@pytest.mark.asyncio
async def test_exact_old_passing_contract_cannot_finish_new_failing_task(tmp_path):
    conn = _threaded_receipt_connection()
    try:
        db = SimpleNamespace(execute=lambda s, p=(): conn.execute(s, p).fetchall())
        state, command = checker(tmp_path)
        receipt = run_check(
            db, profile_id="default", task_id="build", run_id="old-job", producer="checker",
            command=command, cwd=tmp_path,
        )
        assert receipt["verdict"] == "PASS"
        state.write_text("fail")
        contract = {
            key: receipt[key] for key in ("task_id", "run_id", "producer", "command_digest")
        }
        contract["min_timestamp"] = receipt["produced_at"]
        outcome = await harness(db)("new-build", "build", max_iterations=1, gate_contract=contract)
        assert outcome["status"] == "HALT", outcome
        assert outcome["passed"] is False
        assert len(db.execute("SELECT * FROM loop_execution_receipts")) == 1
        assert run_check(
            db, profile_id="default", task_id="build", run_id="probe", producer="checker",
            command=command, cwd=tmp_path,
        )["exit_code"] == 1
    finally:
        conn.close()


@pytest.fixture
def receipt_db():
    conn = _threaded_receipt_connection()
    yield SimpleNamespace(execute=lambda s, p=(): conn.execute(s, p).fetchall())
    conn.close()


def registered(db, tmp_path, *, prior=True, producer="checker", ttl=3600):
    from superlocalmemory.loops.receipt_jobs import register_job

    state, command = checker(tmp_path)
    job = register_job(
        db, profile_id="default", task_id="build", watcher_name="build", producer=producer,
        command=command, cwd=tmp_path, artifact_paths=[state],
        allow_prestart_evidence=prior, ttl_seconds=ttl,
    )
    return state, job


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["changed", "superseded", "other_task", "expired", "overrides"])
async def test_registered_job_cannot_be_replayed_or_substituted(receipt_db, tmp_path, case):
    from superlocalmemory.loops.receipt_jobs import run_registered_check

    state, job = registered(receipt_db, tmp_path, ttl=.2 if case == "expired" else 3600)
    try:
        receipt = run_registered_check(receipt_db, job['job_id'])
    except ValueError as error:
        assert case == 'expired' and 'expired' in str(error)
    else:
        assert receipt['verdict'] == 'PASS'
    contract, name = {"job_id": job["job_id"]}, "build"
    if case == "changed":
        state.write_text("fail")
    elif case == "superseded":
        registered(receipt_db, tmp_path)
    elif case == "other_task":
        name = "new-build"
    elif case == "expired":
        await asyncio.sleep(.25)
    else:
        contract["min_timestamp"] = "1900-01-01T00:00:00+00:00"
    outcome = await harness(receipt_db)(name, "build", max_iterations=1, gate_contract=contract)
    assert outcome["status"] == "HALT" and not outcome["passed"], outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["approved_prior", "unapproved_prior", "self", "failed"])
async def test_only_operator_approved_current_job_can_finish(receipt_db, tmp_path, case):
    from superlocalmemory.loops.receipt_jobs import run_registered_check

    state, job = registered(
        receipt_db, tmp_path, prior=case != "unapproved_prior",
        producer="mcp_client" if case == "self" else "checker",
    )
    if case == "failed":
        # Register this failing revision, rather than reject only an outdated revision.
        state.write_text("fail")
        from superlocalmemory.loops.receipt_jobs import register_job

        _, command = checker(tmp_path)
        state.write_text("fail")
        job = register_job(
            receipt_db, profile_id="default", task_id="build", watcher_name="build",
            producer="checker", command=command, cwd=tmp_path, artifact_paths=[state],
            allow_prestart_evidence=True,
        )
    receipt = run_registered_check(receipt_db, job["job_id"])
    assert receipt["verdict"] == ("FAIL" if case == "failed" else "PASS")
    outcome = await harness(receipt_db)(
        "build", "build", max_iterations=1, gate_contract={"job_id": job["job_id"]},
    )
    assert outcome["status"] == ("DONE" if case == "approved_prior" else "HALT"), outcome


@pytest.mark.asyncio
async def test_waiter_observes_new_operator_check(receipt_db, tmp_path):
    from superlocalmemory.loops.receipt_jobs import run_registered_check

    _, job = registered(receipt_db, tmp_path, prior=False)

    async def execute():
        await asyncio.sleep(.1)
        return await asyncio.to_thread(run_registered_check, receipt_db, job["job_id"])

    checker_task = asyncio.create_task(execute())
    outcome = await harness(receipt_db)(
        # Budget, not a deadline: the check runs a real subprocess, which on a
        # busy machine outlasted 3 x 0.25 s and made this test flaky.
        "build", "build", max_iterations=40, poll_interval_s=.25,
        gate_contract={"job_id": job["job_id"]},
    )
    await checker_task
    assert outcome["status"] == "DONE" and outcome["laps"] >= 2, outcome


@pytest.mark.asyncio
async def test_artifact_alias_retarget_cannot_replay_pass(receipt_db, tmp_path):
    from superlocalmemory.loops.receipt_jobs import register_job

    state, command = checker(tmp_path)
    passed, failed = tmp_path / "pass_state", tmp_path / "fail_state"
    passed.write_text("pass")
    failed.write_text("fail")
    state.unlink()
    state.symlink_to(passed)
    with pytest.raises(ValueError, match="symlink"):
        register_job(
            receipt_db, profile_id="default", task_id="build", watcher_name="build",
            producer="checker", command=command, cwd=tmp_path, artifact_paths=[state],
            allow_prestart_evidence=True,
        )
    outcome = await harness(receipt_db)(
        "build", "build", max_iterations=1, gate_contract={"job_id": "0" * 32},
    )
    assert outcome["status"] == "HALT", outcome


def test_job_registration_immutable_and_erased_with_profile(receipt_db, tmp_path):
    from superlocalmemory.loops.receipt_jobs import run_registered_check

    receipt_db.execute("PRAGMA foreign_keys=ON")
    _, job = registered(receipt_db, tmp_path)
    run_registered_check(receipt_db, job["job_id"])
    with pytest.raises(sqlite3.IntegrityError):
        receipt_db.execute("UPDATE loop_gate_jobs SET allow_prestart_evidence=1")
    with pytest.raises(sqlite3.IntegrityError):
        receipt_db.execute("DELETE FROM loop_gate_jobs")
    receipt_db.execute("DELETE FROM profiles WHERE profile_id='default'")
    assert not receipt_db.execute("SELECT * FROM loop_gate_jobs")
    assert not receipt_db.execute("SELECT * FROM loop_execution_receipts")


@pytest.mark.asyncio
async def test_registered_job_cannot_cross_profile(receipt_db, tmp_path):
    from superlocalmemory.loops.receipt_jobs import run_registered_check

    _, job = registered(receipt_db, tmp_path)
    run_registered_check(receipt_db, job["job_id"])
    outcome = await harness(receipt_db, "foreign")(
        "build", "build", max_iterations=1, gate_contract={"job_id": job["job_id"]},
    )
    assert outcome["status"] == "HALT", outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["artifact", "registration"])
async def test_gate_rechecks_revision_at_receipt_read(receipt_db, tmp_path, monkeypatch, change):
    from superlocalmemory.loops.receipt_jobs import run_registered_check

    state, job = registered(receipt_db, tmp_path)
    run_registered_check(receipt_db, job["job_id"])
    execute = receipt_db.execute
    interleaved = False

    def read(sql, params=()):
        nonlocal interleaved
        if sql.startswith("SELECT * FROM loop_execution_receipts") and not interleaved:
            interleaved = True
            if change == "artifact":
                state.write_text("fail")
            else:
                registered(receipt_db, tmp_path)
        return execute(sql, params)

    monkeypatch.setattr(receipt_db, "execute", read)
    outcome = await harness(receipt_db)(
        "build", "build", max_iterations=1, gate_contract={"job_id": job["job_id"]},
    )
    assert outcome["status"] == "HALT", outcome
