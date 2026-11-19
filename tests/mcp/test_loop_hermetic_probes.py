"""PR-247 recheck 3: independent adversarial probes against the fetched head.

Every probe uses real register_job / run_registered_check / slm_loop_run; only
the recall result is a fixed fake (it says the build FAILED with score .99).
Markers live OUTSIDE the check's cwd and its declared ancestors, so the only
thing the guard can observe is the attack itself.
"""

import mmap
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests._isolation_plugin  # noqa: F401
from superlocalmemory.loops import private_workspace as workspace
from superlocalmemory.loops import receipt_jobs
from superlocalmemory.loops.execution_guard import admission
from superlocalmemory.loops.execution_receipts import run_check
from superlocalmemory.loops.receipt_jobs import register_job, run_registered_check
from tests._portable import GUARD_ENV
from tests.mcp.test_loop_independent_receipts import _threaded_receipt_connection
from tests.mcp.test_loop_receipt_replay import harness

PY = sys.executable


@pytest.fixture
def db():
    conn = _threaded_receipt_connection()
    conn.execute("INSERT OR IGNORE INTO profiles VALUES ('other')")
    yield SimpleNamespace(execute=lambda sql, p=(): conn.execute(sql, p).fetchall())
    conn.close()


def receipts(db):
    if not db.execute("SELECT name FROM sqlite_master WHERE name='loop_execution_receipts'"):
        return []
    return [dict(r) for r in db.execute("SELECT * FROM loop_execution_receipts")]


def register(db, cwd, command, artifacts, *, profile="default", watcher="build"):
    return register_job(
        db,
        profile_id=profile,
        task_id="build",
        watcher_name=watcher,
        producer="checker",
        command=command,
        cwd=cwd,
        check_paths=[p for p in artifacts if Path(p).suffix == ".py"],
        artifact_paths=[p for p in artifacts if Path(p).suffix != ".py"],
        allow_prestart_evidence=True,
    )


async def watch(db, job_id, name="build", profile="default"):
    return await harness(db, profile)(
        name, "build", max_iterations=1, gate_contract={"job_id": job_id}
    )


def rerun(command, cwd):
    return subprocess.run(
        command, cwd=cwd, env={"PATH": os.defpath, "HOME": str(cwd), **GUARD_ENV}, capture_output=True
    ).returncode


def wait_for(path, limit=20):
    deadline = time.monotonic() + limit
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.005)


SYNC_CHECK = """import sys, time
from pathlib import Path
sync = Path(sys.argv[1]); target = sys.argv[2]
(sync / 'started').touch()
deadline = time.monotonic() + 20
while not (sync / 'go').exists():
    if time.monotonic() > deadline: raise SystemExit(2)
    time.sleep(.005)
passed = Path(target).read_text() == 'pass'
(sync / 'read_done').touch()
while not (sync / 'restored').exists():
    if time.monotonic() > deadline: raise SystemExit(2)
    time.sleep(.005)
raise SystemExit(0 if passed else 1)
"""


def sync_job(db, cwd, sync, target_rel, state_path):
    script = cwd / "check.py"
    script.write_text(SYNC_CHECK)
    command = [PY, "-I", str(script), str(sync), target_rel]
    return command, register(db, cwd, command, [state_path, script])


def run_with(db, job, attack):
    errors = []

    def body():
        try:
            attack()
        except Exception as error:  # surfaced by the assertion below
            errors.append(repr(error))

    thread = threading.Thread(target=body)
    thread.start()
    try:
        try:
            outcome = run_registered_check(db, job["job_id"])
            outcome = {"verdict": outcome["verdict"], "exit": outcome["exit_code"]}
        except ValueError as error:
            outcome = {"refused": str(error)}
    finally:
        thread.join(30)
    assert not errors, errors
    return outcome


def layout(tmp_path):
    cwd, sync = tmp_path / "work", tmp_path / "sync"
    cwd.mkdir()
    sync.mkdir()
    return cwd, sync


# --- Recheck races, markers outside cwd: ctime alone must refuse ------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("restore_mtime", [False, True])
async def test_recheck_r1_aba_cannot_change_copied_inputs(db, tmp_path, restore_mtime):
    cwd, sync = layout(tmp_path)
    state = cwd / "state"
    state.write_text("fail")
    command, job = sync_job(db, cwd, sync, "state", state)
    base = state.stat()

    def attack():
        wait_for(sync / "started")
        with open(state, "r+b", buffering=0) as handle:
            handle.write(b"pass")
        (sync / "go").touch()
        wait_for(sync / "read_done")
        with open(state, "r+b", buffering=0) as handle:
            handle.write(b"fail")
        if restore_mtime:
            os.utime(state, ns=(base.st_atime_ns, base.st_mtime_ns))
        (sync / "restored").touch()

    outcome = run_with(db, job, attack)
    result = await watch(db, job["job_id"])
    print(
        {
            "probe": "R1",
            "restore_mtime": restore_mtime,
            "run": outcome,
            "receipts": len(receipts(db)),
            "watcher": result["status"],
        }
    )
    assert outcome == {"verdict": "FAIL", "exit": 1} and result["status"] == "HALT"


@pytest.mark.asyncio
async def test_recheck_r2_rejected_pass_not_usable_after_restore(db, tmp_path):
    cwd, sync = layout(tmp_path)
    state = cwd / "state"
    state.write_text("fail")
    command, job = sync_job(db, cwd, sync, "state", state)

    def attack():
        wait_for(sync / "started")
        state.write_text("pass")
        (sync / "go").touch()
        wait_for(sync / "read_done")
        (sync / "restored").touch()

    outcome = run_with(db, job, attack)
    state.write_text("fail")
    result = await watch(db, job["job_id"])
    print(
        {"probe": "R2", "run": outcome, "receipts": len(receipts(db)), "watcher": result["status"]}
    )
    assert "refused" in outcome and not receipts(db) and result["status"] == "HALT"


# --- New bypasses ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bypass_mmap_dirty_page_aba(db, tmp_path):
    """Write through an already-dirty shared mapping: APFS updates no stamp until msync."""
    cwd, sync = layout(tmp_path)
    state = cwd / "state"
    state.write_text("fail")
    command, job = sync_job(db, cwd, sync, "state", state)
    fd = os.open(state, os.O_RDWR)
    mapping = mmap.mmap(fd, 4)
    mapping[:] = b"fail"  # dirty the page with identical bytes before the guard starts
    time.sleep(0.05)

    def attack():
        wait_for(sync / "started")
        mapping[:] = b"pass"
        (sync / "go").touch()
        wait_for(sync / "read_done")
        mapping[:] = b"fail"
        (sync / "restored").touch()

    try:
        outcome = run_with(db, job, attack)
        result = await watch(db, job["job_id"])
    finally:
        mapping.close()
        os.close(fd)
    assert outcome == {"verdict": "FAIL", "exit": 1}
    print(
        {
            "probe": "mmap",
            "registered_bytes": state.read_text(),
            "run": outcome,
            "watcher": result["status"],
            "passed": result["passed"],
        }
    )
    assert result["status"] == "HALT", f"mmap ABA certified failing state: {outcome} {result}"


def hdi(*args):
    subprocess.run(["hdiutil", *map(str, args)], check=True, stdin=subprocess.DEVNULL)


@pytest.mark.asyncio
async def test_bypass_mount_over_declared_ancestor(db, tmp_path):
    """Mount ABA in the live tree must never affect the already-copied input."""
    if sys.platform != "darwin":
        pytest.skip("hdiutil probe")
    cwd, sync = layout(tmp_path)
    data = cwd / "data"
    data.mkdir()
    state = data / "state"
    state.write_text("fail")
    src = tmp_path / "img_src"
    src.mkdir()
    (src / "state").write_text("pass")
    image = tmp_path / "pass.dmg"
    hdi("create", "-quiet", "-srcfolder", src, "-format", "UDRO", "-ov", image)
    _, job = sync_job(db, cwd, sync, "data/state", state)

    def attack():
        wait_for(sync / "started")
        hdi("attach", "-quiet", "-nobrowse", "-readonly", "-mountpoint", data, image)
        try:
            (sync / "go").touch()
            wait_for(sync / "read_done")
        finally:
            hdi("detach", "-quiet", data)
            (sync / "restored").touch()

    outcome = run_with(db, job, attack)
    assert outcome == {"verdict": "FAIL", "exit": 1}
    assert state.read_text() == "fail"
    assert (await watch(db, job["job_id"]))["status"] == "HALT"


def rewrite_restore(state, sync, base):
    wait_for(sync / "started")
    fd = os.open(state, os.O_RDWR)
    try:
        os.pwrite(fd, b"pass", 0)
        os.fsync(fd)
        (sync / "go").touch()
        wait_for(sync / "read_done")
        os.pwrite(fd, b"fail", 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.utime(state, ns=(base.st_atime_ns, base.st_mtime_ns))
    (sync / "restored").touch()


@pytest.mark.asyncio
@pytest.mark.parametrize("fs", ["ExFAT", "MS-DOS"])
async def test_bypass_posix_host_without_change_time(db, tmp_path, fs):
    """Filesystem timestamp semantics have no authority over copied bytes."""
    if sys.platform != "darwin":
        pytest.skip("hdiutil probe")
    image, mount = tmp_path / "vol.dmg", tmp_path / "vol"
    mount.mkdir()
    hdi("create", "-quiet", "-size", "8m", "-fs", fs, "-volname", "V", "-ov", image)
    hdi("attach", "-quiet", "-nobrowse", "-mountpoint", mount, image)
    try:
        cwd, sync = mount / "work", tmp_path / "sync"
        cwd.mkdir()
        sync.mkdir()
        state = cwd / "state"
        state.write_text("fail")
        _, job = sync_job(db, cwd, sync, "state", state)
        time.sleep(2.2)  # retain the original FAT-granularity attack
        base = state.stat()
        outcome = run_with(db, job, lambda: rewrite_restore(state, sync, base))
        assert outcome == {"verdict": "FAIL", "exit": 1}
        assert state.read_text() == "fail"
        assert (await watch(db, job["job_id"]))["status"] == "HALT"
    finally:
        hdi("detach", "-quiet", "-force", mount)


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["shadow_module", "pytest_conftest"])
async def test_bypass_undeclared_file_planted_after_registration(db, tmp_path, variant):
    """The revision binds declared bytes only; the namespace is first captured at exec."""
    cwd = tmp_path / "work"
    cwd.mkdir()
    state = cwd / "state"
    state.write_text("fail")
    if variant == "shadow_module":
        script = cwd / "check.py"
        script.write_text(
            "import json, pathlib\n"
            "raise SystemExit(0 if pathlib.Path('state').read_text()=='pass' else 1)\n"
        )
        command = [PY, "check.py"]
        declared = [state, script]
        plant = {"json.py": "import os\nos._exit(0)\n"}
    else:
        test = cwd / "test_state.py"
        test.write_text(
            "from pathlib import Path\n\n"
            "def test_state():\n    assert Path('state').read_text() == 'pass'\n"
        )
        command = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_state.py"]
        declared = [state, test]
        plant = {"conftest.py": "import os\nos._exit(0)\n"}
    (cwd / "__pycache__").mkdir()  # an earlier honest run leaves this behind
    job = register(db, cwd, command, declared)
    honest = rerun(command, cwd)
    for name, body in plant.items():  # task writer, after registration, before execution
        (cwd / name).write_text(body)
    receipt = run_registered_check(db, job["job_id"])
    result = await watch(db, job["job_id"])
    for name in plant:
        (cwd / name).unlink()
    after = rerun(command, cwd)
    assert honest != 0 and after != 0 and receipt["verdict"] == "FAIL"
    assert result["status"] == "HALT", (
        f"planted undeclared input certified failing revision {result}"
    )


# --- Boundary checks that should hold -----------------------------------------------


def _simple(db, tmp_path, code="raise SystemExit(0)", **kw):
    cwd = tmp_path / "work"
    cwd.mkdir(exist_ok=True)
    script = cwd / "check.py"
    script.write_text(code)
    return register(db, cwd, [PY, "-I", str(script)], [script], **kw), cwd, script


@pytest.mark.asyncio
async def test_cross_profile_job_cannot_finish_other_profile(db, tmp_path):
    job, *_ = _simple(db, tmp_path, profile="other")
    run_registered_check(db, job["job_id"])
    result = await watch(db, job["job_id"], profile="default")
    assert result["status"] == "HALT"


@pytest.mark.asyncio
async def test_other_watchers_receipt_and_superseded_job(db, tmp_path):
    job_a, cwd, script = _simple(db, tmp_path, watcher="build")
    run_registered_check(db, job_a["job_id"])
    job_b = register(db, cwd, [PY, "-I", str(script)], [script], watcher="build")
    assert (await watch(db, job_a["job_id"]))["status"] == "HALT"  # superseded
    assert (await watch(db, job_b["job_id"]))["status"] == "HALT"  # B never ran
    assert (await watch(db, job_a["job_id"], name="deploy"))["status"] == "HALT"


@pytest.mark.asyncio
async def test_unguarded_writer_receipt_without_admission_is_ineligible(db, tmp_path):
    job, cwd, script = _simple(db, tmp_path)
    run_check(
        db,
        profile_id="default",
        task_id="build",
        run_id=job["run_id"],
        producer="checker",
        command=[PY, "-I", str(script)],
        cwd=cwd,
    )
    assert (await watch(db, job["job_id"]))["status"] == "HALT"


@pytest.mark.asyncio
async def test_public_writer_mints_admission_without_guard(db, tmp_path):
    """run_check is public and trusts any validate_result: proof is unauthenticated."""
    job, cwd, script = _simple(db, tmp_path, code="raise SystemExit(1)")
    script.write_text("raise SystemExit(0)")  # differs from registered revision
    run_check(
        db,
        profile_id="default",
        task_id="build",
        run_id=job["run_id"],
        producer="checker",
        command=[PY, "-I", str(script)],
        cwd=cwd,
        validate_result=lambda: admission(job),
    )
    script.write_text("raise SystemExit(1)")  # restore registered failing revision
    result = await watch(db, job["job_id"])
    print({"probe": "public_writer_forgery", "watcher": result["status"]})
    assert result["status"] == "HALT", "admission proof can be minted by any in-process caller"


def test_validation_exception_publishes_nothing(db, tmp_path, monkeypatch):
    job, *_ = _simple(db, tmp_path)
    monkeypatch.setattr(receipt_jobs, "matches", lambda j: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        run_registered_check(db, job["job_id"])
    assert not receipts(db)


def test_binary_output_is_hashed_without_raw_receipt_output(db, tmp_path):
    job, *_ = _simple(
        db, tmp_path, code="import os; os.write(1, b'\\xff\\xfe'); raise SystemExit(0)"
    )
    import hashlib
    import json

    receipt = run_registered_check(db, job['job_id'])
    evidence = json.loads(receipt['evidence'])
    assert evidence['stdout_sha256'] == hashlib.sha256(b'\xff\xfe').hexdigest()
    assert 'stdout' not in evidence
    assert not any(k.endswith('_excerpt') for k in evidence)
    assert len(receipts(db)) == 1


def test_check_inherits_operator_stdin(db, tmp_path):
    """stdin is not pinned to DEVNULL: an undeclared input can decide the verdict."""
    job, *_ = _simple(
        db, tmp_path, code="import sys\nraise SystemExit(0 if sys.stdin.read(4)=='pass' else 1)"
    )
    read, write = os.pipe()
    os.write(write, b"pass")
    os.close(write)
    saved = os.dup(0)
    os.dup2(read, 0)
    try:
        verdict = run_registered_check(db, job["job_id"])["verdict"]
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read)
    print({"probe": "stdin", "verdict": verdict})
    assert verdict == "FAIL"  # stdin is never a declared input


# --- Availability / honesty ---------------------------------------------------------


def test_unrelated_parent_sibling_write_refuses_honest_pass(db, tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    script = cwd / "check.py"
    script.write_text(
        "import sys, time\nfrom pathlib import Path\nPath(sys.argv[1]).touch(); time.sleep(.3)\n"
    )
    marker = tmp_path / "sibling_marker"  # creating it changes tmp_path, a parent of cwd
    job = register(db, cwd, [PY, "-I", str(script), str(marker)], [script])
    assert run_registered_check(db, job["job_id"])["verdict"] == "PASS"


def test_check_writing_home_cache_refuses_honest_pass(db, tmp_path):
    job, *_ = _simple(
        db,
        tmp_path,
        code=(
            "import os\nfrom pathlib import Path\n"
            "Path(os.environ['HOME'], '.cache').mkdir(exist_ok=True)\n"
        ),
    )
    assert run_registered_check(db, job["job_id"])["verdict"] == "PASS"


@pytest.mark.asyncio
async def test_unsupported_platform_refuses_registration_and_explains_halt(
    db, tmp_path, monkeypatch
):
    monkeypatch.setattr(workspace.os, "supports_dir_fd", set())
    with pytest.raises(ValueError, match="unsupported.*no-follow") as error:
        _simple(db, tmp_path)
    result = await watch(db, "0" * 32)
    assert result["status"] == "HALT"
    assert str(error.value) in result["reason"]
    assert "no-follow" in result["receipt_reason"]


@pytest.fixture(autouse=True)
def private_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
