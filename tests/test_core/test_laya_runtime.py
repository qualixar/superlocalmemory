# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Laya (the local answer check) must never strand a user: detect() always
tells the truth about what is actually on disk and verified, install() is
resumable and single-flight, verify() always kills its worker, and remove()
only ever deletes the exact folder it is told to manage.

Every subprocess boundary is mocked here — this suite loads no real model.
The one real, offline, model-loading check lives in test_laya_runtime_native.py
and is skipped unless the local install it needs is present.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

try:
    import fcntl
except ImportError:  # Windows: Laya is Apple-Silicon only; detect() already
    fcntl = None  # type: ignore[assignment]  # refuses before any lock is taken.
    # One test below exercises flock() directly to prove a second installer
    # process is refused; it is POSIX-only and is skipped on this platform.

import pytest

from superlocalmemory.core import laya_runtime as lr
from tests.helpers.owned_python import owned_link, owned_python

# Writes Laya runtime state: pin the data root to tmp_path even without the
# root conftest (--noconftest), see tests/isolation_guard.py.
from ..isolation_guard import explicit_slm_root  # noqa: F401

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_install_job():
    lr.LayaInstallJob._instance = None
    yield
    lr.LayaInstallJob._instance = None


@pytest.fixture(autouse=True)
def _apple_silicon(monkeypatch):
    """Most tests assume the supported platform; override per-test when not."""
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


_FAKE_WORKER = r'''
import json, os, sys, time
mode = os.environ.get("FAKE_LAYA_MODE", "pass")
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    req = json.loads(raw)
    cmd = req.get("cmd")
    if cmd == "quit":
        break
    if cmd == "load":
        if mode == "load_fail":
            print(json.dumps({"ok": False, "error": "weights not found"}), flush=True)
        else:
            print(json.dumps({"ok": True, "model": req.get("model")}), flush=True)
        continue
    if cmd == "judge":
        if mode == "pass":
            print(json.dumps({"ok": True, "probabilities": [0.9, 0.1]}), flush=True)
        elif mode == "fail_values":
            print(json.dumps({"ok": True, "probabilities": [0.4, 0.6]}), flush=True)
        elif mode == "judge_error":
            print(json.dumps({"ok": False, "error": "boom"}), flush=True)
        elif mode == "malformed":
            print(json.dumps({"ok": True, "probabilities": [1.5]}), flush=True)
        elif mode == "hang":
            time.sleep(10)
        continue
    print(json.dumps({"ok": False, "error": "unknown"}), flush=True)
'''


@pytest.fixture()
def fake_worker(tmp_path, monkeypatch):
    path = tmp_path / "fake_laya_worker.py"
    path.write_text(_FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(lr, "WORKER_PATH", path)
    return path


# ---------------------------------------------------------------------------
# verify() — against a fake worker that speaks the real protocol
# ---------------------------------------------------------------------------

class TestVerify:
    def test_passes_when_the_canary_separates_cleanly(self, fake_worker, monkeypatch):
        monkeypatch.setenv("FAKE_LAYA_MODE", "pass")
        ok, reason = lr.verify(str(owned_python(fake_worker.parent)), "", str(fake_worker.parent))
        assert ok is True
        assert isinstance(reason, str) and reason

    def test_fails_when_the_canary_does_not_separate(self, fake_worker, monkeypatch):
        monkeypatch.setenv("FAKE_LAYA_MODE", "fail_values")
        ok, _ = lr.verify(str(owned_python(fake_worker.parent)), "", str(fake_worker.parent))
        assert ok is False

    def test_fails_when_load_fails(self, fake_worker, monkeypatch):
        monkeypatch.setenv("FAKE_LAYA_MODE", "load_fail")
        ok, reason = lr.verify(str(owned_python(fake_worker.parent)), "", str(fake_worker.parent))
        assert ok is False
        assert "load" in reason.lower()

    def test_fails_when_judge_errors(self, fake_worker, monkeypatch):
        monkeypatch.setenv("FAKE_LAYA_MODE", "judge_error")
        ok, _ = lr.verify(str(owned_python(fake_worker.parent)), "", str(fake_worker.parent))
        assert ok is False

    def test_fails_on_a_malformed_response(self, fake_worker, monkeypatch):
        monkeypatch.setenv("FAKE_LAYA_MODE", "malformed")
        ok, _ = lr.verify(str(owned_python(fake_worker.parent)), "", str(fake_worker.parent))
        assert ok is False

    def test_never_contains_a_traceback_or_secret(self, fake_worker, monkeypatch):
        monkeypatch.setenv("FAKE_LAYA_MODE", "judge_error")
        ok, reason = lr.verify(str(owned_python(fake_worker.parent)), "", str(fake_worker.parent))
        assert ok is False
        assert "Traceback" not in reason
        assert "boom" not in reason  # the worker's raw error never leaks out

    def test_times_out_and_kills_a_hung_worker(self, fake_worker, monkeypatch):
        monkeypatch.setenv("FAKE_LAYA_MODE", "hang")
        start = time.monotonic()
        ok, reason = lr.verify(str(owned_python(fake_worker.parent)), "", str(fake_worker.parent), timeout_s=0.5)
        elapsed = time.monotonic() - start
        assert ok is False
        assert "time" in reason.lower()
        # Proves the worker was killed rather than waited on (it sleeps 10s).
        assert elapsed < 5.0

    def test_fails_gracefully_when_the_python_does_not_exist(self):
        ok, reason = lr.verify("/no/such/python", "", "/no/such/model")
        assert ok is False
        assert reason


# ---------------------------------------------------------------------------
# detect() — the order matrix
# ---------------------------------------------------------------------------

class TestDetectOrder:
    def test_unsupported_off_apple_silicon(self, monkeypatch):
        monkeypatch.setattr(lr, "_apple_silicon", lambda: False)
        assert lr.detect().state == lr.STATE_UNSUPPORTED

    def test_not_installed_when_nothing_present(self):
        assert lr.detect().state == lr.STATE_NOT_INSTALLED

    def test_ready_from_managed_marker(self, tmp_path):
        run_dir = lr.runtime_dir()
        python = run_dir / "venv" / "bin" / "python"
        owned_link(python, tmp_path)
        model = run_dir / "hf-cache" / "model"
        model.mkdir(parents=True)
        _write_json(run_dir / ".slm-managed", {
            "python": str(python), "model_path": str(model),
            "verified": True, "revision": "rev1",
        })
        status = lr.detect()
        assert status.state == lr.STATE_READY
        assert status.managed is True
        assert status.model_revision == "rev1"

    def test_failed_when_managed_marker_is_unverified(self, tmp_path):
        run_dir = lr.runtime_dir()
        python = run_dir / "venv" / "bin" / "python"
        owned_link(python, tmp_path)
        model = run_dir / "hf-cache" / "model"
        model.mkdir(parents=True)
        _write_json(run_dir / ".slm-managed", {
            "python": str(python), "model_path": str(model), "verified": False,
        })
        status = lr.detect()
        assert status.state == lr.STATE_FAILED
        assert status.step == "Needs repair — choose Repair."
        assert status.action == lr.ACTION_SETUP

    def test_failed_when_managed_files_are_missing_despite_verified_true(self):
        run_dir = lr.runtime_dir()
        run_dir.mkdir(parents=True)
        _write_json(run_dir / ".slm-managed", {
            "python": str(run_dir / "venv" / "bin" / "python"),
            "model_path": str(run_dir / "hf-cache" / "model"),
            "verified": True,
        })
        status = lr.detect()
        assert status.state == lr.STATE_FAILED

    def test_ready_from_adopted_record(self, tmp_path):
        python = tmp_path / "ext-venv" / "bin" / "python"
        owned_link(python, tmp_path)
        model = tmp_path / "ext-model"
        model.mkdir()
        run_dir = lr.runtime_dir()
        _write_json(run_dir / "adopted.json", {
            "python": str(python), "model_path": str(model), "verified": True,
        })
        status = lr.detect()
        assert status.state == lr.STATE_READY
        assert status.managed is False

    def test_adopted_takes_precedence_over_managed(self, tmp_path):
        run_dir = lr.runtime_dir()
        managed_python = run_dir / "venv" / "bin" / "python"
        owned_link(managed_python, tmp_path)
        managed_model = run_dir / "hf-cache" / "model"
        managed_model.mkdir(parents=True)
        _write_json(run_dir / ".slm-managed", {
            "python": str(managed_python), "model_path": str(managed_model), "verified": True,
        })

        adopted_python = tmp_path / "ext-venv" / "bin" / "python"
        owned_link(adopted_python, tmp_path)
        adopted_model = tmp_path / "ext-model"
        adopted_model.mkdir()
        _write_json(run_dir / "adopted.json", {
            "python": str(adopted_python), "model_path": str(adopted_model), "verified": True,
        })

        status = lr.detect()
        assert status.managed is False
        assert status.python == str(adopted_python)

    def test_explicit_cfg_matching_adopted_wins(self, tmp_path):
        python = tmp_path / "ext-venv" / "bin" / "python"
        owned_link(python, tmp_path)
        model = tmp_path / "ext-model"
        model.mkdir()
        run_dir = lr.runtime_dir()
        _write_json(run_dir / "adopted.json", {
            "python": str(python), "model_path": str(model),
            "verified": True, "hf_home": "/orig/hf",
        })
        cfg = SimpleNamespace(sufficiency_python=str(python), sufficiency_model=str(model),
                               sufficiency_hf_home="/override/hf")
        status = lr.detect(cfg)
        assert status.state == lr.STATE_READY
        assert status.hf_home == "/override/hf"

    def test_explicit_cfg_matching_managed_wins(self, tmp_path):
        run_dir = lr.runtime_dir()
        python = run_dir / "venv" / "bin" / "python"
        owned_link(python, tmp_path)
        model = run_dir / "hf-cache" / "model"
        model.mkdir(parents=True)
        _write_json(run_dir / ".slm-managed", {
            "python": str(python), "model_path": str(model), "verified": True,
        })
        cfg = SimpleNamespace(sufficiency_python=str(python), sufficiency_model=str(model),
                               sufficiency_hf_home="")
        status = lr.detect(cfg)
        assert status.state == lr.STATE_READY
        assert status.managed is True

    def test_explicit_cfg_with_no_matching_record_is_failed(self, tmp_path):
        cfg = SimpleNamespace(
            sufficiency_python=str(tmp_path / "nowhere" / "python"),
            sufficiency_model=str(tmp_path / "nowhere-model"),
            sufficiency_hf_home="",
        )
        status = lr.detect(cfg)
        assert status.state == lr.STATE_FAILED
        # An install made elsewhere is checked, never "set up again" (a download).
        assert status.step == "Needs a check — choose Check this install."
        assert status.action == lr.ACTION_CHECK

    def test_cfg_with_default_repo_id_is_not_treated_as_explicit(self, tmp_path):
        """sufficiency_model defaults to the bare repo id ('aac6fef/laya-mlx'),
        not a filesystem path — that must never be read as "explicit"."""
        python = tmp_path / "ext-venv" / "bin" / "python"
        owned_link(python, tmp_path)
        model = tmp_path / "ext-model"
        model.mkdir()
        run_dir = lr.runtime_dir()
        _write_json(run_dir / "adopted.json", {
            "python": str(python), "model_path": str(model), "verified": True,
        })
        cfg = SimpleNamespace(sufficiency_python="", sufficiency_model="aac6fef/laya-mlx",
                               sufficiency_hf_home="")
        status = lr.detect(cfg)
        # Falls through to adopted.json, exactly as if cfg had been None.
        assert status.state == lr.STATE_READY
        assert status.python == str(python)

    def test_real_retrieval_config_accepted(self):
        from superlocalmemory.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        status = lr.detect(cfg)
        assert status.state == lr.STATE_NOT_INSTALLED

    def test_installing_while_job_runs_reports_progress(self):
        job = lr.LayaInstallJob.instance()
        job._status = lr.LayaRuntimeStatus(
            state=lr.STATE_INSTALLING, progress=0.4, step="Downloading")
        job._running = True
        status = lr.detect()
        assert status.state == lr.STATE_INSTALLING
        assert status.progress == 0.4

    def test_detect_never_raises_on_a_corrupt_marker(self):
        run_dir = lr.runtime_dir()
        run_dir.mkdir(parents=True)
        (run_dir / ".slm-managed").write_text("{not json", encoding="utf-8")
        status = lr.detect()
        assert status.state == lr.STATE_NOT_INSTALLED


# ---------------------------------------------------------------------------
# install() — resumability, locking, disk space, plain-language failures
# ---------------------------------------------------------------------------

class TestInstall:
    def _patch_happy_steps(self, monkeypatch, *, weights_ok=True):
        monkeypatch.setattr(lr, "_check_disk_space", lambda p: True)
        monkeypatch.setattr(lr, "_create_venv", lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(lr, "_pip_install", lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(
            lr, "_download_weights",
            lambda *a, **k: (weights_ok, "" if weights_ok else "other", ""),
        )
        monkeypatch.setattr(lr, "verify", lambda *a, **k: (True, "Looks good."))

    def test_happy_path_writes_the_managed_marker_and_returns_ready(self, monkeypatch):
        self._patch_happy_steps(monkeypatch)
        result = lr.install()
        assert result.state == lr.STATE_READY
        assert result.managed is True
        marker = json.loads((lr.runtime_dir() / ".slm-managed").read_text(encoding="utf-8"))
        assert marker["verified"] is True
        assert marker["requirement"] == lr.LAYA_MLX_REQUIREMENT
        assert marker["revision"] == lr.LAYA_MODEL_REVISION

    def test_reports_progress_through_every_step(self, monkeypatch):
        self._patch_happy_steps(monkeypatch)
        seen = []
        lr.install(progress=lambda f, s: seen.append((f, s)))
        fractions = [f for f, _ in seen]
        assert fractions[0] == 0.0
        assert fractions[-1] == 1.0
        assert fractions == sorted(fractions)

    def test_unsupported_platform_never_touches_disk(self, monkeypatch):
        monkeypatch.setattr(lr, "_apple_silicon", lambda: False)
        called = []
        monkeypatch.setattr(lr, "_check_disk_space", lambda p: called.append(1) or True)
        result = lr.install()
        assert result.state == lr.STATE_UNSUPPORTED
        assert called == []

    def test_disk_space_failure_short_circuits_before_any_step(self, monkeypatch):
        monkeypatch.setattr(lr, "_check_disk_space", lambda p: False)
        called = []
        monkeypatch.setattr(lr, "_create_venv", lambda *a, **k: called.append(1) or (True, "", ""))
        result = lr.install()
        assert result.state == lr.STATE_FAILED
        assert result.error == "Not enough free disk space (needs about 1.5 GB)"
        assert called == []

    def test_network_failure_during_download_is_plain_language(self, monkeypatch):
        monkeypatch.setattr(lr, "_check_disk_space", lambda p: True)
        monkeypatch.setattr(lr, "_create_venv", lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(lr, "_pip_install", lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(
            lr, "_download_weights",
            lambda *a, **k: (False, "network", "ConnectionError: [Errno 61] Connection refused"),
        )
        result = lr.install()
        assert result.state == lr.STATE_FAILED
        assert result.error == lr._NETWORK_MESSAGE
        assert "Errno" not in result.error
        assert "ConnectionError" not in result.error

    def test_resumable_install_skips_steps_already_done(self, monkeypatch, tmp_path):
        run_dir = lr.runtime_dir()
        venv_python = run_dir / "venv" / "bin" / "python"
        owned_link(venv_python, tmp_path)
        _write_json(run_dir / ".install-steps.json", {"venv": True, "pip": True})

        venv_calls, pip_calls, weight_calls = [], [], []
        monkeypatch.setattr(lr, "_check_disk_space", lambda p: True)
        monkeypatch.setattr(
            lr, "_create_venv", lambda *a, **k: venv_calls.append(1) or (True, "", ""))
        monkeypatch.setattr(
            lr, "_pip_install", lambda *a, **k: pip_calls.append(1) or (True, "", ""))
        monkeypatch.setattr(
            lr, "_download_weights",
            lambda *a, **k: weight_calls.append(1) or (True, "", ""),
        )
        monkeypatch.setattr(lr, "verify", lambda *a, **k: (True, "ok"))

        result = lr.install()
        assert result.state == lr.STATE_READY
        assert venv_calls == []
        assert pip_calls == []
        assert weight_calls == [1]

    def test_redoes_a_step_whose_marker_lied(self, monkeypatch):
        """Stamp says venv is done, but the python binary is gone — redo it."""
        run_dir = lr.runtime_dir()
        _write_json(run_dir / ".install-steps.json", {"venv": True})

        self._patch_happy_steps(monkeypatch)
        venv_calls = []
        monkeypatch.setattr(
            lr, "_create_venv", lambda *a, **k: venv_calls.append(1) or (True, "", ""))

        result = lr.install()
        assert result.state == lr.STATE_READY
        assert venv_calls == [1]

    def test_verify_failure_still_writes_marker_as_unverified(self, monkeypatch):
        monkeypatch.setattr(lr, "_check_disk_space", lambda p: True)
        monkeypatch.setattr(lr, "_create_venv", lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(lr, "_pip_install", lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(lr, "_download_weights", lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(lr, "verify", lambda *a, **k: (False, "The check did not pass."))

        result = lr.install()
        assert result.state == lr.STATE_FAILED
        assert result.step == "Needs repair — choose Repair."
        marker = json.loads((lr.runtime_dir() / ".slm-managed").read_text(encoding="utf-8"))
        assert marker["verified"] is False

    @pytest.mark.skipif(
        fcntl is None, reason="fcntl is POSIX-only; this test exercises flock() directly",
    )
    def test_install_refuses_when_lock_already_held_by_another_process(self, monkeypatch):
        self._patch_happy_steps(monkeypatch)
        venv_calls = []
        monkeypatch.setattr(
            lr, "_create_venv", lambda *a, **k: venv_calls.append(1) or (True, "", ""))

        run_dir = lr.runtime_dir()
        run_dir.mkdir(parents=True)
        lock_path = run_dir / "install.lock"
        external_fh = open(lock_path, "w", encoding="utf-8")
        fcntl.flock(external_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = lr.install()
            assert result.state == lr.STATE_FAILED
            assert "already running" in result.error
            assert venv_calls == []
        finally:
            fcntl.flock(external_fh.fileno(), fcntl.LOCK_UN)
            external_fh.close()

    @pytest.mark.skipif(
        fcntl is None,
        reason="_acquire_install_lock() is a deliberate no-op without fcntl "
               "(Laya is Apple-Silicon only); nothing POSIX-specific to prove here",
    )
    def test_lock_primitive_refuses_a_second_holder(self, tmp_path):
        lock_path = tmp_path / "install.lock"
        fh1 = lr._acquire_install_lock(lock_path)
        assert fh1 is not None
        fh2 = lr._acquire_install_lock(lock_path)
        assert fh2 is None
        lr._release_install_lock(fh1)
        fh3 = lr._acquire_install_lock(lock_path)
        assert fh3 is not None
        lr._release_install_lock(fh3)


# ---------------------------------------------------------------------------
# LayaInstallJob — single-flight, status snapshot
# ---------------------------------------------------------------------------

class TestLayaInstallJob:
    def test_second_start_is_refused_while_the_first_runs(self, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def _blocking_install(*, progress=None, slm_home=None):
            started.set()
            release.wait(timeout=5)
            return lr.LayaRuntimeStatus(state=lr.STATE_READY)

        monkeypatch.setattr(lr, "install", _blocking_install)
        job = lr.LayaInstallJob.instance()

        assert job.start() is True
        assert started.wait(timeout=2)
        assert job.start() is False  # refused: an install is already running

        release.set()
        job._thread.join(timeout=5)
        assert job.status().state == lr.STATE_READY

    def test_a_crashing_install_is_reported_not_raised(self, monkeypatch):
        def _boom(*, progress=None, slm_home=None):
            raise RuntimeError("disk exploded")

        monkeypatch.setattr(lr, "install", _boom)
        job = lr.LayaInstallJob.instance()
        assert job.start() is True
        job._thread.join(timeout=5)
        status = job.status()
        assert status.state == lr.STATE_FAILED
        assert isinstance(status.error, str) and status.error  # reported, not raised

    def test_instance_is_a_singleton(self):
        assert lr.LayaInstallJob.instance() is lr.LayaInstallJob.instance()


# ---------------------------------------------------------------------------
# adopt() — never touches the adopted files
# ---------------------------------------------------------------------------

class TestAdopt:
    def _venv(self, tmp_path, name="venv"):
        # A real interpreter is required: verify() actually spawns it to run
        # the fake worker script. A link to an owned launcher keeps the
        # "venv/bin/python" shape adopt() expects without a real venv.
        return owned_link(tmp_path / name / "bin" / "python", tmp_path)

    def test_refuses_a_missing_python(self, tmp_path):
        result = lr.adopt(str(tmp_path / "nope" / "python"), "")
        assert result.state == lr.STATE_FAILED

    def test_refuses_a_temp_location_for_real(self, tmp_path):
        """tmp_path itself lives under the OS temp root — no monkeypatch needed."""
        python = self._venv(tmp_path)
        result = lr.adopt(str(python), "", str(tmp_path))
        assert result.state == lr.STATE_FAILED
        assert "temporary" in result.error

    def test_records_a_verified_install(self, tmp_path, monkeypatch, fake_worker):
        monkeypatch.setattr(lr, "_is_in_temp_dir", lambda p: False)
        monkeypatch.setenv("FAKE_LAYA_MODE", "pass")
        python = self._venv(tmp_path)
        model = tmp_path / "model-snap"
        model.mkdir()
        (model / "weights.bin").write_text("fake", encoding="utf-8")

        result = lr.adopt(str(python), "", str(model))
        assert result.state == lr.STATE_READY
        adopted = json.loads((lr.runtime_dir() / "adopted.json").read_text(encoding="utf-8"))
        assert adopted["verified"] is True
        assert adopted["python"] == str(python)
        assert adopted["model_path"] == str(model)

    def test_records_unverified_when_the_canary_fails(self, tmp_path, monkeypatch, fake_worker):
        monkeypatch.setattr(lr, "_is_in_temp_dir", lambda p: False)
        monkeypatch.setenv("FAKE_LAYA_MODE", "fail_values")
        python = self._venv(tmp_path)
        model = tmp_path / "model-snap"
        model.mkdir()

        result = lr.adopt(str(python), "", str(model))
        assert result.state == lr.STATE_FAILED
        assert result.step == "Needs a check — choose Check this install."
        adopted = json.loads((lr.runtime_dir() / "adopted.json").read_text(encoding="utf-8"))
        assert adopted["verified"] is False

    def test_finds_the_pinned_snapshot_when_model_path_is_empty(
        self, tmp_path, monkeypatch, fake_worker,
    ):
        monkeypatch.setattr(lr, "_is_in_temp_dir", lambda p: False)
        monkeypatch.setenv("FAKE_LAYA_MODE", "pass")
        python = self._venv(tmp_path)
        hf_home = tmp_path / "hf-home"
        snap = hf_home / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / lr.LAYA_MODEL_REVISION
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")

        result = lr.adopt(str(python), str(hf_home))
        assert result.state == lr.STATE_READY
        assert result.model_path == str(snap)

    def test_falls_back_to_refs_main(self, tmp_path, monkeypatch, fake_worker):
        monkeypatch.setattr(lr, "_is_in_temp_dir", lambda p: False)
        monkeypatch.setenv("FAKE_LAYA_MODE", "pass")
        python = self._venv(tmp_path)
        hf_home = tmp_path / "hf-home"
        model_dir = hf_home / "hub" / "models--aac6fef--laya-mlx"
        commit = "deadbeef0000000000000000000000000000000"
        (model_dir / "snapshots" / commit).mkdir(parents=True)
        (model_dir / "refs").mkdir(parents=True)
        (model_dir / "refs" / "main").write_text(commit, encoding="utf-8")

        result = lr.adopt(str(python), str(hf_home))
        assert result.state == lr.STATE_READY
        assert result.model_path == str(model_dir / "snapshots" / commit)

    def test_fails_when_no_snapshot_can_be_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lr, "_is_in_temp_dir", lambda p: False)
        python = self._venv(tmp_path)
        hf_home = tmp_path / "hf-home"
        hf_home.mkdir()
        result = lr.adopt(str(python), str(hf_home))
        assert result.state == lr.STATE_FAILED

    def test_never_modifies_the_adopted_files(self, tmp_path, monkeypatch, fake_worker):
        monkeypatch.setattr(lr, "_is_in_temp_dir", lambda p: False)
        monkeypatch.setenv("FAKE_LAYA_MODE", "pass")
        python = self._venv(tmp_path, name="venv-checked")
        model = tmp_path / "model-checked"
        model.mkdir()
        weights = model / "weights.bin"
        weights.write_bytes(b"original-weights-bytes-do-not-touch")

        before_python_hash = hashlib.sha256(python.read_bytes()).hexdigest()
        before_weights_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        before_python_mtime = python.stat().st_mtime_ns
        before_weights_mtime = weights.stat().st_mtime_ns

        result = lr.adopt(str(python), "", str(model))
        assert result.state == lr.STATE_READY

        assert hashlib.sha256(python.read_bytes()).hexdigest() == before_python_hash
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == before_weights_hash
        assert python.stat().st_mtime_ns == before_python_mtime
        assert weights.stat().st_mtime_ns == before_weights_mtime


# ---------------------------------------------------------------------------
# remove() — only the exact managed folder, with four independent guards
# ---------------------------------------------------------------------------

class TestRemove:
    def _managed_install(self, tmp_path) -> Path:
        run_dir = tmp_path / "runtimes" / "laya"
        run_dir.mkdir(parents=True)
        (run_dir / ".slm-managed").write_text("{}", encoding="utf-8")
        (run_dir / "venv").mkdir()
        (run_dir / "adopted.json").write_text("{}", encoding="utf-8")
        return run_dir

    def test_happy_path_removes_the_managed_install(self, tmp_path):
        run_dir = self._managed_install(tmp_path)
        result = lr.remove(slm_home=tmp_path)
        assert result.state == lr.STATE_NOT_INSTALLED
        assert not run_dir.exists()

    def test_guard_refuses_without_the_marker(self, tmp_path):
        run_dir = tmp_path / "runtimes" / "laya"
        run_dir.mkdir(parents=True)
        (run_dir / "venv").mkdir()
        result = lr.remove(slm_home=tmp_path)
        assert result.state == lr.STATE_FAILED
        assert run_dir.exists()  # nothing was touched

    def test_guard_refuses_a_path_outside_the_data_dir(self, tmp_path):
        outside_root = Path(tempfile.mkdtemp())
        try:
            real_dir = outside_root / "runtimes" / "laya"
            real_dir.mkdir(parents=True)
            (real_dir / ".slm-managed").write_text("{}", encoding="utf-8")
            (tmp_path / "runtimes").symlink_to(outside_root / "runtimes")

            result = lr.remove(slm_home=tmp_path)
            assert result.state == lr.STATE_FAILED
            assert "outside the data directory" in result.error
            assert real_dir.exists()  # never touched
        finally:
            import shutil as _shutil
            _shutil.rmtree(outside_root, ignore_errors=True)

    def test_guard_refuses_an_unexpected_leaf_name(self, tmp_path):
        (tmp_path / "runtimes").mkdir(parents=True)
        other = tmp_path / "other_name"
        other.mkdir()
        (other / ".slm-managed").write_text("{}", encoding="utf-8")
        (tmp_path / "runtimes" / "laya").symlink_to(other)

        result = lr.remove(slm_home=tmp_path)
        assert result.state == lr.STATE_FAILED
        assert "unexpected install path" in result.error
        assert other.exists()

    def test_guard_refuses_when_a_parent_is_a_symlink(self, tmp_path):
        actual = tmp_path / "actual" / "runtimes" / "laya"
        actual.mkdir(parents=True)
        (actual / ".slm-managed").write_text("{}", encoding="utf-8")
        (tmp_path / "runtimes").symlink_to(tmp_path / "actual" / "runtimes")

        result = lr.remove(slm_home=tmp_path)
        assert result.state == lr.STATE_FAILED
        assert "symlink" in result.error
        assert actual.exists()


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

class TestSmallHelpers:
    def test_runtime_dir_honours_explicit_slm_home(self, tmp_path):
        assert lr.runtime_dir(tmp_path) == tmp_path / "runtimes" / "laya"

    def test_runtime_dir_defaults_to_canonical_data_root(self, tmp_path):
        # The autouse SLM_DATA_DIR fixture already points canonical_data_root()
        # at this test's tmp_path.
        assert lr.runtime_dir() == tmp_path / "runtimes" / "laya"

    def test_is_in_temp_dir_true_for_the_real_temp_root(self):
        assert lr._is_in_temp_dir(Path(tempfile.gettempdir()) / "x" / "y") is True

    def test_is_in_temp_dir_false_for_a_normal_home_path(self):
        assert lr._is_in_temp_dir(Path("/Users/someone/.local/share/laya-venv")) is False

    def test_folder_size_sums_real_files(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x" * 10)
        (tmp_path / "b.bin").write_bytes(b"y" * 20)
        assert lr._folder_size(tmp_path) == 30

    def test_folder_size_is_zero_for_a_missing_folder(self, tmp_path):
        assert lr._folder_size(tmp_path / "nope") == 0

    def test_classify_subprocess_error_recognises_network_hints(self):
        assert lr._classify_subprocess_error("ConnectionError: timed out") == "network"
        assert lr._classify_subprocess_error("SyntaxError: invalid syntax") == "other"

