"""The managed environment, driven by a fake runner and a local model source."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from superlocalmemory.runtimes import managed_env as me
from superlocalmemory.runtimes.managed_env import (
    EnvSpec, HuggingFaceSource, LocalDirSource, ManagedEnv,
)

LOCK_TEXT = "# --index-url https://example.invalid/cpu\nfoo==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"


class Runner:
    """Records calls; creates a fake venv python on `venv`, fails `pip` on demand."""

    def __init__(self, fail_pip=False, fail_kind=""):
        self.calls = []
        self.fail_pip = fail_pip

    def __call__(self, cmd, *, timeout_s, env, classify):
        self.calls.append(list(cmd))
        if "venv" in cmd:
            py = Path(cmd[-1]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
            return True, "", ""
        if "pip" in cmd and self.fail_pip:
            return False, classify("Connection refused by host"), "Connection refused /secret/path"
        return True, "", ""

    def count(self, word):
        return sum(1 for c in self.calls if word in c)


@pytest.fixture()
def src(tmp_path):
    d = tmp_path / "weights_src"
    d.mkdir()
    (d / "model.bin").write_bytes(b"w" * 10)
    return d


@pytest.fixture()
def lock(tmp_path, monkeypatch):
    d = tmp_path / "locks"
    d.mkdir()
    path = d / me.lock_name()
    path.write_text(LOCK_TEXT)
    monkeypatch.setattr(me, "LOCKS_DIR", d)
    return path


def make(tmp_path, src, runner=None, canary=lambda p: True, source=None, **kw):
    spec = EnvSpec(name="t", requirements=("foo==1.0",),
                   model_source=source or LocalDirSource(src, model_id="m", revision="r1"),
                   min_free_disk_bytes=1, min_ram_bytes_warn=1, canary=canary,
                   expected_download_bytes=10, **kw)
    return ManagedEnv(spec, root=tmp_path / "env", runner=runner or Runner())


def test_full_install_reaches_ready(tmp_path, src, lock):
    runner, seen = Runner(), []
    env = make(tmp_path, src, runner)
    st = env.install(on_progress=lambda f, s: seen.append(f))
    assert st.state == "ready" and st.progress == 1.0
    assert seen == sorted(seen) and seen
    pip = [c for c in runner.calls if "pip" in c][0]
    assert "--require-hashes" in pip and "--no-deps" in pip and str(lock) in pip
    assert "--index-url" in pip
    assert (env.weights_dir() / "model.bin").exists()
    assert env.status().state == "ready"
    assert not list((tmp_path / "env").glob("*.tmp*"))


def test_pip_failure_then_resume_skips_venv(tmp_path, src, lock):
    runner = Runner(fail_pip=True)
    env = make(tmp_path, src, runner)
    st = env.install()
    assert st.state == "failed" and st.error_kind == "network"
    assert "/secret" not in st.step and "Connection" not in st.step
    runner.fail_pip = False
    assert env.install().state == "ready"
    assert runner.count("venv") == 1


def test_cancel_before_start_fails_cancelled(tmp_path, src, lock):
    cancel = threading.Event()
    cancel.set()
    st = make(tmp_path, src).install(cancel=cancel)
    assert st.state == "failed" and st.error_kind == "cancelled"


def test_canary_false_fails(tmp_path, src, lock):
    st = make(tmp_path, src, canary=lambda p: False).install()
    assert st.state == "failed" and st.error_kind == "canary"


def test_canary_exception_fails_without_raising(tmp_path, src, lock):
    def boom(p):
        raise RuntimeError("x")

    assert make(tmp_path, src, canary=boom).install().error_kind == "canary"


def test_second_concurrent_install_does_nothing(tmp_path, src, lock):
    runner = Runner()
    env = make(tmp_path, src, runner)
    other = make(tmp_path, src, Runner())
    from superlocalmemory.infra.instance_lock import InstanceLock

    held = InstanceLock(tmp_path / "env" / "install.lock")
    assert held.try_acquire()
    try:
        st = other.install()
    finally:
        held.release()
    assert other._runner.calls == []
    assert st.state != "ready"


def test_stale_installing_with_dead_pid_is_failed(tmp_path, src, lock):
    env = make(tmp_path, src)
    (tmp_path / "env").mkdir(parents=True)
    (tmp_path / "env" / "state.json").write_text(json.dumps(
        {"state": "installing", "progress": 0.3, "step": "x", "pid": 2 ** 22 + 12345}))
    st = env.status()
    assert st.state == "failed" and "stopped before it finished" in st.step


def test_remove_keeps_or_drops_weights(tmp_path, src, lock):
    env = make(tmp_path, src)
    env.install()
    env.remove(keep_weights=True)
    assert env.status().state == "not_installed"
    assert (env.weights_dir() / "model.bin").exists() and not env.python().exists()
    env.install()
    env.remove(keep_weights=False)
    assert not env.weights_dir().exists()


def test_unsupported_python(tmp_path, src, lock, monkeypatch):
    monkeypatch.setattr(me, "_python_supported", lambda: False)
    st = make(tmp_path, src).install()
    assert st.state == "unsupported"


def test_missing_lock_is_unsupported(tmp_path, src, monkeypatch):
    monkeypatch.setattr(me, "LOCKS_DIR", tmp_path / "nolocks")
    runner = Runner()
    st = make(tmp_path, src, runner).install()
    assert st.state == "unsupported" and "package list" in st.step
    assert runner.calls == []


def test_intel_mac_is_unsupported(tmp_path, src, lock, monkeypatch):
    monkeypatch.setattr(me, "_platform_tag", lambda: "darwin-x86_64")
    assert make(tmp_path, src).install().state == "unsupported"


def test_huggingface_without_revision_is_unpinned(tmp_path, src, lock):
    runner = Runner()
    env = make(tmp_path, src, runner, source=HuggingFaceSource("org/model", ""))
    st = env.install()
    assert st.state == "failed" and st.error_kind == "unpinned"
    assert "not pinned" in st.step


def test_python_change_needs_repair(tmp_path, src, lock, monkeypatch):
    env = make(tmp_path, src)
    assert env.install().state == "ready"
    monkeypatch.setattr(me, "_base_python", lambda: ("/elsewhere/python3", "3.13.0"))
    st = env.status()
    assert st.state == "failed" and "Needs repair" in st.step


def test_precheck_reports_vector_extension(tmp_path, src):
    pre = make(tmp_path, src).precheck()
    for k in ("disk_ok", "free_bytes", "ram_bytes", "ram_warn", "python_ok", "python", "sqlite_vec_ok"):
        assert k in pre


def test_no_vector_extension_is_unsupported(tmp_path, src, lock, monkeypatch):
    monkeypatch.setattr(me, "_sqlite_vec_ok", lambda: False)
    st = make(tmp_path, src).install()
    assert st.state == "unsupported" and "vector extension" in st.step


def test_ram_threshold_is_7_5_gib():
    from superlocalmemory.runtimes.media_env import MEDIA_ENV

    assert MEDIA_ENV.min_ram_bytes_warn == int(7.5 * 1024 ** 3)
    assert MEDIA_ENV.model_source.revision == ""


def test_real_venv_and_isolated_script(tmp_path, src):
    """Real stdlib venv, no packages, no network; proves run_script is isolated."""
    spec = EnvSpec(name="real", requirements=(), model_source=LocalDirSource(src, "m", "r"),
                   min_free_disk_bytes=1, min_ram_bytes_warn=1)
    env = ManagedEnv(spec, root=tmp_path / "renv")
    assert env._create_venv()[0]
    script = tmp_path / "probe.py"
    script.write_text("import sys, os, json\n"
                      "print(json.dumps({'isolated': sys.flags.isolated, 'cwd': os.getcwd(), "
                      "'pp': os.environ.get('PYTHONPATH'), 'slm': 'superlocalmemory' in sys.modules, "
                      "'arg': sys.argv[1:]}))\n")
    r = env.run_script(script, ["x"], timeout_s=60,
                       extra_env={"PYTHONPATH": "/should/be/dropped"})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["isolated"] == 1 and out["pp"] is None and out["slm"] is False
    assert Path(out["cwd"]).resolve() == (tmp_path / "renv").resolve() and out["arg"] == ["x"]
