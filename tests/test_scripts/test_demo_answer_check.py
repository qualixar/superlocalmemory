# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The marketing demo refuses unsafe folders, refuses without a real Laya, and
fails unless every run really says "I don't have that"."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "demo_answer_check.py"
_spec = importlib.util.spec_from_file_location("demo_answer_check", _SCRIPT)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

_FIXTURE = json.loads(demo.FIXTURE.read_text(encoding="utf-8"))


def test_fixture_is_synthetic_and_has_both_shots() -> None:
    assert len(_FIXTURE["memories"]) == 12
    assert all("Kestrel" in m for m in _FIXTURE["memories"])
    q = _FIXTURE["questions"]
    assert q["abstained"]["expect"] == {"status": "judged", "abstained": True}
    assert q["answered"]["expect"] == {"status": "judged", "abstained": False}
    assert _FIXTURE["ceiling_ms"] == 3000


def test_refuses_the_owners_store_and_its_parents(tmp_path) -> None:
    owner = tmp_path / "home" / ".superlocalmemory"
    owner.mkdir(parents=True)
    for bad in (owner, owner.parent, owner / "sub"):
        with pytest.raises(demo.DemoError) as err:
            demo.check_data_dir(str(bad), [owner.resolve()])
        assert err.value.code == 2


def test_refuses_a_non_empty_folder(tmp_path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(demo.DemoError) as err:
        demo.check_data_dir(str(tmp_path / "x"), [])
    assert err.value.code == 2
    assert (tmp_path / "x" / "keep.txt").exists()


def test_accepts_new_or_empty_folder_and_marks_ownership(tmp_path) -> None:
    path, created = demo.check_data_dir(str(tmp_path / "new"), [])
    assert path.is_dir() and created is False      # user-supplied: never deleted
    made, created = demo.check_data_dir(None, [])
    assert created is True and made.name.startswith("slm-answer-check-demo-")
    made.rmdir()


def _account_home(tmp_path, monkeypatch) -> Path:
    """A stand-in account home served by the passwd lookup, never the real one."""
    home = tmp_path / "account"
    (home / ".superlocalmemory").mkdir(parents=True)
    monkeypatch.setattr(
        demo.pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    # owner_roots() also resolves the current root from HOME and reads its
    # config.json; left alone, that read reached the guarded live root under
    # the parallel CI gate (v4.2.5, 14 Oct).
    monkeypatch.setenv("HOME", str(home))
    for name in ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME"):
        monkeypatch.delenv(name, raising=False)
    return home


#: The account lookup reads the POSIX account database. Windows has none, and
#: there the demo stops before any lookup (see the test below this marker).
_POSIX_ACCOUNTS = pytest.mark.skipif(
    demo.pwd is None, reason="no POSIX account database (pwd) on this platform")


def test_a_computer_without_an_account_database_exits_3_before_anything(
    tmp_path, monkeypatch, capsys,
) -> None:
    monkeypatch.setattr(demo, "pwd", None)
    touched = []
    monkeypatch.setattr(demo, "find_laya", lambda home: touched.append("laya"))
    monkeypatch.setattr(demo, "start_daemon", lambda *a: touched.append("daemon"))
    target = tmp_path / "d"
    assert demo.main(["--data-dir", str(target), "--once"]) == 3
    assert touched == [] and not target.exists()
    assert "Apple silicon" in capsys.readouterr().err


@_POSIX_ACCOUNTS
def test_owner_roots_include_the_account_home_not_just_home_env(
    tmp_path, monkeypatch,
) -> None:
    home = _account_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "override"))
    roots = demo.owner_roots()
    assert (home / ".superlocalmemory").resolve() in roots
    assert (tmp_path / "elsewhere" / ".superlocalmemory").resolve() in roots
    assert (tmp_path / "override").resolve() not in roots


@_POSIX_ACCOUNTS
def test_owner_roots_follow_the_account_config_redirect(tmp_path, monkeypatch) -> None:
    home = _account_home(tmp_path, monkeypatch)
    moved = tmp_path / "moved-store"
    (home / ".superlocalmemory" / "config.json").write_text(
        json.dumps({"base_dir": str(moved)}), encoding="utf-8",
    )
    roots = demo.owner_roots()
    assert moved.resolve() in roots
    assert (home / ".superlocalmemory").resolve() in roots
    with pytest.raises(demo.DemoError):
        demo.check_data_dir(str(moved / "demo"), roots)


def test_exits_3_without_a_ready_laya(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(demo, "owner_roots", lambda: [])
    monkeypatch.setattr(demo, "find_laya", lambda home: SimpleNamespace(state="not_installed"))
    started = []
    monkeypatch.setattr(demo, "start_daemon", lambda *a: started.append(a))
    code = demo.main(["--data-dir", str(tmp_path / "d"), "--once"])
    assert code == 3 and started == []
    assert "Set up the on-device check first" in capsys.readouterr().err


def _run(status="judged", abstained=True, total=1200.0):
    return {"status": status, "abstained": abstained, "total_ms": total,
            "answer_confidence": 0.2, "threshold": 0.5}


def test_verify_runs_accepts_three_identical_abstentions() -> None:
    expect = _FIXTURE["questions"]["abstained"]["expect"]
    assert demo.verify_runs("abstained", [_run()] * 3, expect, 3000) == []


@pytest.mark.parametrize("runs,needle", [
    ([_run(), _run(abstained=False), _run()], "abstained was False"),
    ([_run(), _run(status="warming", abstained=False), _run()], "status was 'warming'"),
    ([_run(), _run(total=3001.0), _run()], "over the 3000 ms limit"),
    ([_run(), _run(total=None), _run()], "None ms"),
])
def test_verify_runs_fails_loudly(runs, needle) -> None:
    expect = _FIXTURE["questions"]["abstained"]["expect"]
    problems = demo.verify_runs("abstained", runs, expect, 3000)
    assert problems and any(needle in p for p in problems), problems


def test_mismatch_exits_5_and_stops_the_daemon(tmp_path, monkeypatch) -> None:
    class Proc:
        terminated = False
        def poll(self): return None if not self.terminated else 0
        def terminate(self): self.terminated = True
        def wait(self, timeout=None): return 0
    proc = Proc()
    monkeypatch.setattr(demo, "owner_roots", lambda: [])
    monkeypatch.setattr(demo, "find_laya", lambda home: SimpleNamespace(state="ready"))
    monkeypatch.setattr(demo, "prepare_demo", lambda d, laya: None)
    monkeypatch.setattr(demo, "start_daemon", lambda d, p: proc)
    monkeypatch.setattr(demo, "seed", lambda *a: None)
    monkeypatch.setattr(demo, "warm", lambda *a: None)
    monkeypatch.setattr(demo, "ask", lambda port, q: _run(abstained=False))  # never abstains
    assert demo.main(["--once"]) == 5
    assert proc.terminated
