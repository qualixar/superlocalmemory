"""Choosing a folder with the computer's own dialog, and one-click suggestions."""

from __future__ import annotations

import os
import threading

import pytest

from superlocalmemory.sources import picker
from tests.test_sources.test_routes import client

PICK = "/api/v3/sources/pick-folder"
SUGG = "/api/v3/sources/suggestions"


class Runner:
    def __init__(self, code=0, out="/Users/me/Notes/\n", missing=()):
        self.code, self.out, self.missing, self.calls = code, out, missing, []

    def __call__(self, argv, timeout_s):
        self.calls.append((list(argv), timeout_s))
        if argv[0] in self.missing:
            raise FileNotFoundError(argv[0])
        return self.code, self.out


@pytest.fixture
def fake(monkeypatch):
    runner = Runner()
    real = picker.pick_folder
    monkeypatch.setattr(picker, "pick_folder", lambda: real(runner))
    return runner


def _platform(monkeypatch, name, tools=()):
    monkeypatch.setattr(picker.sys, "platform", name)
    monkeypatch.setattr(picker.shutil, "which", lambda t: f"/bin/{t}" if t in tools else None)


@pytest.fixture(autouse=True)
def allow(monkeypatch):
    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_permission", lambda *a, **k: None)


def test_macos_uses_osascript_and_returns_the_path(env, fake, monkeypatch):
    _platform(monkeypatch, "darwin", ("osascript",))
    r = client(env).post(PICK)
    assert r.status_code == 200 and r.json() == {"path": "/Users/me/Notes"}
    argv, timeout = fake.calls[0]
    assert argv[0] == "osascript" and "choose folder" in argv[2] and timeout == 120


def test_windows_uses_powershell(env, fake, monkeypatch):
    _platform(monkeypatch, "win32", ("powershell",))
    fake.out = "C:\\Users\\me\\Notes\r\n"
    r = client(env).post(PICK)
    assert r.json() == {"path": "C:\\Users\\me\\Notes"}
    assert fake.calls[0][0][0] == "powershell" and "FolderBrowserDialog" in fake.calls[0][0][-1]


def test_linux_prefers_zenity(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ("zenity", "kdialog"))
    client(env).post(PICK)
    assert [c[0][0] for c in fake.calls] == ["zenity"]
    assert fake.calls[0][0][1:3] == ["--file-selection", "--directory"]


def test_linux_falls_back_to_kdialog(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ("zenity", "kdialog"))
    fake.missing = ("zenity",)
    r = client(env).post(PICK)
    assert r.json() == {"path": "/Users/me/Notes"}
    assert [c[0][0] for c in fake.calls] == ["zenity", "kdialog"]


def test_linux_with_only_kdialog(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ("kdialog",))
    client(env).post(PICK)
    assert fake.calls[0][0][:2] == ["kdialog", "--getexistingdirectory"]


def test_no_dialog_tool_is_501_picker_unavailable(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ())
    r = client(env).post(PICK)
    assert r.status_code == 501 and r.json()["detail"]["code"] == "picker_unavailable"
    assert fake.calls == []


def test_cancel_is_not_an_error(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ("zenity",))
    fake.code, fake.out = 1, ""
    r = client(env).post(PICK)
    assert r.status_code == 200 and r.json() == {"cancelled": True}


def test_an_empty_answer_is_a_cancel(env, fake, monkeypatch):
    _platform(monkeypatch, "win32", ("powershell",))
    fake.out = "\r\n"
    assert client(env).post(PICK).json() == {"cancelled": True}


def test_a_second_picker_while_one_is_open_is_409(env, monkeypatch):
    _platform(monkeypatch, "linux", ("zenity",))
    started, release = threading.Event(), threading.Event()

    def slow(argv, timeout_s):
        started.set()
        release.wait(5)
        return 0, "/a\n"

    first = {}
    t = threading.Thread(target=lambda: first.update(r=picker.pick_folder(slow)))
    t.start()
    assert started.wait(5)
    try:
        r = client(env).post(PICK)
        assert r.status_code == 409 and r.json()["detail"]["code"] == "picker_busy"
    finally:
        release.set()
        t.join()
    assert first["r"] == "/a"
    monkeypatch.setattr(picker, "_run", lambda a, t: (0, "/b\n"))
    assert client(env).post(PICK).json() == {"path": "/b"}


def test_remote_access_refuses_both_routes(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ("zenity",))
    env.remote = True
    c = client(env)
    for r in (c.post(PICK), c.get(SUGG)):
        assert r.status_code == 409 and r.json()["detail"]["code"] == "remote_access_on"
    assert fake.calls == []


def test_the_pick_needs_credentials_and_a_local_caller(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ("zenity",))
    assert client(env, actor=None).post(PICK).status_code in (401, 403)
    assert client(env, peer=("10.1.2.3", 50000)).post(PICK).status_code == 403
    assert client(env, peer=("10.1.2.3", 50000)).get(SUGG).status_code == 403
    assert fake.calls == []


def test_a_picked_path_is_only_returned_never_connected(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ("zenity",))
    fake.out = str(env.root) + "\n"
    c = client(env)
    assert c.post(PICK).json() == {"path": str(env.root)}
    assert c.get("/api/v3/sources").json() == {"sources": []}


def test_no_text_from_the_caller_reaches_the_command(env, fake, monkeypatch):
    _platform(monkeypatch, "linux", ("zenity",))
    client(env).post(PICK, json={"path": "; rm -rf /", "title": "x"})
    assert all("rm" not in a for a in fake.calls[0][0])


# ------------------------------------------------------------------ suggestions
@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    return h


def _vault(path):
    (path / ".obsidian").mkdir(parents=True)
    return path


def test_suggestions_find_vaults_documents_and_desktop(env, home):
    _vault(home / "Notes")
    _vault(home / "Work" / "deep" / "Vault")
    (home / "Documents").mkdir()
    (home / "Desktop").mkdir()
    (home / "Documents" / "secret.txt").write_text("x")
    body = client(env).get(SUGG).json()["suggestions"]
    paths = [s["path"] for s in body]
    assert paths == [str(home / "Notes"), str(home / "Work" / "deep" / "Vault"),
                     str(home / "Documents"), str(home / "Desktop")]
    assert [s["kind"] for s in body] == ["obsidian", "obsidian", "documents", "desktop"]
    assert not any("secret" in p for p in paths)


def test_missing_documents_and_desktop_are_left_out(env, home):
    assert client(env).get(SUGG).json() == {"suggestions": []}


def test_the_scan_is_bounded_in_depth_and_skips_hidden_folders(env, home):
    _vault(home / "a" / "b" / "c" / "d")          # depth 4: too deep
    _vault(home / ".hidden" / "v")
    _vault(home / "a" / "b" / "c")                # depth 3: found
    paths = [s["path"] for s in client(env).get(SUGG).json()["suggestions"]]
    assert paths == [str(home / "a" / "b" / "c")]


def test_the_scan_visits_at_most_200_folders(env, home, monkeypatch):
    for i in range(300):
        (home / f"d{i:03d}").mkdir()
    _vault(home / "d299")
    seen = []
    real = picker._children
    monkeypatch.setattr(picker, "_children", lambda p: (seen.append(p), real(p))[1])
    assert client(env).get(SUGG).json() == {"suggestions": []}
    assert len(seen) == picker.MAX_DIRS_VISITED


def test_symlinked_folders_are_not_followed(env, home, tmp_path):
    _vault(tmp_path / "elsewhere")
    os.symlink(tmp_path / "elsewhere", home / "link")
    assert client(env).get(SUGG).json() == {"suggestions": []}


def test_the_data_folder_is_never_suggested(env, home, monkeypatch):
    _vault(env.data / "inner")                     # a vault inside SLM's own folder
    os.symlink(env.data, home / "Documents")      # a Documents that is the data folder
    monkeypatch.setattr(picker, "_find_vaults", lambda h: [str(env.data / "inner"), str(home / "ok")])
    (home / "ok").mkdir()
    paths = [s["path"] for s in client(env).get(SUGG).json()["suggestions"]]
    assert paths == [str(home / "ok")]
